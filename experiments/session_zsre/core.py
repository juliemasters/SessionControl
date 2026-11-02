"""Query-only activation routing; reference targets never enter this module."""
import torch


def capture_query_keys(model, tok, names, query):
    """The same last-question-token FFN inputs as the earlier session controller.

    Raw ZsRE prompts are used without adding Q:/A: so inputs match native WISE.
    Call only with base weights restored. No answer or case ID is accepted.
    """
    keys = {}
    handles = []
    for name in names:
        def capture(module, args, name=name):
            keys[name] = args[0][:, -1].detach().clone()
        handles.append(model.get_submodule(name.removesuffix('.weight')).register_forward_pre_hook(capture))
    try:
        encoded = tok(query, return_tensors='pt', truncation=False).to(model.device)
        with torch.inference_mode():
            model.model(**encoded, use_cache=False)
        if set(keys) != set(names):
            raise ValueError('Missing question-only activation keys')
        return keys
    finally:
        for handle in handles:
            handle.remove()


def select_candidate(bank, keys):
    """Argmax sum_l ||delta_l k_l||² / sum_l ||delta_l||F².

    Reuse the exact production factor scorer. Ties use stable memory ID order.
    All candidates are considered; no lexical retrieval or abstention is added.
    """
    from compact_session_memory import factor_scores
    scores = [(float(factor_scores(payload, keys)[0]), sid) for sid, payload in bank.items()]
    if not scores or any(not torch.isfinite(torch.tensor(score)) for score, _ in scores):
        raise ValueError('Missing or nonfinite routing scores')
    scores.sort(key=lambda pair: (-pair[0], pair[1]))
    rank = {sid: i + 1 for i, (_, sid) in enumerate(scores)}
    return scores[0][1], scores[0][0], scores[0][0] - scores[1][0] if len(scores) > 1 else None, rank


def raw_write_request(request, tok):
    """Exact question is the key; complete counterfactual answer plus EOS is value.

    ZsRE supplies no session context. Do not invent one or expose test paraphrases.
    One ungrounded-question template uses the existing grounded-value optimizer
    without a fabricated conversation. The ordinary KL prompt is unchanged.
    """
    # SentencePiece drops the initial separator when decoding a standalone target
    # prefix. Keep that separator explicitly in the value context, matching the
    # official evaluator's `prompt + ' ' + target`. The write key stays raw Q.
    return dict(case_id=request['case_id'], subject=request['prompt'], prompt='{}',
                target_new={'str': request['target_new'] + tok.eos_token}, target_true={'str': ''},
                value_context_templates=[['{} ']])


def nullspace(matrix, threshold):
    """Same empirical uncentered-token-moment projector as the earlier method."""
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    eigenvalues = singular.square() / matrix.shape[0]
    protected = torch.linalg.qr(vh[eigenvalues >= threshold].T, mode='reduced').Q
    result = torch.eye(matrix.shape[1], device=matrix.device) - protected @ protected.T
    residual = float((result @ protected).norm())
    if residual > 1e-3:
        raise ValueError('Nullspace projector numerical check failed')
    return result, dict(tokens=matrix.shape[0], width=matrix.shape[1], protected_rank=protected.shape[1],
                        null_dimension=matrix.shape[1]-protected.shape[1], protected_residual_norm=residual)
