"""Fixed-split BERT warmup, session ranking and independent listwise calibration.

Only ``train_and_calibrate`` imports GPU libraries. Pure calibration functions
are available to CPU tests. Calibration rescales confidence, never candidate
ordering. A question/candidate-only ranker cannot identify indistinguishable
memories; its session-ranking performance is an empirical measurement.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import time


def _canonical_id(value):
    result = str(value)
    while result.endswith('_abs'):
        result = result[:-4]
    return result


def _question_key(value):
    return ' '.join(re.findall(r'\w+', value.casefold()))


def _check_group(margins, gold_index, temperature=1.):
    if (not margins or not 0 <= gold_index < len(margins)
            or not all(math.isfinite(m) for m in margins)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Require a finite nonempty margin group, valid gold index and positive finite temperature')


def listwise_probs(margins, temperature=1.):
    _check_group(margins, 0, temperature)
    highest = max(margins)
    # Subtract BEFORE division to avoid exponent overflow.
    weights = [math.exp((m-highest)/temperature) for m in margins]
    total = sum(weights)
    return [w/total for w in weights]


def listwise_nll(groups, temperature=1.):
    """Mean negative log probability of each group's gold session."""
    if not groups:
        raise ValueError('At least one calibration group is required')
    losses = []
    for group in groups:
        margins, gold = group['margins'], group['gold_index']
        _check_group(margins, gold, temperature)
        highest = max(margins)
        shifted = [(m-highest)/temperature for m in margins]
        losses.append(math.log(sum(math.exp(z) for z in shifted))-shifted[gold])
    return sum(losses)/len(losses)


def _wilson(k, n):
    z = 1.959963984540054
    p = k/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., center-half), min(1., center+half)]


def grouped_measures(groups, temperature=1.):
    """Multiclass probability metrics, one observation per query/session group.

    Brier uses the SUM of squared class-probability errors per query (range 0–2).
    ECE compares top-one confidence with top-one session selection accuracy.
    Tie breaking is lexicographic session ID, matching the routing evaluator.
    """
    if not groups:
        raise ValueError('Cannot measure an empty query bank')
    observations = []
    for group in groups:
        margins, gold = group['margins'], group['gold_index']
        _check_group(margins, gold, temperature)
        sids = group.get('sids', [str(i) for i in range(len(margins))])
        if len(sids) != len(margins) or len(set(sids)) != len(sids):
            raise ValueError('Candidate session IDs must be unique and match margins')
        # Select from margins, not rounded probabilities, to preserve ranking.
        predicted = min(range(len(margins)), key=lambda i: (-margins[i], sids[i]))
        probabilities = listwise_probs(margins, temperature)
        observations.append(dict(correct=int(predicted == gold), confidence=probabilities[predicted],
                                 brier=sum((p-int(i == gold))**2 for i, p in enumerate(probabilities)),
                                 tied=sum(m == margins[predicted] for m in margins) > 1))
    n = len(groups)
    k = sum(row['correct'] for row in observations)
    ece = 0.
    bins = []
    for b in range(10):
        subset = [r for r in observations if min(int(r['confidence']*10), 9) == b]
        confidence = sum(r['confidence'] for r in subset)/len(subset) if subset else None
        accuracy = sum(r['correct'] for r in subset)/len(subset) if subset else None
        if subset:
            ece += len(subset)/n*abs(confidence-accuracy)
        bins.append(dict(lower=b/10, upper=(b+1)/10, n=len(subset),
                         mean_confidence=confidence, accuracy=accuracy))
    return dict(n_queries=n, n_independent_sessions=len({g.get('qid', i) for i, g in enumerate(groups)}),
                bank_sizes=sorted({len(g['margins']) for g in groups}), temperature=temperature,
                nll=listwise_nll(groups, temperature),
                brier=sum(r['brier'] for r in observations)/n,
                brier_definition='Mean sum of squared class errors per query; range 0–2',
                ece_10_bins=ece, reliability_bins=bins,
                mean_top1_confidence=sum(r['confidence'] for r in observations)/n,
                top1_accuracy=k/n, top1_correct=k, top1_accuracy_ci95=_wilson(k, n),
                n_exact_ties=sum(r['tied'] for r in observations),
                ci_unit='One question from each independent source session within a variant')


def fit_listwise_temperature(groups):
    """Deterministic positive scalar fit minimizing calibration session NLL."""
    if not groups:
        raise ValueError('Calibration groups must be nonempty')
    for group in groups:
        _check_group(group['margins'], group['gold_index'])
    low, high = math.log(.05), math.log(20.)
    grid = [low+(high-low)*i/160 for i in range(161)]
    best = min(range(len(grid)), key=lambda i: listwise_nll(groups, math.exp(grid[i])))
    a, b = grid[max(0, best-1)], grid[min(160, best+1)]
    ratio = (math.sqrt(5)-1)/2
    c, d = b-ratio*(b-a), a+ratio*(b-a)
    for _ in range(80):
        if listwise_nll(groups, math.exp(c)) < listwise_nll(groups, math.exp(d)):
            b, d = d, c
            c = b-ratio*(b-a)
        else:
            a, c = c, d
            d = a+ratio*(b-a)
    candidates = [1., math.exp(grid[best]), math.exp((a+b)/2)]
    temperature = min(candidates, key=lambda t: listwise_nll(groups, t))
    return dict(temperature=temperature, bounds=[.05, 20.], fit_groups=len(groups),
                before=grouped_measures(groups, 1.), after=grouped_measures(groups, temperature),
                target='Probability of the correct source session within the complete candidate bank',
                objective='Mean listwise multiclass negative log likelihood on calibration exact queries only',
                ranking='Positive temperature preserves raw-margin candidate ordering',
                fit_metrics_scope='Calibration-fit diagnostics; independent test metrics are stored separately')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _data_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _save_json(out, name, value):
    path = out/name
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def _trial_metadata(trials):
    """Inspect split identifiers only; heldout candidate text remains unread."""
    qids, sessions, query_ids = set(), set(), set()
    for trial in trials:
        if trial['query_id'] in query_ids:
            raise ValueError('Duplicate query_id within a split')
        query_ids.add(trial['query_id'])
        qids.add(_canonical_id(trial['qid']))
        bank_sids = [c['sid'] for c in trial['candidates']]
        if len(set(bank_sids)) != len(bank_sids) or not bank_sids:
            raise ValueError('Each trial requires a nonempty bank of unique deltas')
        own = [c for c in trial['candidates'] if c['sid'] == trial['own_sid']]
        if len(own) != 1 or own[0]['source_session_id'] not in trial['answer_session_ids']:
            raise ValueError('own_sid must identify the single correct answer session')
        if sum(c['source_session_id'] in trial['answer_session_ids'] for c in trial['candidates']) != 1:
            raise ValueError('A trial bank must contain exactly one ground-truth source session')
        sessions.update(_canonical_id(c['source_session_id']) for c in trial['candidates'])
    return qids, sessions


def validate_protocol(train_trials, calibration_trials, test_trials, train_examples):
    splits = [_trial_metadata(rows) for rows in [train_trials, calibration_trials, test_trials]]
    for i in range(3):
        for j in range(i+1, 3):
            if splits[i][0] & splits[j][0] or splits[i][1] & splits[j][1]:
                raise ValueError('Training, calibration and test must be disjoint in qid and candidate source sessions')
    train_qids, train_sessions = splits[0]
    train_question_keys = {_question_key(t['question']) for t in train_trials}
    for example in train_examples:
        if (_canonical_id(example['qid']) not in train_qids
                or _canonical_id(example['source_session_id']) not in train_sessions
                or _question_key(example['question']) not in train_question_keys):
            raise ValueError('Answerability warmup example is outside the training split')
        if ('donor_qid' in example and (_canonical_id(example['donor_qid']) not in train_qids
                or _canonical_id(example['donor_session_id']) not in train_sessions)):
            raise ValueError('Answerability warmup donor is outside the training split')
    exact_counts = [sum(t['variant'] == 'exact' for t in rows)
                    for rows in [train_trials, calibration_trials, test_trials]]
    if exact_counts != [24, 8, 24] or [len(qids) for qids, _ in splits] != [24, 8, 24]:
        raise ValueError('This predeclared protocol requires 24 train / 8 calibration / 24 test sessions')
    for rows, expected in [(train_trials, 24), (calibration_trials, 8), (test_trials, 24)]:
        variants = Counter(t['variant'] for t in rows)
        if any(v not in {'exact', 'paraphrase'} for v in variants):
            raise ValueError('Supported query variants are exact and paraphrase')
        if any(n != expected for n in variants.values()):
            raise ValueError('Each present variant must contain one query per split source session')
        exact_qids = {_canonical_id(t['qid']) for t in rows if t['variant'] == 'exact'}
        for variant in variants:
            variant_qids = {_canonical_id(t['qid']) for t in rows if t['variant'] == variant}
            if len(variant_qids) != expected or variant_qids != exact_qids:
                raise ValueError('Each variant must use the same distinct source question groups')
        if any(len(t['candidates']) != expected for t in rows):
            raise ValueError('Train/calibration/test banks must contain 24/8/24 candidates respectively')
    if set(e['label'] for e in train_examples) != {0, 1}:
        raise ValueError('Warmup requires both answerability classes')
    return dict(train_qids=sorted(splits[0][0]), calibration_qids=sorted(splits[1][0]),
                test_qids=sorted(splits[2][0]), train_source_sessions=sorted(splits[0][1]),
                calibration_source_sessions=sorted(splits[1][1]), test_source_sessions=sorted(splits[2][1]),
                qid_overlap=False, candidate_source_session_overlap=False,
                train_examples=len(train_examples), train_exact_groups=24,
                calibration_exact_groups=8, test_exact_groups=24)


def _groups_for(trials, scores, method):
    groups = []
    for trial in trials:
        sids = [c['sid'] for c in trial['candidates']]
        groups.append(dict(query_id=trial['query_id'], qid=trial['qid'], sids=sids,
                           margins=[scores[trial['query_id']][method][sid] for sid in sids],
                           gold_index=sids.index(trial['own_sid'])))
    return groups


def train_and_calibrate(out, train_trials, calibration_trials, test_trials,
                        train_examples, init_checkpoint, seed=0):
    """Train fixed epochs, reload the final checkpoint, fit T, then score test.

    The caller must select physical GPU 1 before calling this function. Returns
    ``scores[query_id]['original'|'retrained'][sid]`` raw margins, calibration T,
    and independent ``heldout_calibration[variant]`` full-bank metrics.
    """
    out, init_checkpoint = Path(out), Path(init_checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out/'checkpoint'
    if checkpoint.exists() and any(checkpoint.iterdir()):
        raise ValueError('Final checkpoint output must be empty; existing artifacts are preserved')
    if checkpoint.resolve() == init_checkpoint.resolve() or init_checkpoint.resolve() in checkpoint.resolve().parents:
        raise ValueError('Do not overwrite the original checkpoint')
    audit = validate_protocol(train_trials, calibration_trials, test_trials, train_examples)
    audit.update(seed=seed, warmup_epochs=4, listwise_epochs=3,
                 warmup_learning_rate=2e-5, listwise_learning_rate=1e-5, warmup_batch_size=16,
                 model_selection='Fixed epochs; no model selection on calibration or test',
                 calibration_selection='Calibration exact queries only; test read for scoring after checkpoint and T frozen',
                 warmup_data_sha256=_data_sha256(train_examples),
                 training_trials_sha256=_data_sha256(train_trials),
                 initial_checkpoint_sha256={p.name: _sha256(p) for p in init_checkpoint.iterdir() if p.is_file()},
                 limitations='Question and candidate text cannot identify indistinguishable or equally compatible session facts.')
    _save_json(out, 'bert_provenance.json', audit)
    # Torch is already masked to physical GPU 1 by the caller.
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA device, selected as physical GPU 1 by caller')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(init_checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        init_checkpoint, local_files_only=True).float().to('cuda:0')
    if model.config.id2label.get(1) != 'relevant':
        raise ValueError('Expected original BERT positive class 1 to be relevant')
    max_tokens = model.config.max_position_embeddings

    def encode(question, candidate):
        value = tokenizer(question, candidate, truncation=False)
        if len(value['input_ids']) > max_tokens:
            raise ValueError('BERT context exceeded; candidate text is never silently truncated')
        return value

    def tensor_batch(encoded):
        return tokenizer.pad(encoded, padding=True, return_tensors='pt').to('cuda:0')

    warmup_inputs = [encode(e['question'], e.get('candidate', e.get('probe'))) for e in train_examples]
    warmup_labels = torch.tensor([e['label'] for e in train_examples], device='cuda:0', dtype=torch.long)
    class_counts = torch.bincount(warmup_labels, minlength=2).float()
    class_weights = len(train_examples)/(2*class_counts)
    warmup_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    history = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    for epoch in range(4):
        model.train()
        order = list(range(len(train_examples)))
        random.Random(seed+epoch).shuffle(order)
        summed_loss = 0.
        for start in range(0, len(order), 16):
            indices = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            logits = model(**tensor_batch([warmup_inputs[i] for i in indices])).logits.float()
            loss = warmup_loss(logits, warmup_labels[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite warmup training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            summed_loss += loss.detach().item()*len(indices)
        history.append(dict(stage='answerability_warmup', epoch=epoch+1,
                            train_loss=summed_loss/len(train_examples), n_examples=len(train_examples)))
        _save_json(out, 'train_history.json', history)
        print(json.dumps(history[-1]), flush=True)
    del optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
    train_exact = [t for t in train_trials if t['variant'] == 'exact']
    ranking_inputs = [[encode(t['question'], c['candidate']) for c in t['candidates']] for t in train_exact]
    targets = [next(i for i, c in enumerate(t['candidates']) if c['sid'] == t['own_sid']) for t in train_exact]
    for epoch in range(3):
        model.train()
        order = list(range(len(train_exact)))
        random.Random(seed+1000+epoch).shuffle(order)
        summed_loss = 0.
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            logits = model(**tensor_batch(ranking_inputs[index])).logits.float()
            margins = logits[:, 1]-logits[:, 0]
            target = torch.tensor([targets[index]], device='cuda:0', dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(margins.unsqueeze(0), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite listwise session-ranking training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            summed_loss += loss.detach().item()
        history.append(dict(stage='listwise_session_ranking', epoch=epoch+1,
                            train_loss=summed_loss/len(train_exact), n_groups=len(train_exact), bank_size=24))
        _save_json(out, 'train_history.json', history)
        print(json.dumps(history[-1]), flush=True)
    del optimizer, warmup_labels, class_weights, warmup_loss, loss, logits, margins, target
    expected_bytes = sum(p.numel()*2 for p in model.parameters())+16*1024**2
    storage_root = checkpoint.resolve() if checkpoint.exists() else checkpoint.parent.resolve()
    if shutil.disk_usage(storage_root).free < expected_bytes:
        raise RuntimeError('Not enough space for the single final half-precision BERT checkpoint')
    # Save once, then reload. All scored margins come from the saved artifact.
    model.eval().half()
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    del model
    torch.cuda.empty_cache()
    checkpoint_hashes = {p.name: _sha256(p) for p in checkpoint.iterdir() if p.is_file()}
    audit.update(final_checkpoint_sha256=checkpoint_hashes, final_checkpoint=str(checkpoint.resolve()),
                 final_checkpoint_dtype='float16 storage, float32 inference after explicit reload',
                 train_elapsed_seconds=time.monotonic()-started)
    _save_json(out, 'bert_provenance.json', audit)
    scores = {}

    def score_trials(scoring_model, trials, method):
        scoring_model.eval()
        with torch.inference_mode():
            for trial in trials:
                encoded = [encode(trial['question'], c['candidate']) for c in trial['candidates']]
                logits = scoring_model(**tensor_batch(encoded)).logits.float()
                margins = (logits[:, 1]-logits[:, 0]).cpu().tolist()
                if not all(math.isfinite(m) for m in margins):
                    raise ValueError('Nonfinite BERT score margin')
                scores.setdefault(trial['query_id'], {})[method] = {
                    c['sid']: m for c, m in zip(trial['candidates'], margins)}

    reloaded = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True).float().to('cuda:0').eval()
    score_trials(reloaded, calibration_trials, 'retrained')
    calibration_exact = [t for t in calibration_trials if t['variant'] == 'exact']
    fit_groups = _groups_for(calibration_exact, scores, 'retrained')
    calibration = fit_listwise_temperature(fit_groups)
    temperature = calibration['temperature']
    calibration.update(fit_split='calibration', fit_variant='exact', fit_bank_size=8,
                       calibration_qids=sorted({t['qid'] for t in calibration_exact}),
                       calibration_trials_sha256=_data_sha256(calibration_trials),
                       checkpoint_sha256=checkpoint_hashes,
                       frozen_before_test_scoring=True)
    _save_json(out, 'calibration.json', calibration)
    # Only now inspect heldout question/candidate text, after model and T freeze.
    seen_training_questions = {_question_key(t['question']) for t in train_trials}
    seen_calibration_questions = {_question_key(t['question']) for t in calibration_trials}
    seen_test_questions = {_question_key(t['question']) for t in test_trials}
    if (seen_training_questions & seen_calibration_questions or seen_training_questions & seen_test_questions
            or seen_calibration_questions & seen_test_questions):
        raise ValueError('Normalized question overlap among training/calibration/test splits')
    score_trials(reloaded, test_trials, 'retrained')
    del reloaded
    torch.cuda.empty_cache()
    original = AutoModelForSequenceClassification.from_pretrained(
        init_checkpoint, local_files_only=True).float().to('cuda:0').eval()
    score_trials(original, calibration_trials, 'original')
    score_trials(original, test_trials, 'original')
    del original
    torch.cuda.empty_cache()
    _save_json(out, 'bert_scores.json', scores)
    heldout = {}
    for variant in sorted({t['variant'] for t in test_trials}):
        subset = [t for t in test_trials if t['variant'] == variant]
        methods = {}
        for name, model_key, scalar in [('original', 'original', 1.),
                                        ('retrained_uncalibrated', 'retrained', 1.),
                                        ('retrained_calibrated', 'retrained', temperature)]:
            methods[name] = grouped_measures(_groups_for(subset, scores, model_key), scalar)
        heldout[variant] = dict(split='test', variant=variant, independent=True,
                               n_sessions=len({t['qid'] for t in subset}), methods=methods,
                               calibration_fit_used_test=False,
                               aggregation='Each variant reported separately; exact plus paraphrase are not 48 independent sessions')
    _save_json(out, 'heldout_calibration.json', heldout)
    fit_split_metrics = {}
    for variant in sorted({t['variant'] for t in calibration_trials}):
        subset = [t for t in calibration_trials if t['variant'] == variant]
        fit_split_metrics[variant] = {
            name: grouped_measures(_groups_for(subset, scores, model_key), scalar)
            for name, model_key, scalar in [('original', 'original', 1.),
                                             ('retrained_uncalibrated', 'retrained', 1.),
                                             ('retrained_calibrated', 'retrained', temperature)]}
    _save_json(out, 'calibration_split_metrics.json', fit_split_metrics)
    unchanged = {p.name: _sha256(p) for p in init_checkpoint.iterdir() if p.is_file()}
    if unchanged != audit['initial_checkpoint_sha256']:
        raise RuntimeError('Original synthetic BERT checkpoint changed')
    if {p.name: _sha256(p) for p in checkpoint.iterdir() if p.is_file()} != checkpoint_hashes:
        raise RuntimeError('Final frozen BERT checkpoint changed while scoring')
    audit.update(test_trials_sha256=_data_sha256(test_trials), normalized_question_overlap=False,
                 total_elapsed_seconds=time.monotonic()-started,
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                 original_checkpoint_unchanged=True, final_checkpoint_unchanged=True,
                 source='Original synthetic BERT initialization; no prior short-answer checkpoint reused')
    _save_json(out, 'bert_provenance.json', audit)
    return dict(checkpoint=str(checkpoint), scores=scores, calibration=calibration,
                heldout_calibration=heldout, calibration_split_metrics=fit_split_metrics,
                train_history=history, provenance=audit)
