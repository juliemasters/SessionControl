"""Freeze a matched comparison without changing any existing measured run."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from experiments.wise_paper.native import ROOT, digest, modules
from experiments.wise_paper.prepare import save

REFERENCE = ROOT / 'runs/wise-paper-zsre-mistral-v1'
DEPENDENCIES = ['compact_session_memory.py', 'AlphaEdit/__init__.py', 'AlphaEdit/compute_z.py',
                'AlphaEdit/compute_ks.py', 'AlphaEdit/grounded_value.py', 'AlphaEdit/AlphaEdit_hparams.py',
                'config/alphaedit-hparams.json', 'grounding_core.py', 'question_key.py',
                'rome/repr_tools.py', 'util/nethook.py', 'util/hparams.py',
                'experiments/wise_paper/native.py', 'experiments/wise_paper/prepare.py',
                'experiments/wise_paper/worker.py', 'experiment_utils.py']


def prepare(out,projection_from=None):
    if out.exists():
        raise FileExistsError('Use a fresh matched-study directory')
    reference = json.loads((REFERENCE / 'design.json').read_text())
    completion = json.loads((REFERENCE / 'completion.json').read_text())
    assert completion['completed'] and completion['edits_per_variant'] == 1000
    modules()  # Verify the unchanged official evaluator/source dependencies.
    requests = json.loads((REFERENCE / 'requests.json').read_text())
    assert len(requests) == 1000 and [r['case_id'] for r in requests] == list(range(1000))
    # Full CPU preflight of the exact strings the unchanged value optimizer will
    # tokenize. Validate all targets, not merely the first successful pilot item.
    from transformers import AutoTokenizer
    from .core import raw_write_request
    tok=AutoTokenizer.from_pretrained(reference['model'],local_files_only=True)
    for r in requests:
        request=raw_write_request(r,tok)
        text=' '+request['target_new']['str'].lstrip()
        target_ids=tok.encode(text)[1:]
        template=request['value_context_templates'][0][0].format(request['prompt'])+tok.decode(target_ids[:-1],clean_up_tokenization_spaces=False)
        assembled=tok.encode(template.format(request['subject']))
        expected=tok.encode(r['prompt'])+target_ids[:-1]
        if assembled!=expected:raise ValueError('Mistral value-token alignment failed at case '+str(r['case_id']))
    # Only training-locality prompts are allowed into projector construction.
    corpus = [r['loc_prompt'] for r in requests[:200]]
    out.mkdir(parents=True)
    (out / 'factors').mkdir()
    save(out / 'requests.json', requests)
    save(out / 'preservation_corpus.json', corpus)
    source_files = DEPENDENCIES + [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/session_zsre').glob('*.py')]
    d = dict(name='Activation-routed independent AlphaEdit on matched ZsRE', reference_run=str(REFERENCE),
             reference_files={name:digest(REFERENCE/name) for name in ['design.json','requests.json','baseline.json',
                             'summary.csv','merge/checkpoint-1000.json','retrieve/checkpoint-1000.json','model_manifest.json']},
             model=reference['model'], model_repository=reference['model_repository'], model_revision=reference['model_revision'],
             physical_gpu=1, seed=42, precision='float32', checkpoints=reference['checkpoints'], selected_count=1000,
             requests_sha256=digest(out/'requests.json'), corpus_sha256=digest(out/'preservation_corpus.json'),
             code_sha256={name:digest(ROOT/name) for name in source_files}, layers=[13,14,15],
             hyperparameters=json.loads((ROOT/'config/alphaedit-hparams.json').read_text()),
             unit='One official ZsRE edit/Q&A record per independent delta, NOT a LongMemEval conversation.',
             write='Exact raw src question -> complete alt answer plus EOS. Existing compact independent AlphaEdit writer, '
                   'value optimization, layers 13-15, L2=.1, 50 steps, lr=.1 and norm clamp unchanged. '
                   'No contextual conversation is available in ZsRE; use only the question template and normal KL preservation prompt.',
             adaptation='Mistral replaces Llama using the same compatible module paths and 32-layer loss index. '
                        'Omit the prior Q:/A: wrapper at both write and inference to match the native WISE raw prompt. '
                        'Use an explicit question/answer separator in the value context because SentencePiece target decoding drops its leading space. '
                        'All 1000 value token sequences are checked against direct question-ID plus target-ID concatenation before launch. '
                        'Preserve our prior EOS write target; WISE uses its own native target policy. This is a complete-system comparison.',
             preservation='Rebuild model-specific empirical nullspace projectors from all tokens of first 200 train-split locality prompts '
                          '(loc + loc_ans), same native WISE training data access. Uncentered second moment, thin SVD, threshold .02; '
                          'no held-out evaluation-locality questions, paraphrases, alt targets or edit questions used for projection. '
                          'This is the earlier small-corpus projector procedure, not full Wikipedia AlphaEdit statistics.',
             routing='Base-model last-question-token FFN inputs, normalized delta update energy across layers 13-15; '
                     'argmax over all first-N independent deltas, fixed memory-ID tie break. Select separately for rewrite, '
                     'paraphrase and locality questions before official teacher-forced evaluation. No target, case ID, '
                     'reference answer, original-question lookup, BERT, gate, oracle routing or retrieval enters selection.',
             evaluation='Same requests/order, base model/revision, FP32, seed, raw prompts, checkpoints, official unchanged evaluator '
                        'and unedited baseline as completed WISE study. At T evaluate all first-T original prompts, held-out '
                        'paraphrases and locality queries. Routing sees the query only, not teacher-forced answer tokens.',
             metrics=reference['metrics'], repetition='One frozen bank / one seed, matching one trajectory per WISE variant; '
                     'not ten independent trials. Primary metrics are NOT free-generated F1, exact match or TTFT.',
             extra_metrics='Query-only routing hit@1 and mean correct-delta rank on rewrite and paraphrase questions; '
                           'all-target-token indicators; compact retained disk bytes; peak allocated evaluation GPU memory. '
                           'Projector is a construction artifact, excluded from retained bank bytes. No TTFT comparison is claimed.',
             checkpoint_protocol=reference['checkpoint_protocol'], prepared_utc=datetime.now(timezone.utc).isoformat())
    if projection_from:
        source=projection_from.resolve()
        prior=json.loads((source/'manifest.json').read_text())
        if prior['corpus_sha256']!=d['corpus_sha256']:raise ValueError('Cannot reuse a different preservation corpus')
        prior_design=json.loads((source.parent/'design.json').read_text())
        for key in ['model','model_revision','layers']:
            if prior_design[key]!=d[key]:raise ValueError('Incompatible preserved projector: '+key)
        target=out/'projection';target.mkdir()
        for file,want in prior['files'].items():
            if digest(source/file)!=want:raise ValueError('Prior projector failed checksum')
            (target/file).symlink_to(source/file)
        save(target/'manifest.json',prior)
        d['projection_reuse']=dict(source=str(source),files=prior['files'],note='Identical base model, corpus and projector procedure; no edit delta reused.')
    save(out/'design.json',d)
    save(out/'token_alignment_validation.json',dict(exact=True,checked=1000,
         note='Every unchanged value-optimizer string has exactly the raw question IDs plus target prefix IDs. Explicit separator preserves SentencePiece boundaries.'))
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1))
    print('PREPARED',out,'1000 independent memories; checkpoints',d['checkpoints'],flush=True)


def check(out):
    d=json.loads((out/'design.json').read_text())
    for path,want in [(out/'requests.json',d['requests_sha256']), (out/'preservation_corpus.json',d['corpus_sha256'])]:
        if digest(path)!=want:raise ValueError('Frozen input changed: '+str(path))
    for name,want in d['code_sha256'].items():
        if digest(ROOT/name)!=want:raise ValueError('Execution source changed: '+name)
    for name,want in d['reference_files'].items():
        if digest(Path(d['reference_run'])/name)!=want:raise ValueError('WISE reference changed: '+name)
    modules()
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--projection-from',type=Path)
    a=p.parse_args();prepare(a.out.resolve(),a.projection_from)
