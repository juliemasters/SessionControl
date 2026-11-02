"""Freeze the six-method, ten-pass generated-answer ZsRE comparison."""
import argparse
import os
from pathlib import Path
import shutil
from .core import ROOT,VENDOR,SOURCE,OURS,ENCODER,SIZES,METHODS,read,save,digest,now,modules


def prepare(out):
    if out.exists():raise FileExistsError('Use a fresh generation-study output directory')
    assert read(SOURCE/'completion.json')['completed'] and read(OURS/'completion.json')['completed']
    if read(SOURCE/'requests.json')!=read(OURS/'requests.json'):raise ValueError('Existing ZsRE data differ')
    native=modules();prior=read(SOURCE/'design.json');own=read(OURS/'design.json')
    rows=read(SOURCE/'requests.json')[:1000];ledger=read(OURS/'ledger.json')[:1000]
    if len(rows)!=1000 or len(ledger)!=1000:raise ValueError('Missing official Q&A memories')
    if shutil.disk_usage(out.parent).free<25*2**30:raise RuntimeError('Require 25 GiB reserve for construction/snapshots')
    out.mkdir(parents=True)
    for name in ['factors','plain_alphaedit','wise_merge','wise_retrieve','rag','kv_full','our_delta']:(out/name).mkdir()
    for row in ledger:
        name=row['sid']+'.pt';source=OURS/'factors'/name
        if digest(source)!=row['sha256']:raise ValueError('Saved independent delta changed')
        os.link(source,out/'factors'/name)
    save(out/'requests.json',rows);save(out/'factor_ledger.json',ledger)
    save(out/'context_templates.json',read(SOURCE/'context_templates.json'))
    files=[str(p.relative_to(ROOT)) for p in (ROOT/'experiments/zsre_generated').glob('*.py')]
    files += ['compact_session_memory.py','comparison_core.py','question_key.py','answer_only_generation.py',
        'ttft_timing.py','memory_comparison_core.py','experiment_utils.py','grounding_core.py',
        'experiments/session_zsre/core.py','experiments/wise_paper/native.py','experiments/wise_paper/worker.py',
        'experiments/wise_paper/prepare.py','experiments/wise_qa500/core.py','config/alphaedit-hparams.json']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'AlphaEdit').glob('*.py')]
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'rome').glob('*.py')]
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'util').glob('*.py')]
    files += [str((VENDOR/p).relative_to(ROOT)) for p in read(VENDOR/'UPSTREAM.json')['files']]
    projection=OURS/'projection';projector=read(projection/'manifest.json')
    for f,h in projector['files'].items():
        if digest(projection/f)!=h:raise ValueError('Mistral preservation projector changed')
    d=dict(name=out.name,model=prior['model'],model_repository=prior['model_repository'],model_revision=prior['model_revision'],
        model_manifest=read(SOURCE/'model_manifest.json'),embedding_model=str(ENCODER),physical_gpu=1,precision='float32',seed=42,
        sizes=SIZES,methods=METHODS,repeats=10,answer_cap=128,selected_count=1000,
        plan=[dict(n=n,questions=n,case_ids=list(range(n))) for n in SIZES],
        code_sha256={name:digest(ROOT/name) for name in sorted(set(files))},
        source_requests_sha256=digest(SOURCE/'requests.json'),source_run=str(SOURCE),our_source_run=str(OURS),
        projection=str(projection),projector=projector,layers=[13,14,15],alpha_hparams=own['hyperparameters'],
        wise_hparams=prior['hyperparameters'],
        unit='One official ZsRE edit/Q&A record per memory; no conversation context exists in this dataset.',
        write='First 1000 ordered official ZsRE edit records, same original questions and counterfactual alt answers. '
              'Our/ordinary AlphaEdit writes complete alt plus EOS with the validated raw-question key/value template. '
              'WISE retains native generated prefix augmentation, left padding, SGD/loss/masks and target policy, with training-split locality prompts. '
              'RAG/KV store every original question and complete alt answer. No evaluation paraphrase is written.',
        plain_solver='Cumulative AlphaEdit with the continuous post-write key preservation accumulator, represented exactly as key factors. '
                     'Woodbury reduction of the original linear system; FP64 core solve followed by FP32 updates. '
                     'Dense-solve parity is tested on tractable matrices; no full-size bitwise parity claim.',
        reuse='All 1000 independent factors and model-specific projector reused by checksum. '
              'Original WISE run retained only the rolling 1000-edit weights; its JSON checkpoint scores cannot reconstruct earlier weights. '
              'Rebuild separate continuous native WISE-Merge and WISE-Retrieve trajectories through 1000 to obtain every missing prefix, preserving the original augmentation templates. '
              'Build ordinary cumulative AlphaEdit and the matched RAG index. No old evaluation outputs or timings are reused.',
        evaluation='All first-N ORIGINAL src questions at every N, ten fresh inference passes per method. '
                   'No paraphrases, teacher-forced targets, oracle memory lookup or evaluation-driven tuning. '
                   'Deterministic shuffled order is matched across methods within each repeat; method order rotates.',
        routing='SessionControl retains original normalized FFN delta-response energy argmax on the raw question, layers 13-15. '
                'WISE retains native main/side/shard routing. Plain AlphaEdit has no session control. '
                'RAG uses question-key MiniLM cosine top-one; full-bank KV retains all first-N Q&A records.',
        generation='Raw question prompts for all parametric methods, matching their existing ZsRE write keys. '
                   'RAG/KV use the same fixed plain-text memory-record instruction and query suffix; Mistral base has no chat template. '
                   'Greedy decoding, 128-token cap, EOS and reference-independent next-turn stopping for all methods.',
        metrics='Complete returned final-answer normalized word-token F1 and exact match, visible/generated TTFT, total latency, output length/cap rate, '
                'retained bank disk bytes, sampled process RSS, peak allocated/reserved GPU. Same metric helpers as LongMemEval. '
                'No answer extraction, containment scoring or teacher-forced accuracy substituted.',
        timing='Query arrival through routing/embedding/delta load+apply/prefill to first nonempty final-answer text. '
               'Checkpoint installation, model load and memory writing excluded. KV uses warm prefix cache; RAG builds selected-record cache per request. '
               'Two unrelated one-token warmups. Empty outputs remain in accuracy with missing visible TTFT. OS file cache not flushed.',
        memory='Separate disk, host RAM and GPU tiers. GPU includes common Mistral weights; RAG also includes its encoder. '
               'Reset peaks/include bank setup at each N in isolated method/pass workers. Construction projector/accumulator excluded. '
               'Full-bank KV capacity and actual payload checked; no truncation or retrieval fallback. Native Mistral sliding_window=4096 remains unchanged and is reported.',
        wise_merge='At N=500, finalize an evaluation-only snapshot using unchanged native WISE.__call__ with an unrelated setup prompt, '
                   'then restore the pre-finalization training state before continuing the same trajectory. '
                   'The final native merge at N=1000 happens in the ordinary native edit routine. '
                   'Checkpoint setup uses no evaluation target. Retrieve retains its native shards and threshold state.',
        repetition='Ten inference measurement passes on one frozen trajectory per method; not ten independently trained banks. '
                   'Report mean and sample SD across ten per-pass metrics. Identical greedy answers can have zero quality SD.',
        limits='Supervised exact-question counterfactual Q&A memory test, not blind LongMemEval or an official ZsRE paper score. '
               'The prior LongMemEval study used Llama-3.1, while this study uses the already validated paper Mistral model. '
               'Between-dataset differences cannot be attributed solely to the dataset. No measured write-time comparison is claimed.',
        prepared_utc=now())
    save(out/'design.json',d)
    save(out/'input_manifest.json',{f:digest(out/f) for f in ['design.json','requests.json','factor_ledger.json','context_templates.json']})
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1))
    print('PREPARED',out,'six methods, ten passes,',sum(SIZES)*10,'planned questions per method',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);prepare(p.parse_args().out.resolve())
