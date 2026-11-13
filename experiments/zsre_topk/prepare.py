"""Freeze two matched top-k experiments while reusing all written memories."""
import argparse
from pathlib import Path
import shutil
from .core import ROOT,METHODS,LABELS,read,save,digest,now


def prepare_one(out,dataset):
    source=ROOT/'runs'/('zsre-generated-sixmethod-1000-v3' if dataset=='zsre' else 'qa500-checkpoints-tenpass-v2')
    old=read(source/'design.json')
    if not read(source/'completion.json')['completed']:raise ValueError('Original run incomplete')
    for name,want in read(source/'report_manifest.json').items():
        if digest(source/name)!=want:raise ValueError('Original artifact changed: '+name)
    if dataset=='zsre':
        requests=[dict(case_id=r['case_id'],prompt=r['prompt'],target_new=r['target_new'],answer_sids=[f"case_{r['case_id']:04d}"],abstention=False)
                  for r in read(source/'requests.json')]
        ledger=read(source/'factor_ledger.json')
        factors=[dict(sid=r['sid'],path=str(source/'factors'/(r['sid']+'.pt')),sha256=r['sha256']) for r in ledger]
        bindings=[dict(sid=r['answer_sids'][0],case_id=r['case_id'],prompt=r['prompt']) for r in requests]
        plan=[dict(n=n,case_ids=list(range(n))) for n in old['sizes']]
        protected=dict(old['code_sha256']);manifest_name='input_manifest.json';seed=old['seed']
        model_manifest=old['model_manifest'];revision=old['model_revision']
    else:
        requests=[dict(case_id=r['qid'],prompt=r['question'],target_new=r['answer'],answer_sids=r['support_sids'],
                       abstention=r['abstention'],question_type=r['question_type']) for r in read(source/'queries.json')]
        sessions=read(source/'sessions.json');ledger=read(source/'independent/ledger.json')
        factors=[dict(sid=r['sid'],path=str(source/'independent/factors'/r['file']),sha256=r['sha256']) for r in ledger]
        if [s['sid'] for s in sessions]!=[r['sid'] for r in factors]:raise ValueError('Conversation/factor order mismatch')
        bindings=[dict(sid=r['sid'],case_id=b['qid'],prompt=b['question']) for r in sessions for b in r['bindings']]
        plan=[dict(n=p['n'],case_ids=p['qids']) for p in old['plan']]
        original=read(source/'our_delta/run-00/provenance.json')['code_sha256']
        protected={str(Path(name).resolve().relative_to(ROOT)):want for name,want in original.items()}
        manifest_name='manifest.json';seed=0;revision='Same local Llama snapshot; exact file SHA256 recorded'
        model_files=[p for p in sorted(Path(old['model']).iterdir()) if p.is_file() and p.suffix in {'.safetensors','.json','.model'}]
        model_manifest=dict(sha256={p.name:digest(p) for p in model_files})
    for name,want in protected.items():
        if digest(ROOT/name)!=want:raise ValueError('Original source changed: '+name)
    for row in factors:
        if digest(row['path'])!=row['sha256']:raise ValueError('Saved delta changed')
    if len({r['case_id'] for r in requests})!=len(requests):raise ValueError('Duplicate question IDs')
    by_id={r['case_id']:r for r in requests}
    for b in bindings:
        if b['prompt']!=by_id[b['case_id']]['prompt']:raise ValueError('Write key mismatch')
    local_files=list((ROOT/'experiments/zsre_topk').glob('*.py'))
    code={**protected,**{str(p.relative_to(ROOT)):digest(p) for p in local_files}}
    d=dict(name=out.name,dataset=dataset,physical_gpu=1,precision='float32',model=old['model'],model_revision=revision,
           model_manifest=model_manifest,seed=seed,layers=old['layers'],answer_cap=old['answer_cap'],
           source_run=str(source),source_manifest_name=manifest_name,source_manifest_sha256=digest(source/manifest_name),
           source_report_manifest_sha256=digest(source/'report_manifest.json'),
           methods=METHODS,labels=LABELS,sizes=[p['n'] for p in plan],plan=plan,repeats=10,ks=[1,8,16,32],
           bank_sids=[r['sid'] for r in factors],factor_files=factors,code_sha256=code,
           expected_requests=10*len(METHODS)*sum(len(p['case_ids']) for p in plan),
           index='One normalized CPU FP32 concatenation of base last-question-token FFN inputs at layers13–15 per write question. '
                 'LongMemEval has508 binding rows for500 conversations; max cosine per session, then top-k distinct deltas.',
           scope='Exact original write-time questions; stored activation keys add an external question index. No paraphrase or blind generalization claim.',
           routing='Cosine shortlist then unchanged original delta-response energy. Stable descending score, ascending SID ties. k1 has one fine candidate.',
           generation='Original dataset-specific prompts and decoding unchanged: raw question for ZsRE, Q:/A: for LongMemEval, greedy cap128/EOS/next-turn stopping.',
           timing='All arms freshly measured. Equal CPU-preloaded factor bank; chosen factor reread from disk for every request as before. '
                  'TTFT includes index search, candidate scoring, transfers, delta application, prefill and first visible answer; model/index construction excluded.',
           memory='Extra normalized activation index counted in disk/RAM; only prefix index loaded. Full factor preload equal across arms. '
                  'No reduced precision, quantization or decoder change. GPU includes common model weights. Index build cost separately recorded.',
           repetition='Ten fresh inference passes on one fixed written bank, not ten training seeds. Rotate method order; matched deterministic query order.',
           preflight='Fresh query-only routing over final eligible cohort for all four k values. Historical full-scan selected IDs used only as descriptive reference, never as new timing.',
           prepared_utc=now())
    out.mkdir(parents=True)
    save(out/'requests.json',requests);save(out/'bindings.json',bindings);save(out/'design.json',d)
    save(out/'input_manifest.json',{name:digest(out/name) for name in ['design.json','requests.json','bindings.json']})
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1))
    print('PREPARED',dataset,'memories',len(factors),'binding keys',len(bindings),'eligible',len(plan[-1]['case_ids']),
          'fresh generation requests',d['expected_requests'],flush=True)


def prepare(out):
    if out.exists():raise FileExistsError('Use a fresh experiment directory')
    if shutil.disk_usage(ROOT).free<5*2**30:raise RuntimeError('Insufficient disk reserve')
    prepare_one(out/'zsre','zsre');prepare_one(out/'longmemeval','longmemeval')
    save(out/'datasets.json',{name:digest(out/name/'design.json') for name in ['zsre','longmemeval']})
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1,total_planned_requests=133350))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);prepare(p.parse_args().out.resolve())
