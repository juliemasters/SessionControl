"""Prepare an explicitly versioned recovery without discarding trained memories."""
import argparse
import os
from pathlib import Path
import shutil
from .core import ROOT,METHODS,read,save,digest,modules,now


def prepare_recovery(source,out):
    source=source.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('Recovery requires a fresh output directory')
    prior=read(source/'design.json');failure=read(source/'completion.json')
    if failure.get('completed') or failure.get('stage')!='failed':raise ValueError('Expected the recorded failed run')
    for name,want in read(source/'input_manifest.json').items():
        if digest(source/name)!=want:raise ValueError('Prior frozen input changed: '+name)
    changed={name for name,want in prior['code_sha256'].items() if digest(ROOT/name)!=want}
    allowed={'experiments/zsre_generated/core.py','experiments/zsre_generated/build.py'}
    if changed!=allowed:raise ValueError('Unexpected source changes during checkpoint repair: '+repr(changed))
    native=modules()
    import torch
    # Inspect only this trusted, locally generated native restart file. Native
    # checkpoints contain config/stat objects and require weights_only=False.
    resume={}
    for method in ['wise_merge','wise_retrieve']:
        state=torch.load(source/method/'restart.pt',map_location='cpu',weights_only=False)
        resume[method]=int(state['additional_info']['editing_total_cnt']);del state
        if not 0<resume[method]<=len(read(source/method/'ledger.json')):raise ValueError('Invalid native resume count')
    trained=read(source/'plain_alphaedit/construction.json')
    if not trained.get('completed') or trained['count']!=1000:raise ValueError('Expected completed cumulative AlphaEdit bank')
    if len(read(source/'factor_ledger.json'))!=1000:raise ValueError('Independent factor bank incomplete')
    # Audit every reusable deployed prefix before linking it into the new run.
    verified={}
    for method,stem in [('plain_alphaedit','weights'),('wise_merge','wise'),('wise_retrieve','wise')]:
        for meta in (source/method).glob(stem+'-*.json'):
            info=read(meta);file=meta.with_suffix('.pt');want=info['sha256']
            if digest(file)!=want:raise ValueError('Saved prefix changed: '+str(file))
            verified[str(file.relative_to(source))]=want
    for entry in read(source/'factor_ledger.json'):
        file=source/'factors'/(entry['sid']+'.pt')
        if digest(file)!=entry['sha256']:raise ValueError('Saved factor changed')
    for name,want in read(source/'rag/index_manifest.json').items():
        if digest(source/'rag'/name)!=want:raise ValueError('Saved RAG index changed')
    out.mkdir(parents=True)
    for method in METHODS:(out/method).mkdir()
    (out/'factors').mkdir()
    for name in ['requests.json','factor_ledger.json','context_templates.json']:
        shutil.copy2(source/name,out/name)
    def linked(file):
        dest=out/file.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);os.link(file,dest)
    for file in (source/'factors').glob('*.pt'):linked(file)
    for method,stem in [('plain_alphaedit','weights'),('wise_merge','wise'),('wise_retrieve','wise')]:
        for file in (source/method).glob(stem+'-*.pt'):linked(file)
        for file in (source/method).glob(stem+'-*.json'):shutil.copy2(file,out/method/file.name)
        shutil.copy2(source/method/'construction.json',out/method/'construction.json')
        ledger=read(source/method/'ledger.json')
        if method in resume:ledger=ledger[:resume[method]]
        save(out/method/'ledger.json',ledger)
    for method in resume:
        for name in ['restart.pt','rng.pt']:linked(source/method/name)
        shutil.copy2(source/method/'rng.json',out/method/'rng.json')
    for file in (source/'rag').glob('index-*/*'):
        if file.is_file():linked(file)
    shutil.copy2(source/'rag/index_manifest.json',out/'rag/index_manifest.json')
    d=dict(prior,name=out.name)
    d['code_sha256']={name:digest(ROOT/name) for name in prior['code_sha256']}
    recovery_file=str(Path(__file__).resolve().relative_to(ROOT))
    d['code_sha256'][recovery_file]=digest(__file__)
    d['recovery']=dict(source=str(source),source_commit=read(source/'github_code_receipt.json')['commit'],
        failure=failure,source_design_sha256=digest(source/'design.json'),
        changed_runtime_files=sorted(changed),resume_counts=resume,
        completed_plain_memories=1000,independent_factors_reused=1000,
        deployed_checkpoint_sha256=verified,previous_inference_rows_copied=0,
        repair='Finalize the evaluation-only merge under no_grad with inference mode disabled; '
               'restore ordinary tensors before the next native optimizer step. '
               'No changes to WISE source, weights, masks, thresholds, optimizer, data, metrics or generation policy.',
        prepared_utc=now())
    d['reuse']+=' Recovery reuses the completed 1000-item cumulative bank and resumes native WISE from saved '
    d['reuse']+='weights/RNG state after a checkpoint tensor-lifecycle error. The previous run is preserved; '
    d['reuse']+='all ten evaluation passes are freshly measured in this version, with no previous answer/timing rows copied.'
    save(out/'design.json',d)
    save(out/'input_manifest.json',{name:digest(out/name) for name in ['design.json','requests.json','factor_ledger.json','context_templates.json']})
    save(out/'recovery_audit.json',d['recovery'])
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1,recovered_from=str(source)))
    print('PREPARED RECOVERY',out,'native resumes',resume,'completed plain memories reused: 1000',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare_recovery(a.source,a.out)
