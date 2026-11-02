"""Ten new LongMemEval measurement passes over frozen native WISE memories.

The successful ZsRE reproduction establishes a separate native reference; it did
not identify a change to the previously audited LongMemEval checkpoints. Keep
those weights, thresholds, model, questions and metrics unchanged. No training,
test-set tuning, forced-memory routing or reuse of old inference cells occurs.
"""
import argparse
from pathlib import Path
import html
import os
import random
import shutil
import subprocess
import sys
import time
from . import core

ROOT=core.ROOT
SOURCE=ROOT/'runs/wise-qa500-v1'


def source_files():
    old=core.read(SOURCE/'wise/run-00/provenance.json')['code_sha256']
    for name,want in old.items():
        if core.digest(ROOT/name)!=want:raise ValueError('Original inference source changed: '+name)
    return dict(old,**{str(Path(__file__).resolve().relative_to(ROOT)):core.digest(__file__)})


def prepare(out):
    if out.exists():raise FileExistsError('Fresh inference measurements require a new directory')
    prior=core.check(SOURCE);done=core.read(SOURCE/'completion.json')
    if not done.get('completed') or done['passes']!=10:raise ValueError('Incomplete frozen WISE source run')
    hashes=source_files()
    if core.digest(prior['dataset'])!=prior['dataset_sha256']:raise ValueError('Original LongMemEval dataset changed')
    metadata=core.read(core.HERE/'reference/metadata.json')
    model=Path(prior['model'])
    if core.digest(model/'config.json')!=metadata['model_config_sha256']:raise ValueError('Base model config changed')
    if core.digest(model/'tokenizer.json')!=metadata['tokenizer_sha256']:raise ValueError('Original tokenizer changed')
    construction=core.read(SOURCE/'construction.json')
    if not construction.get('completed') or construction['conversations']!=500:raise ValueError('Incomplete WISE bank')
    for n in prior['sizes']:
        want=construction['checkpoint_sha256'][str(n)]
        if core.digest(SOURCE/'checkpoints'/f'wise-{n}.pt')!=want:raise ValueError('Saved WISE memory changed')
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'reference').mkdir()
    for name in ['sessions.json','queries.json','locality.json','ledger.json','construction.json','first_write_contract.json',
                 'reference/comparison.json','reference/comparison.csv']:
        shutil.copy2(SOURCE/name,out/name)
    for n in prior['sizes']:
        # Hard links preserve each frozen checkpoint if an older result directory
        # is later archived, without copying gigabytes or modifying its bytes.
        os.link(SOURCE/'checkpoints'/f'wise-{n}.pt',out/'checkpoints'/f'wise-{n}.pt')
        shutil.copy2(SOURCE/'checkpoints'/f'wise-{n}.json',out/'checkpoints'/f'wise-{n}.json')
    d=dict(prior,name=out.name)
    d['evaluation']=('Ten NEW actual inference measurement passes on the unchanged saved WISE trajectory, '
        'with the same eligible original questions, matched per-pass ordering, whole final-answer F1/exact match, '
        'visible/generated TTFT, total latency and separate disk/host/GPU memory. No prior WISE outputs or timings copied.')
    d['remeasurement']=dict(source=str(SOURCE),new_passes=10,previous_passes=10,
        reused='Seven frozen WISE checkpoints, original complete-context/Q/A write provenance and matched inputs. '
               'Four baseline aggregates retained from the previous completed ten-pass study; their measurements are not rerun.',
        source_files={name:core.digest(SOURCE/name) for name in ['design.json','completion.json','construction.json','ledger.json']},
        code_sha256=hashes,native_inference_class='_sc_wise.models.wise.WISE.WISE',
        repetition='Ten inference passes over one frozen trained model trajectory; not ten independently trained models. '
                   'Aggregate the NEW ten only, without pooling the previous ten. Greedy answer-quality SD may be zero.',
        relation_to_zsre='The separately validated ZsRE reproduction used Mistral and teacher-forced metrics. '
                         'Return here to Llama-3.1, LongMemEval and final generated-answer metrics. '
                         'No correction to the old trained weights was established; native WISE behavior is retained.',
        prepared_utc=core.now())
    core.save(out/'design.json',d)
    core.save(out/'input_manifest.json',{name:core.digest(out/name) for name in ['design.json','sessions.json','queries.json','locality.json','reference/comparison.json','reference/comparison.csv']})
    core.save(out/'reuse_audit.json',dict(checkpoints_reused=prior['sizes'],saved_bindings=construction['bindings'],
        source_code_files_verified=len(hashes)-1,prior_inference_results_copied=0,baseline_aggregates_reused=True,
        native_weights_thresholds_and_hyperparameters_changed=False))
    core.save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1,new_passes=10))
    print('PREPARED',out,[(p['n'],p['questions']) for p in d['plan']],flush=True)


def check(out):
    d=core.check(out)
    for name,want in d['remeasurement']['code_sha256'].items():
        if core.digest(ROOT/name)!=want:raise ValueError('Frozen remeasurement source changed: '+name)
    for name,want in d['remeasurement']['source_files'].items():
        if core.digest(Path(d['remeasurement']['source'])/name)!=want:raise ValueError('Source write provenance changed')
    return d


def native_load(out,folder):
    """Direct official WISE instance, with the same checkpoint and input codec.

    The original integration subclass only optimized training. Its training
    methods are not needed during this inference-only run and are not installed.
    """
    d=check(out)
    from experiment_utils import require_gpu1
    uuid=require_gpu1();os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    import numpy as np
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if torch.cuda.device_count()!=1:raise RuntimeError('Only physical GPU 1 may be visible')
    torch.set_num_threads(8);torch.manual_seed(0);torch.cuda.manual_seed_all(0);random.seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    folder.mkdir(parents=True,exist_ok=True)
    hashes=d['remeasurement']['code_sha256']
    if (folder/'provenance.json').exists():
        if core.read(folder/'provenance.json')['code_sha256']!=hashes:raise ValueError('Inference provenance mismatch')
    else:
        for name in hashes:
            dest=folder/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        core.save(folder/'provenance.json',dict(code_sha256=hashes,physical_gpu=1,gpu_uuid=uuid,torch_version=torch.__version__,
            native_class=d['remeasurement']['native_inference_class'],started_utc=core.now()))
    official,hparams=core.vendor_modules()
    hp=hparams.WISEHyperParams.from_hparams(str(core.HERE/'config/official-llama3.1-8b.yaml'))
    hp.model_name=d['model'];hp.device=0;hp.use_chat_template=False;hp.sequential_edit=True
    tok=AutoTokenizer.from_pretrained(d['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='left'
    core.status(folder,'loading_model')
    model=AutoModelForCausalLM.from_pretrained(d['model'],torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
    editor=official.WISE(hp,model,torch.device('cuda:0'))
    if type(editor) is not official.WISE:raise ValueError('Expected direct native WISE class')
    return d,torch,tok,editor


def evaluate(out,repeat,sizes=None):
    # Reuse the original generation/timing/memory measurement loop unchanged.
    # Supply a direct official-class loader, without altering WISE methods.
    from . import worker
    worker.load=native_load
    worker.evaluate(out,repeat,sizes)


def report(out):
    check(out)
    from . import report as original
    original.build(out)  # Recompute and audit all 6,630 new WISE answers/timings.
    text=(out/'report.md').read_text()
    text=text.replace('Completed 500 sequential conversation writes and ten actual inference passes at every checkpoint on physical GPU 1.',
        'Reused the unchanged saved 500-conversation WISE trajectory and completed ten NEW inference passes at every checkpoint on physical GPU 1.')
    text=text.replace('only WISE was newly trained and evaluated.',
        'only WISE was newly evaluated; its saved weights and thresholds were reused unchanged.')
    text+='\n## Fresh remeasurement provenance\n\n'+core.read(out/'design.json')['remeasurement']['repetition']+'\n\n'
    text+='All WISE inference uses the direct official WISE class, with unchanged native activation routing, saved side weights and thresholds. '
    text+='The original training-only optimization subclass is not installed. The 500-session data, all original exact questions, decoder, generation cap and metric helpers match the previous experiment.\n\n'
    text+='The ZsRE study remains separate. No teacher-forced ZsRE token accuracy is substituted for generated-answer F1 or exact match here. '
    text+='Baseline accuracy and timing curves remain the prior completed measurements and are explicitly labeled as reused.\n'
    (out/'report.md').write_text(text)
    svgs=''.join('<svg'+(out/f'{name}.svg').read_text().split('<svg',1)[1] for name in ['ttft','accuracy_final_answer','memory'])
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>WISE LongMemEval: ten fresh passes</title>'
        '<style>body{font:16px/1.6 system-ui;max-width:1450px;margin:40px auto;padding:20px}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style>'
        '<h1>WISE · ten fresh LongMemEval measurement passes</h1><p>Whole final-answer F1 and exact match, TTFT, latency and separate memory tiers. '
        'Frozen WISE memories reused; previous inference results excluded. Four baseline curves reuse their earlier ten-pass measurements.</p>'
        +svgs+'<pre>'+html.escape(text)+'</pre>')
    manifest=core.read(out/'report_manifest.json')
    core.save(out/'report_manifest.json',{name:core.digest(out/name) for name in manifest})
    done=core.read(out/'completion.json');done.update(new_measurement_passes=10,previous_passes_pooled=False,
        checkpoints_reused=True,source_run=str(SOURCE));core.save(out/'completion.json',done)
    print('FINAL_REMEASUREMENT_REPORT_READY',out/'report.html',flush=True)


def child(out,repeat,sizes=None):
    while True:
        busy=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
        if not busy.stdout.strip():break
        core.status(out,'waiting_for_gpu1',repeat=repeat);time.sleep(30)
    args=[sys.executable,'-u','-m','experiments.wise_qa500.remeasure','eval','--out',str(out),'--repeat',str(repeat)]
    if sizes:args+=['--sizes',*[str(n) for n in sizes]]
    p=subprocess.Popen(args,cwd=ROOT)
    core.save(out/'driver_status.json',dict(stage='inference',repeat=repeat,worker_pid=p.pid,started_utc=core.now()))
    core.status(out,'running',repeat=repeat)
    if p.wait():raise RuntimeError('WISE remeasurement worker failed; see driver.log')


def drive(out):
    check(out);core.save(out/'completion.json',dict(completed=False,stage='running',physical_gpu=1,new_passes=10))
    try:
        child(out,0,[8])
        pilot=core.read(out/'wise/run-00/bank-8.json')
        if not pilot.get('completed') or len(pilot['trials'])!=8:raise ValueError('Incomplete first inference cell')
        core.save(out/'pilot_validation.json',dict(functional=True,questions=8,
            note='First cell of the NEW ten inference passes. No accuracy threshold, hyperparameter change or retraining.'))
        for repeat in range(10):child(out,repeat)
        core.status(out,'audit_and_report');report(out)
        core.status(out,'completed');core.save(out/'driver_status.json',dict(stage='completed',finished_utc=core.now()))
    except BaseException as error:
        core.save(out/'completion.json',dict(completed=False,stage='failed',physical_gpu=1,error=repr(error)))
        core.status(out,'failed',error=repr(error));raise


def launch(out):
    check(out)
    if core.read(out/'completion.json').get('completed'):raise RuntimeError('Remeasurement already complete')
    meta=out/'detached_launch.json'
    if meta.exists():
        prior=core.read(meta);p=Path('/proc')/str(prior['pid'])/'cmdline'
        if p.exists() and b'wise_qa500.remeasure' in p.read_bytes():raise RuntimeError('Driver already running')
    args=[sys.executable,'-u','-m','experiments.wise_qa500.remeasure','drive','--out',str(out)]
    with (out/'driver.log').open('a') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 close_fds=True,start_new_session=True)
    core.save(meta,dict(pid=process.pid,command=args,physical_gpu=1,detached=True,started_utc=core.now()))
    print('DETACHED_WISE_REMEASUREMENT_DRIVER',process.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','drive','eval','report'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repeat',type=int,default=0);p.add_argument('--sizes',type=int,nargs='+')
    a=p.parse_args();out=a.out.resolve()
    if a.action=='eval':evaluate(out,a.repeat,a.sizes)
    else:globals()[a.action](out)
