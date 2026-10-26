"""Frozen protocol helpers for the official-only four-method experiment."""
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time

from comparison_core import digest,read,save

SIZES=(8,16,32,64)
ROOT=Path(__file__).resolve().parent
CODE=Path('/home/julie/Desktop/draft_1')
SOURCE=Path('/home/julie/sessioncontrol-backups/cleanup-20261015T022610Z/runs/scaling-real512-compact-v1')
METHODS=('plain_alphaedit','our_delta','rag_qa','kv_full_qa')


def question_plan(items):
    if len(items)!=64:raise ValueError('Require 64 official items')
    for field in ['qid','sid','question','source_session_id']:
        if len({r[field] for r in items})!=64:raise ValueError('Duplicate '+field)
    plan=[];previous=0
    for n in SIZES:
        indices=list(range(previous,n));plan.append(dict(n=n,indices=indices,questions=len(indices),coverage=len(indices)/n));previous=n
    return plan


def prepare(out):
    if out.exists():raise FileExistsError('New experiment directory required')
    source_design=read(SOURCE/'design.json');done=read(SOURCE/'completion.json')
    if not done.get('completed'):raise ValueError('Incomplete source bank')
    items=read(SOURCE/'items.json')[:64];plan=question_plan(items)
    dataset=Path(source_design['data_root'])/'data/longmemeval_s.json';official={r['question_id']:r for r in read(dataset)}
    for r in items:
        p=official[r['qid']];sessions=dict(zip(p['haystack_session_ids'],p['haystack_sessions']))
        if (p['question_type']!='single-session-user' or str(p['question_id']).endswith('_abs')
            or (r['question'],r['answer'])!=(p['question'],p['answer'])
            or r['source_session_id'] not in p['answer_session_ids'] or r['context']!=sessions[r['source_session_id']]):
            raise ValueError('Item is not unchanged answerable official single-session-user')
    design=dict(name='official64-four-method-v1',source=str(SOURCE),code_root=str(CODE),model=source_design['model'],
        embedding_model=source_design['embedding_model'],projection=source_design['projection'],projection_sha256=source_design['projection_sha256'],
        layers=[13,14,15],dtype='float32',physical_gpu=1,seed=0,write_steps=50,l2=.1,generation_cap=64,repeats=1,
        sizes=list(SIZES),cohorts=plan,unique_questions=64,
        protocol='Official answerable LongMemEval single-session-user only. User chose no repeated questions and N<=64; coverage revised from 80% to all eight questions at N=8 and the newly added half at N=16/32/64. Every item is asked once per method; F1, TTFT and memory come from the same requests.',
        sampling='8,8,16,32 different questions at N=8,16,32,64. No question is reused across checkpoints or in separate timing passes. Identical question sets and orders for all methods.',
        scope='Exact dataset questions and answers are visible with the full source conversation at write time. No paraphrases. Different cohorts change difficulty and emphasize newly added memories, so the N curve is not a controlled retention-only or held-out generalization test.',
        rag='Matched Q/A-key dense RAG: existing MiniLM question embeddings, top-1 exact search, full selected conversation and stored Q/A supplied to the base model. The supplied question is an intentional write key for every method.',
        kv='All N complete memory records and stored Q/A in one reusable prefix cache. Warm query TTFT excludes already prepared prefix; one cold-prefix first request is separately timed. No truncation or CPU offload beyond context/GPU capacity.',
        ttft='Local query arrival to first nonempty decoded answer text. Includes activation scoring/selected-delta read+apply for our method, query embedding/search/prompt construction/prefill for RAG, question tokenization/suffix prefill for warm KV, and tokenization/prefill for shared AlphaEdit. Excludes model startup and memory writing. No network/serving queue. One request per unique item after unrelated model warmups; p50/p95 describe the different questions, not repeated trials. Small sample sizes and missing visible-token timestamps are reported; OS caches are not flushed.',
        memory='Separate actual extra disk bank/patch bytes, process host RSS, retained cache/vector/factor payload, and CUDA peak allocated/reserved. Common model weights are reported separately; tiers are not added or compared as interchangeable capacity. GPU peaks include bank setup and inference; host sampling interval 20ms. Training/projector/accumulator memory is excluded by running shared construction in a separate process.',
        baseline='Original dense AlphaEdit sequentially updates one model and carries its preservation accumulator. Four shared FFN patches are saved from one continuous 64-edit trajectory, then evaluated in a separate inference process. No session router, retrieved context or independent session delta is used by this baseline.',
        baseline_limit='Independent compact writing uses fresh preservation state and a single-key solver; shared AlphaEdit uses the original dense solver and cumulative preservation state. This compares complete systems under matched inputs/hyperparameters, not an isolation-only ablation or independently tuned published AlphaEdit optimum.',
        write_scope='No elapsed write-time comparison is published; the requested write-cost comparison remains theoretical.',
        source_sha256={f:digest(SOURCE/f) for f in ['design.json','items.json','completion.json','written_bank_sha256.json','bank.json','qa_embeddings.pt']},
        dataset=str(dataset),dataset_sha256=digest(dataset))
    out.mkdir(parents=True);save(out/'design.json',design);save(out/'items.json',items)
    save(out/'manifest.json',{f:digest(out/f) for f in ['design.json','items.json']})
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1))
    (out/'shared').mkdir()
    for method in METHODS:(out/method).mkdir()
    print('PREPARED official64: 8/8/16/32 new questions, four methods',flush=True)


def check_inputs(out):
    d=read(out/'design.json')
    for f,h in read(out/'manifest.json').items():
        if digest(out/f)!=h:raise ValueError('Prepared protocol changed')
    source=Path(d['source'])
    for f,h in d['source_sha256'].items():
        if digest(source/f)!=h:raise ValueError('Source changed')
    if digest(d['dataset'])!=d['dataset_sha256']:raise ValueError('Official dataset changed')
    return d


def order_indices(cohort,items,repeat):
    import hashlib
    return sorted(cohort['indices'],key=lambda i:hashlib.sha256(f"{cohort['n']}/{repeat}/{items[i]['qid']}".encode()).hexdigest())


def rss_bytes():
    return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')


class HostPeak:
    def __enter__(self):
        self.before=rss_bytes();self.peak=self.before;self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.02):self.peak=max(self.peak,rss_bytes())
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.after=rss_bytes();self.peak=max(self.peak,self.after);self.stop.set();self.thread.join()
    def result(self):return dict(host_rss_before_bytes=self.before,host_peak_rss_bytes=self.peak,host_rss_after_bytes=self.after,host_sampling_ms=20)


def timing_stats(trials):
    samples=list(trials);v=sorted(r['ttft_seconds'] for r in samples if r['ttft_seconds'] is not None)
    for r in trials:
        t=r['ttft_seconds']
        if t is not None and not 0<=t<=r['query_seconds']:raise ValueError('TTFT outside request interval')
    return dict(timing_samples=len(samples),visible_ttft_samples=len(v),missing_visible_samples=len(samples)-len(v),
        median_seconds=statistics.median(v) if v else None,p95_seconds=v[math.ceil(.95*len(v))-1] if v else None)


def verify_trial_grid(trials,cohort,items,repeats):
    expected={(repeat,items[i]['qid']) for repeat in range(repeats) for i in cohort['indices']}
    actual=[(r['repeat'],r['qid']) for r in trials]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Incomplete/repeated trial grid')


def snapshot_code(out,files):
    target=out/'code_snapshot';target.mkdir(exist_ok=True)
    import shutil
    hashes={};snapshots={}
    for i,p in enumerate(files):
        p=Path(p);dest=target/(str(i)+'_'+p.name);shutil.copy2(p,dest)
        hashes[str(p)]=digest(p);snapshots[str(p)]=str(dest.relative_to(out))
    return dict(code_sha256=hashes,code_snapshots=snapshots)
