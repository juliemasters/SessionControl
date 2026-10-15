"""Replay completed compact512 memories on GPU1; measure TTFT and answer F1."""
import argparse
import gc
import os
from pathlib import Path
import random
import subprocess
import time

from experiment_utils import read,save,sha256
from experiment_utils import document,evaluated,metrics
from fixed_query_scaling_core import SIZES,check_complete,latency_summary
from ttft_timing import FirstAnswerClock,observe_generate,decode_cached_observed,ttft_summary


def prepare(source,out):
    if out.exists():raise ValueError('Require a new output directory')
    done=read(source/'completion.json');design=read(source/'design.json')
    if not done.get('completed') or done.get('real_sessions')!=512 or not done.get('compact_validation_passed'):
        raise ValueError('Require the completed validated512 source')
    names=['completion.json','design.json','items.json','evaluation.json','bank.json','qa_embeddings.pt']
    out.mkdir(parents=True)
    save(out,'design.json',dict(source=str(source),source_sha256={n:sha256(source/n) for n in names},
        factor_sha256=done['factor_sha256'],physical_gpu=1,sizes=list(SIZES),repeats=3,batch_size=1,
        model=design['model'],embedding_model=design['embedding_model'],dtype='float32',layers=[13,14,15],
        panel_ids=design['panel_ids'],seed=0,generation_cap=64,
        ttft='Local query arrival to first nonempty decoded final-answer text. Includes activation/probes+selection+factor IO/apply for deltas; embedding/search+prompt assembly/tokenization+prefill for RAG; fresh question tokenization+suffix prefill for warm KV. Excludes model startup and memory writing/index creation. No network or serving queue.',
        repeats_note='Eight fixed original questions. First pass reports accuracy; later16measurements report warm TTFT p50/p95. OS caches not flushed; no file-cache coldness claim.',
        cold_kv='One measured first request per feasible N: full-bank prompt construction/tokenization and prefix prefill through first visible answer. Model already resident; not a model cold-start or a p95 statistic.',
        scope=design['scope'],annotation=design['annotation'],f1='Custom casefolded word-token overlap with articles retained, matching the source experiment; not standard SQuAD normalization.',
        output_replay='Require source selection and emitted token IDs to match the prior corresponding question/repeat before publishing.',
        index_location='Existing MiniLM question vectors on CPU; encoder on GPU1. Full512 metadata/index present at smaller N, as in the source harness.'))
    save(out,'completion.json',dict(completed=False,stage='prepared'))


def run(out):
    from experiment_utils import require_gpu1
    gpu=require_gpu1()
    busy=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
    if busy.stdout.strip():raise RuntimeError('GPU1 busy; no fallback')
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    import torch
    from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
    from compact_session_memory import CompactSessionMemory,capture_keys,factor_scores
    from answer_only_generation import generate_answer_batch
    from memory_comparison_core import bank_prefix,query_suffix,prefill,embed,rank_vectors,cache_payload_bytes,kv_bytes_per_token
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Require only GPU1')
    torch.set_num_threads(8);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    design=read(out/'design.json');source=Path(design['source']);items=read(source/'items.json');panel=items[:8]
    if [r['qid'] for r in panel]!=design['panel_ids']:raise ValueError('Question panel changed')
    baseline=read(source/'evaluation.json')['results'];bank=read(source/'bank.json')
    source_hashes=design['source_sha256'];factor_hashes=design['factor_sha256'];design_hash=sha256(out/'design.json')
    root=Path(__file__).resolve().parent
    files=['experiment_real512_ttft.py','ttft_timing.py','compact_session_memory.py','answer_only_generation.py',
           'memory_comparison_core.py','experiment_utils.py','question_key.py','fixed_query_scaling_core.py']
    code_hashes={n:sha256(root/n) for n in files}
    results={};measurements={};store=None;model=None;encoder=None;cache=None;begin=time.monotonic()
    def sync():torch.cuda.synchronize()
    def status(stage,**extra):save(out,'status.json',dict(stage=stage,elapsed_seconds=time.monotonic()-begin,**extra))
    def verify_source():
        if any(sha256(source/n)!=h for n,h in source_hashes.items()):raise ValueError('Source artifact changed')
        if {r['sid']:sha256(source/'factors'/(r['sid']+'.pt')) for r in items}!=factor_hashes:
            raise ValueError('Source factors changed')
    def start(stage):
        gc.collect();torch.cuda.empty_cache();sync();torch.cuda.reset_peak_memory_stats();status(stage);return time.monotonic()
    def end(stage,started,**extra):
        sync();data=dict(elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),**extra)
        measurements[stage]=data;save(out,'measurements.json',measurements);return data
    def replay(trial,n,method):
        old=next(t for t in baseline[str(n)][method]['trials'] if t['qid']==trial['qid'] and t['repeat']==trial['repeat'])
        trial['tokens_match_source']=trial['final']['token_ids']==old['final']['token_ids']
        trial['source_match_source']=trial['selected_sid']==old['selected_sid'] and trial['retrieved_sids']==old['retrieved_sids']
        if not trial['tokens_match_source'] or not trial['source_match_source']:raise ValueError(f'Replay changed {n}/{method}/{trial["qid"]}')
        if trial['ttft_seconds'] is None:raise ValueError('No visible first answer token')
    def finish(n,method,trials,resources,**extra):
        summary=metrics([t for t in trials if t['repeat']==0])
        results.setdefault(str(n),{})[method]=dict(available=True,trials=trials,summary=summary,
            ttft=ttft_summary(trials),latency=latency_summary(trials),resources=resources,**extra)
        save(out,'evaluation_progress.json',dict(results=results))
        print(method,n,'TTFTmedian',results[str(n)][method]['ttft']['median_seconds'],'F1',summary['mean_token_f1'],flush=True)
    save(out,'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    try:
        status('integrity_before');verify_source()
        save(out,'provenance.json',dict(physical_gpu=1,gpu_uuid=gpu,torch_version=torch.__version__,cuda_version=torch.version.cuda,
            code_sha256=code_hashes,source_sha256=source_hashes))
        started=start('model_loading')
        tok=AutoTokenizer.from_pretrained(design['model'],local_files_only=True);tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(design['model'],torch_dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
        names=[f'model.layers.{l}.mlp.down_proj.weight' for l in design['layers']]
        store=CompactSessionMemory(model,names,source/'factors');end('model_loading',started)
        status('model_warmup')
        for _ in range(2):generate_answer_batch(model,tok,[panel[0]['question']],cap=1)
        sync()
        for n in SIZES:
            folder=out/('bank-'+str(n));folder.mkdir()
            started=start('our_delta_'+str(n));trials=[]
            for repeat in range(design['repeats']):
                order=list(panel);random.Random(100+repeat).shuffle(order)
                for q in order:
                    status('our_delta',n=n,repeat=repeat,qid=q['qid']);sync();arrival=time.monotonic()
                    observer=FirstAnswerClock(tok,arrival);keys=capture_keys(model,tok,names,[q['question']]);scores={}
                    for r in items[:n]:scores[r['sid']]=factor_scores(store.load(r['sid']),keys)[0]
                    ranked=sorted(scores,key=lambda sid:(-scores[sid],sid));sync();route_done=time.monotonic()
                    with observe_generate(model,observer):
                        final=store.run(ranked[0],lambda m:generate_answer_batch(m,tok,[q['question']]))[0]
                    sync();duration=time.monotonic()-arrival
                    trial=evaluated(q,final,duration,selected_sid=ranked[0]);trial.update(observer.result(),repeat=repeat,
                        rank=ranked.index(q['sid'])+1,route_seconds=route_done-arrival)
                    replay(trial,n,'our_delta');trials.append(trial);save(folder,'our_delta.json',dict(trials=trials))
            finish(n,'our_delta',trials,end('our_delta_'+str(n),started))
        store.restore();started=start('rag_index_loading')
        etok=AutoTokenizer.from_pretrained(design['embedding_model'],local_files_only=True)
        encoder=AutoModel.from_pretrained(design['embedding_model'],local_files_only=True).to('cuda:0').float().eval()
        vectors=torch.load(source/'qa_embeddings.pt',map_location='cpu',weights_only=True)
        if vectors.shape!=(512,384):raise ValueError('Unexpected source index shape')
        end('rag_index_loading',started,index_device=str(vectors.device));embed(encoder,etok,[panel[0]['question']])
        for n in SIZES:
            folder=out/('bank-'+str(n));started=start('rag_qa_'+str(n));trials=[]
            for repeat in range(design['repeats']):
                order=list(panel);random.Random(100+repeat).shuffle(order)
                for q in order:
                    status('rag_qa',n=n,repeat=repeat,qid=q['qid']);sync();arrival=time.monotonic();observer=FirstAnswerClock(tok,arrival)
                    query=embed(encoder,etok,[q['question']])[0];ranked=rank_vectors(query,vectors[:n],1);selected=bank[ranked[0][0]]
                    pids=tok.encode(bank_prefix(tok,[document(selected)]),add_special_tokens=False)
                    suffix=tok.encode(query_suffix(tok,q['question']),add_special_tokens=False)
                    if len(pids)+len(suffix)+64>model.config.max_position_embeddings:raise ValueError('RAG context limit')
                    cache=prefill(model,pids,chunk_size=256)
                    final,cache=decode_cached_observed(model,tok,cache,suffix,observer)
                    sync();duration=time.monotonic()-arrival
                    trial=evaluated(q,final,duration,sources=[selected['sid']]);trial.update(observer.result(),repeat=repeat,context_tokens=len(pids))
                    replay(trial,n,'rag_qa');trials.append(trial);cache=None;save(folder,'rag_qa.json',dict(trials=trials))
            finish(n,'rag_qa',trials,end('rag_qa_'+str(n),started))
        encoder=None;etok=None;vectors=None;gc.collect();torch.cuda.empty_cache()
        for n in SIZES:
            folder=out/('bank-'+str(n));status('kv_capacity',n=n);sync();cold_arrival=time.monotonic()
            pids=tok.encode(bank_prefix(tok,[document(r) for r in bank[:n]]),add_special_tokens=False)
            max_suffix=max(len(tok.encode(query_suffix(tok,q['question']),add_special_tokens=False)) for q in panel)
            plan=dict(prefix_tokens=len(pids),required_tokens=len(pids)+max_suffix+64,model_context_limit=model.config.max_position_embeddings,
                calculated_prefix_payload_bytes=len(pids)*kv_bytes_per_token(model.config))
            save(folder,'kv_capacity.json',plan)
            if plan['required_tokens']>plan['model_context_limit']:
                results[str(n)]['kv_full_qa']=dict(available=False,reason='context_limit',trials=[],summary=None,plan=plan)
                save(out,'evaluation_progress.json',dict(results=results));continue
            status('kv_setup',n=n)
            cache=prefill(model,pids,chunk_size=256,progress=lambda done,total:status('kv_setup',n=n,done=done,total=total))
            sync();setup_done=time.monotonic();prefix_bytes=cache_payload_bytes(cache)
            if prefix_bytes!=plan['calculated_prefix_payload_bytes']:raise ValueError('KV formula mismatch')
            save(folder,'kv_setup.json',dict(seconds=setup_done-cold_arrival,actual_prefix_payload_bytes=prefix_bytes))
            # Do not insert collection/cache clearing between cold prefix setup and its first answer.
            started=time.monotonic();trials=[];cold_first=None
            for repeat in range(design['repeats']):
                order=list(panel);random.Random(100+repeat).shuffle(order)
                for q in order:
                    status('kv_full_qa',n=n,repeat=repeat,qid=q['qid']);sync();arrival=time.monotonic();observer=FirstAnswerClock(tok,arrival)
                    suffix=tok.encode(query_suffix(tok,q['question']),add_special_tokens=False)
                    final,cache=decode_cached_observed(model,tok,cache,suffix,observer)
                    sync();duration=time.monotonic()-arrival
                    trial=evaluated(q,final,duration);trial.update(observer.result(),repeat=repeat,context_tokens=len(pids))
                    if not trials:
                        cold_first=dict(qid=q['qid'],ttft_seconds=observer.first_visible_at-cold_arrival,
                            prefix_setup_seconds=setup_done-cold_arrival,model_startup_included=False,samples=1)
                    replay(trial,n,'kv_full_qa');trials.append(trial);save(folder,'kv_full_qa.json',dict(trials=trials,cold_first_request=cold_first))
            finish(n,'kv_full_qa',trials,end('kv_full_qa_'+str(n),started),plan=plan,cold_first_request=cold_first)
            cache=None;gc.collect();torch.cuda.empty_cache()
        check_complete(results,design['panel_ids'],design['repeats']);status('integrity_after');store.restore();verify_source()
        if any(not torch.equal(model.get_parameter(name).detach().cpu(),base) for name,base in store.base.items()):raise ValueError('Base not restored')
        if any(sha256(root/n)!=h for n,h in code_hashes.items()) or sha256(out/'design.json')!=design_hash:raise ValueError('Code/protocol changed')
        save(out,'evaluation.json',dict(results=results,sizes=list(SIZES),panel_ids=design['panel_ids']))
        save(out,'completion.json',dict(completed=True,physical_gpu=1,gpu_uuid=gpu,source_unchanged=True,base_weights_restored=True,
            code_unchanged=True,protocol_unchanged=True,all_outputs_and_routes_replayed=True,real_sessions=512,fixed_original_queries=8,
            elapsed_seconds=time.monotonic()-begin,code_sha256=code_hashes))
        status('completed');print('COMPLETED TTFT REPLAY',flush=True)
    except BaseException as exc:
        save(out,'completion.json',dict(completed=False,stage='failed',error=repr(exc),physical_gpu=1));status('failed',error=repr(exc));raise
    finally:
        if store is not None:store.restore()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args();out=args.out.resolve()
    if not out.exists():prepare(args.source.resolve(),out)
    if not args.prepare_only:run(out)
