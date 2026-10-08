"""GPU1 exact-question scaling: fixed EOS deltas, matched dense RAG, full-bank KV.

Nested banks 24/32/48/64. Context/GPU capacity limits are explicit unavailable
baselines, never zero accuracy or silently truncated conversations.
"""
import argparse
import gc
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from experiment_larger_routing import read, save, sha256
from session_qa_routing_core import select_items, write_request
from question_key_k1_core import normalize, prompt
from answer_only_generation import generate_answer_batch, reference_tokens_present
from experiment_answer_termination_exact import activation_scores, routes
from experiment_memory_comparison import document, metrics, evaluated
from memory_comparison_core import (bank_prefix, query_suffix, kv_bytes_per_token,
                                    cache_payload_bytes, prefill, decode_cached,
                                    embed, rank_vectors)

SIZES = [24, 32, 48, 64]


def prepare(root, out):
    if out.exists(): raise ValueError('Require a new output directory')
    source = root/'results/answer-termination-exact-v1'
    old = read(source/'items.json')
    complete = read(source/'completion.json')
    if not complete.get('completed') or complete['selected_writer'] != 'answer_plus_eos':
        raise ValueError('Completed repaired writer required')
    official = read(root/'data/longmemeval_s.json')
    eligible = select_items(official, n=64, seed=0)
    seen = {r['source_session_id'] for r in old['test']}
    remaining = [r for r in eligible if r['source_session_id'] not in seen]
    random.Random(17).shuffle(remaining)
    items = old['test']+remaining
    if len(items) != 64 or len({r['source_session_id'] for r in items}) != 64:
        raise ValueError('Expected64distinct eligible sessions')
    prior = read(root/'results/evidence-routing-exact-v1/items.json')
    prior_split = {r['qid']: split for split, rows in prior.items() for r in rows}
    old_development = {r['qid'] for split in ['train', 'validation'] for r in old[split]}
    for r in items:
        r['history'] = ('previous_test' if r['source_session_id'] in seen else
                        'writer_development' if r['qid'] in old_development else
                        'previous_'+prior_split[r['qid']] if r['qid'] in prior_split else
                        'earlier_pilot')
        source_record = next(x for x in official if x['question_id'] == r['qid'])
        session = dict(zip(source_record['haystack_session_ids'],source_record['haystack_sessions']))[r['source_session_id']]
        if r['question'] != source_record['question'] or r['answer'] != source_record['answer'] or r['context'] != session:
            raise ValueError('Use literal original question/answer/full session')
    if len({prompt(r['question']) for r in items}) != 64: raise ValueError('Duplicate literal question keys')
    reusable = [r for r in items if (source/'deltas'/r['sid']/'v1.pt').exists()]
    new = [r for r in items if r not in reusable]
    projection = Path(read(source/'design.json')['projection'])
    design = dict(physical_gpu=1, seed=0, nested_bank_sizes=SIZES,
                  item_order='Previous24test items, then remaining eligible sessions shuffled with seed17',
                  source=str(source), delta_root=str(out/'deltas'), layers=[13,14,15],
                  model=str(root/'models/Llama-3.1-8B-Instruct'),
                  embedding_model=str(root/'models/all-MiniLM-L6-v2'), dtype='float32', batch_size=1,
                  projection=str(projection), projection_sha256=sha256(projection),
                  dataset_sha256=sha256(root/'data/longmemeval_s.json'),
                  source_hashes={n:sha256(source/n) for n in ['items.json','design.json','completion.json','evaluation.json']},
                  previous_comparison=str(root/'results/memory-comparison-v1'),
                  comparison_hashes={n:sha256(root/'results/memory-comparison-v1'/n) for n in ['evaluation.json','completion.json','resource_summary.json']},
                  reused_delta_sids=[r['sid'] for r in reusable], new_delta_sids=[r['sid'] for r in new],
                  writer='Frozen answer-plus-EOS AlphaEdit objective; full session + original Q/A;50steps,L2=.1; no new tuning, compression, or calibration',
                  router='Normalized three-layer delta activation energy; lexical sid tie break; question only; full-bank activation matrix',
                  rag='Same MiniLM384-dimensional normalized question-key dense index; top1 full session+supplied Q/A; base Llama generates',
                  secondary='Context-only MiniLM turn chunks192wordpieces,32overlap; top3chunks; no supplied Q/A annotations',
                  kv='Uncompressed float32 full-bank reusable prefix; chunk256; original query suffix; no truncation, no oracle selection',
                  kv_capacity='If prefix+longest question+64tokens exceeds131072, unavailable context_limit. CUDA OOM during genuine prefill becomes unavailable gpu_limit. No accuracy assigned.',
                  allocator='expandable_segments:True for all conditions; fresh24control is included because allocator differs from previous run',
                  recall='Exact original question only; greedy64token cap; EOS/next-turn stopping; no answer labels reach inference',
                  supervision='All primary methods receive full source sessions + supplied original Q/A at memory creation',
                  scope='Exploratory fixed-writer capacity stress test; uses previously inspected test/development/pilot sessions. Not a new held-out estimate or paraphrase/generalization test.',
                  capacity_scope='Only one supplied question/session tested; full-session write does not prove every fact retained by delta',
                  measures='Normalized delivered/emitted exact, verbatim exact, bounded containment, tokenF1, source top1/rank, matched24 anchor outcomes; lexical measures not semantic truth',
                  memory='Separate actual retained payload and tier, peak allocated/reserved GPU setup+recall, per-query decode KV actual or config-derived estimate; no new-write cost mixed into recall peaks',
                  latency='Bulk activation seconds/amortized and per-query generation measured separately; not a serving throughput benchmark',
                  accuracy_target='100% aspirational, not forecast',
                  new_write_expected_bytes=len(new)*704644657,
                  references=read(root/'results/memory-comparison-v1/design.json')['references'])
    if shutil.disk_usage(root).free < design['new_write_expected_bytes']+3*1024**3:
        raise RuntimeError('Insufficient disk space')
    out.mkdir();save(out,'items.json',items);save(out,'design.json',design)
    save(out,'manifest_sha256.json',{n:sha256(out/n) for n in ['items.json','design.json']})
    save(out,'completion.json',dict(completed=False,stage='prepared'))
    print('PREPARED',dict(sizes=SIZES,reused=len(reusable),new=len(new)),flush=True)


def bank_records(items):
    bank=list(items);random.Random(0).shuffle(bank)
    return [dict(record_id='record-'+str(i),sid=r['sid'],stored_question=r['question'],stored_answer=r['answer'],
                 conversation='\n'.join(t['role']+': '+t['content'] for t in r['context'])) for i,r in enumerate(bank)]


def run(root,out):
    design=read(out/'design.json');items=read(out/'items.json');source=Path(design['source']);delta_root=Path(design['delta_root'])
    frozen=read(out/'manifest_sha256.json')
    if read(out/'completion.json').get('completed'):raise ValueError('Completed runs immutable')
    if frozen!={n:sha256(out/n) for n in frozen}:raise ValueError('Protocol changed')
    from zsre_routing_experiment import require_gpu1
    gpu=require_gpu1()
    busy=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if busy.stdout.strip():raise RuntimeError('GPU1busy')
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    import torch
    from transformers import AutoTokenizer,AutoModel
    from routing_ladder_models import AlphaEditAdapter
    if torch.cuda.device_count()!=1:raise RuntimeError('Only physicalGPU1may be visible')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    codes=['experiment_memory_scaling.py','memory_comparison_core.py','answer_only_generation.py',
           'experiment_answer_termination_exact.py','experiment_memory_comparison.py','AlphaEditSessionStore.py',
           'routing_ladder_models.py','AlphaEdit/grounded_value.py','AlphaEdit/AlphaEdit_main.py','session_qa_routing_core.py']
    code_hashes={n:sha256(root/n) for n in codes}
    saved=read(source/'completion.json')['new_delta_sha256'];before={}
    delta_root.mkdir()
    for sid in design['reused_delta_sids']:
        directory=source/'deltas'/sid;before[sid]=sha256(directory/'v1.pt')
        if before[sid]!=saved[sid]:raise ValueError('Historical delta changed')
        (delta_root/sid).symlink_to(directory,target_is_directory=True)
    save(out,'provenance.json',dict(physical_gpu=1,gpu_uuid=gpu,code_sha256=code_hashes,
                                  torch_version=torch.__version__,cuda_version=torch.version.cuda,
                                  allocator=os.environ['PYTORCH_CUDA_ALLOC_CONF'],reused_delta_sha256=before))
    begin=time.monotonic();adapter=None;encoder=None;cache=None;sampler=None;results={};measurements={};writes=[]
    def status(stage,**extra):save(out,'status.json',dict(stage=stage,gpu_uuid=gpu,elapsed_seconds=time.monotonic()-begin,**extra))
    def start(stage,**kw):
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();status(stage,**kw)
        return time.monotonic()
    def end(stage,started,**kw):
        torch.cuda.synchronize()
        row=dict(elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(),end_allocated_bytes=torch.cuda.memory_allocated(),**kw)
        measurements[stage]=row;save(out,'measurements.json',measurements);return row
    def persist():save(out,'evaluation_progress.json',dict(results=results))
    save(out,'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    try:
        sampler=subprocess.Popen([sys.executable,str(root/'sample_session_qa_resources.py'),'--pid',str(os.getpid()),
                                  '--out',str(out),'--delta-root',str(delta_root)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        clock=start('model_loading')
        adapter=AlphaEditAdapter(root,delta_root,layers=design['layers'],projection_path=Path(design['projection']))
        model=adapter.model;tok=adapter.tok;store=adapter.store
        base_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        end('model_loading',clock,base_parameter_bytes=base_bytes)
        eos=tok.eos_token;eos_id=tok.convert_tokens_to_ids(eos)
        if eos_id not in model.generation_config.eos_token_id:raise ValueError('WriteEOSnot generation stop')
        save(out,'writer_config.json',dict(hparams=vars(adapter.hp),eos_token=eos,eos_id=eos_id))
        # Preflight every write before touching a new delta.
        new=[r for r in items if r['sid'] in design['new_delta_sids']]
        for i,r in enumerate(new):
            req=write_request(r,i);req['target_new']['str']=r['answer']+eos
            if tok.encode(' '+req['target_new']['str'],add_special_tokens=False)[-1]!=eos_id:raise ValueError('EOSmissing fromtarget')
            full=req['value_context_templates'][0][1].format(req['prompt']).format(req['subject'])+' '+req['target_new']['str']
            if len(tok.encode(full))+64>model.config.max_position_embeddings:raise ValueError('Full write context overcapacity')
        clock=start('additional_writes')
        for i,r in enumerate(new):
            status('additional_writes',done=i,total=len(new),qid=r['qid']);t=time.monotonic()
            req=write_request(r,i);req['target_new']['str']=r['answer']+eos
            store.write_session(r['sid'],tok,[req],adapter.hp,torch.zeros_like(adapter.P),adapter.P,1)
            final=store.run(r['sid'],1,lambda m:generate_answer_batch(m,tok,[r['question']],stop_at_turn=True))[0]
            file=delta_root/r['sid']/'v1.pt'
            row=dict(qid=r['qid'],sid=r['sid'],sha256=sha256(file),bytes=file.stat().st_size,
                     elapsed_seconds=time.monotonic()-t,immediate=final,immediate_exact=normalize(final['text'])==normalize(r['answer']),
                     immediate_contains=reference_tokens_present(final['text'],r['answer']))
            writes.append(row);save(out,'writes.json',writes)
            print('WRITE',i+1,len(new),r['qid'],'exact',row['immediate_exact'],flush=True)
        end('additional_writes',clock,new_count=len(writes),new_payload_bytes=sum(r['bytes'] for r in writes))
        store.deactivate();adapter.P=None;gc.collect();torch.cuda.empty_cache()
        clock=start('activation_matrix')
        scores=activation_scores(model,tok,store,items,delta_root)
        save(out,'activation_scores.json',scores)
        end('activation_matrix',clock,questions=64,deltas=64,pairs=4096)
        # The matrix is computed once, then each nested bank uses only its own columns.
        for k in SIZES:
            rows=items[:k];allowed={r['sid'] for r in rows}
            sub={r['qid']:{sid:value for sid,value in scores[r['qid']].items() if sid in allowed} for r in rows}
            routed=routes(rows,sub);folder=out/('bank-'+str(k));folder.mkdir();save(folder,'our_routing.json',routed)
            clock=start('our_delta_'+str(k),bank_size=k);trials=[]
            for index,(r,route) in enumerate(zip(rows,routed)):
                status('our_delta',bank_size=k,done=index,total=k);t=time.monotonic()
                final=store.run(route['selected_sid'],1,lambda m:generate_answer_batch(m,tok,[r['question']],stop_at_turn=True))[0]
                final['max_query_cache_payload_bytes']=(len(tok.encode(prompt(r['question'])))+final['tokens']-1)*kv_bytes_per_token(model.config)
                final['cache_measurement']='Model-config estimate; generation API does not return its cache'
                trial=evaluated(r,final,time.monotonic()-t,selected_sid=route['selected_sid'])
                trial.update(rank=route['rank'],routing_correct=route['routing_correct'],history=r['history'])
                trials.append(trial);save(folder,'our_delta.json',dict(trials=trials))
            resource=end('our_delta_'+str(k),clock,persistent_payload_bytes=sum((delta_root/r['sid']/'v1.pt').stat().st_size for r in rows),
                         persistent_tier='Disk dense deltas; one loaded at a time',additional_model_bytes=0,
                         cpu_base_backup_bytes=sum(t.numel()*t.element_size() for t in store._base.values()))
            summary=metrics(trials);summary.update(mean_rank=sum(t['rank'] for t in trials)/k,mrr=sum(1/t['rank'] for t in trials)/k)
            results[str(k)]={'our_delta':dict(available=True,trials=trials,summary=summary,resources=resource)};persist()
            print('OUR',k,'route',summary['retrieval_at1']['count'],'exact',summary['returned_exact']['count'],flush=True)
        store.deactivate();gc.collect();torch.cuda.empty_cache()
        clock=start('rag_index_creation')
        etok=AutoTokenizer.from_pretrained(design['embedding_model'],local_files_only=True)
        encoder=AutoModel.from_pretrained(design['embedding_model'],local_files_only=True).to('cuda:0').float().eval()
        embbytes=sum(p.numel()*p.element_size() for p in encoder.parameters())
        for k in SIZES:
            rows=items[:k];folder=out/('bank-'+str(k));bank=bank_records(rows);save(folder,'bank.json',bank)
            qvec=embed(encoder,etok,[r['stored_question'] for r in bank]);torch.save(qvec,folder/'qa_embeddings.pt')
            chunks=[];lookup={r['sid']:r for r in rows}
            for r in bank:
                for turn in lookup[r['sid']]['context']:
                    ids=etok.encode(turn['role']+': '+turn['content'],add_special_tokens=False)
                    for offset in range(0,len(ids),160):
                        piece=ids[offset:offset+192]
                        if not piece:continue
                        chunks.append(dict(sid=r['sid'],text=etok.decode(piece,skip_special_tokens=True)))
                        if offset+192>=len(ids):break
            save(folder,'context_chunks.json',chunks);cvec=embed(encoder,etok,[r['text'] for r in chunks]);torch.save(cvec,folder/'context_embeddings.pt')
            status('rag_index_creation',bank_size=k,context_chunks=len(chunks))
        end('rag_index_creation',clock,embedding_parameter_bytes=embbytes)
        for k in SIZES:
            rows=items[:k];folder=out/('bank-'+str(k));bank=read(folder/'bank.json');chunks=read(folder/'context_chunks.json')
            qvec=torch.load(folder/'qa_embeddings.pt',weights_only=True);cvec=torch.load(folder/'context_embeddings.pt',weights_only=True)
            for method in ['rag_qa','rag_context_only']:
                clock=start(method+'_'+str(k),bank_size=k);trials=[]
                for index,r in enumerate(rows):
                    status(method,bank_size=k,done=index,total=k);t=time.monotonic();q=embed(encoder,etok,[r['question']])[0]
                    if method=='rag_qa':
                        ranked=rank_vectors(q,qvec,1);selected=[bank[i] for i,_ in ranked];docs=[document(x) for x in selected]
                    else:
                        ranked=rank_vectors(q,cvec,3);selected=[chunks[i] for i,_ in ranked];docs=['Conversation excerpt:\n'+x['text'] for x in selected]
                    pids=tok.encode(bank_prefix(tok,docs),add_special_tokens=False);suffix=tok.encode(query_suffix(tok,r['question']),add_special_tokens=False)
                    if len(pids)+len(suffix)+64>model.config.max_position_embeddings:raise ValueError('RAGcontext overcapacity')
                    cache=prefill(model,pids,chunk_size=256);final,cache=decode_cached(model,tok,cache,suffix)
                    trial=evaluated(r,final,time.monotonic()-t,sources=[x['sid'] for x in selected])
                    trial.update(retrieval_scores=[score for _,score in ranked],context_tokens=len(pids),history=r['history'])
                    trials.append(trial);cache=None;gc.collect();save(folder,method+'.json',dict(trials=trials))
                files=['bank.json','qa_embeddings.pt'] if method=='rag_qa' else ['context_chunks.json','context_embeddings.pt']
                resource=end(method+'_'+str(k),clock,persistent_payload_bytes=sum((folder/n).stat().st_size for n in files),
                             persistent_tier='CPU/disk text and vector index',additional_model_bytes=embbytes)
                results[str(k)][method]=dict(available=True,trials=trials,summary=metrics(trials),resources=resource);persist()
                print(method,k,'exact',results[str(k)][method]['summary']['returned_exact']['count'],flush=True)
        encoder=None;etok=None;gc.collect();torch.cuda.empty_cache()
        # Full-bank KV: real runs when within context; explicit capacity result otherwise.
        for k in SIZES:
            rows=items[:k];folder=out/('bank-'+str(k));bank=read(folder/'bank.json')
            prefix=bank_prefix(tok,[document(r) for r in bank]);pids=tok.encode(prefix,add_special_tokens=False)
            suffixes={r['qid']:tok.encode(query_suffix(tok,r['question']),add_special_tokens=False) for r in rows}
            expected=len(pids)*kv_bytes_per_token(model.config);required=len(pids)+max(map(len,suffixes.values()))+64
            plan=dict(bank_size=k,prefix_tokens=len(pids),required_tokens=required,context_limit=model.config.max_position_embeddings,
                      expected_payload_bytes=expected,bytes_per_token=kv_bytes_per_token(model.config))
            save(folder,'kv_plan.json',plan)
            if required>model.config.max_position_embeddings:
                results[str(k)]['kv_full_qa']=dict(available=False,reason='context_limit',plan=plan,trials=[],summary=None,resources=None)
                persist();print('KV_CAPACITY',k,'context_limit',required,flush=True);continue
            for r in rows:
                if tok.encode(prefix+query_suffix(tok,r['question']),add_special_tokens=False)!=pids+suffixes[r['qid']]:raise ValueError('Prefix/suffix tokenization mismatch')
            clock=start('kv_build_'+str(k),bank_size=k)
            try:
                cache=prefill(model,pids,chunk_size=256,progress=lambda done,total:status('kv_build',bank_size=k,done=done,total=total))
            except torch.OutOfMemoryError as exc:
                attempt=end('kv_failed_build_'+str(k),clock)
                results[str(k)]['kv_full_qa']=dict(available=False,reason='gpu_limit',plan=plan,trials=[],summary=None,resources=attempt,error=str(exc))
                cache=None;gc.collect();torch.cuda.empty_cache();persist();print('KV_CAPACITY',k,'gpu_limit',flush=True);continue
            actual=cache_payload_bytes(cache)
            if actual!=expected:raise ValueError('KV tensor payload disagrees with formula')
            build=end('kv_build_'+str(k),clock,persistent_payload_bytes=actual)
            clock=start('kv_full_qa_'+str(k),bank_size=k);trials=[]
            for index,r in enumerate(rows):
                status('kv_full_qa',bank_size=k,done=index,total=k);t=time.monotonic()
                final,cache=decode_cached(model,tok,cache,suffixes[r['qid']])
                trial=evaluated(r,final,time.monotonic()-t);trial.update(context_tokens=len(pids),history=r['history'])
                trials.append(trial);save(folder,'kv_full_qa.json',dict(trials=trials))
            resource=end('kv_full_qa_'+str(k),clock,persistent_payload_bytes=actual,persistent_tier='GPU reusable full-bank cache',additional_model_bytes=0,
                         setup_peak_allocated_bytes=build['peak_allocated_bytes'],setup_peak_reserved_bytes=build['peak_reserved_bytes'])
            results[str(k)]['kv_full_qa']=dict(available=True,plan=plan,trials=trials,summary=metrics(trials),resources=resource,
                                            payload_matches_formula=True,cache_prefix_reused=True,query_tails_cropped=True)
            cache=None;gc.collect();torch.cuda.empty_cache();persist()
            print('KV',k,'exact',results[str(k)]['kv_full_qa']['summary']['returned_exact']['count'],flush=True)
        # Keep the old24questions as paired anchors at every scale, separately from new entrants.
        anchor_ids={r['qid'] for r in items[:24]}
        for k in SIZES:
            for method,cell in results[str(k)].items():
                if not cell['available']:continue
                cell['anchor24_summary']=metrics([t for t in cell['trials'] if t['qid'] in anchor_ids])
                for history in sorted({t['history'] for t in cell['trials']}):
                    cell.setdefault('history_summaries',{})[history]=metrics([t for t in cell['trials'] if t['history']==history])
        save(out,'evaluation.json',dict(results=results,nested_bank_sizes=SIZES,scope=design['scope']))
        status('integrity_checks');store.deactivate()
        for name,base in store._base.items():
            if not torch.equal(model.get_parameter(name).detach().cpu(),base):raise ValueError('Base weights not restored')
        after={r['sid']:sha256(delta_root/r['sid']/'v1.pt') for r in items}
        if any(after[sid]!=value for sid,value in before.items()):raise ValueError('Reused deltas changed')
        if any(after[r['sid']]!=r['sha256'] for r in writes):raise ValueError('New deltas changed')
        for n,v in code_hashes.items():
            if sha256(root/n)!=v:raise ValueError('Code changed during run')
        for n,v in design['source_hashes'].items():
            if sha256(source/n)!=v:raise ValueError('Historical source changed')
        for n,v in design['comparison_hashes'].items():
            if sha256(Path(design['previous_comparison'])/n)!=v:raise ValueError('Previous comparison changed')
        if sha256(Path(design['projection']))!=design['projection_sha256']:raise ValueError('Projection changed')
        if frozen!={n:sha256(out/n) for n in frozen}:raise ValueError('Protocol changed')
        expected_answers=sum(SIZES)
        for method in ['our_delta','rag_qa','rag_context_only']:
            if sum(len(results[str(k)][method]['trials']) for k in SIZES)!=expected_answers:raise ValueError('Incomplete answer set')
        save(out,'resource_summary.json',dict(elapsed_seconds=time.monotonic()-begin,base_model_parameter_bytes=base_bytes,
                                             embedding_parameter_bytes=embbytes,measurements=measurements,
                                             new_write_bytes=sum(r['bytes'] for r in writes),new_write_count=len(writes),
                                             unique_bank_payload_bytes=sum((delta_root/r['sid']/'v1.pt').stat().st_size for r in items),
                                             scope='New-write setup reported separately; plots compare inference/prefill/index peaks, not AlphaEdit writing'))
        save(out,'completion.json',dict(completed=True,physical_gpu=1,gpu_uuid=gpu,nested_bank_sizes=SIZES,
                                       own_rag_secondary_answers_each=expected_answers,new_writes=len(writes),reused_deltas=len(before),
                                       kv_status={str(k):'completed' if results[str(k)]['kv_full_qa']['available'] else results[str(k)]['kv_full_qa']['reason'] for k in SIZES},
                                       source_artifacts_unchanged=True,delta_hashes_unchanged=True,base_weights_restored=True,
                                       protocol_unchanged=True,frozen_manifest=frozen,delta_sha256=after))
        status('completed');print('COMPLETED',flush=True)
    except BaseException as exc:
        save(out,'completion.json',dict(completed=False,stage='failed',error=repr(exc),physical_gpu=1));status('failed',error=repr(exc));raise
    finally:
        if adapter is not None:adapter.store.deactivate()
        if sampler is not None:sampler.terminate()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('results/memory-scaling-v1'));p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parent;out=a.out.resolve()
    if not out.exists():prepare(root,out)
    if not a.prepare_only:run(root,out)


if __name__=='__main__':main()
