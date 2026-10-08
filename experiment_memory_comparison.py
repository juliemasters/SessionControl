"""Measured GPU1 comparison: dense AlphaEdit deltas, dense RAG, full-bank KV.

Primary baselines receive the same sessions + supplied Q/A at memory creation.
Context-only RAG is reported separately. Exact original questions only.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from experiment_larger_routing import read,save,sha256
from question_key_k1_core import normalize,interval
from answer_only_generation import reference_tokens_present,generate_answer_batch
from experiment_answer_termination_exact import activation_scores,routes
from memory_comparison_core import (bank_prefix,query_suffix,kv_bytes_per_token,cache_payload_bytes,
                                    prefill,decode_cached,embed,rank_vectors,token_f1)


def prepare(root,out,source):
    if out.exists():raise ValueError('Require a new output directory')
    done=read(source/'completion.json')
    if not done.get('completed') or done['selected_writer']!='answer_plus_eos':raise ValueError('Require completed repaired baseline')
    items=read(source/'items.json')['test']
    bank=list(items);random.Random(0).shuffle(bank)
    records=[dict(record_id='record-'+str(i),sid=r['sid'],stored_question=r['question'],stored_answer=r['answer'],
                  conversation='\n'.join(t['role']+': '+t['content'] for t in r['context'])) for i,r in enumerate(bank)]
    design=dict(source=str(source),physical_gpu=1,seed=0,dtype='float32',batch_size=1,question_count=24,
                model=str(root/'models/Llama-3.1-8B-Instruct'),embedding_model=str(root/'models/all-MiniLM-L6-v2'),
                delta_root=str(source/'deltas'),layers=[13,14,15],
                source_hashes={name:sha256(source/name) for name in ['items.json','design.json','evaluation.json','completion.json']},
                methods=['our_delta','rag_qa','kv_full_qa','rag_context_only'],
                matched_information='Primary3methods receive full source sessions and supplied Q/A at memory creation; recall only uses original question',
                kv='Uncompressed float32 reusable full-bank prefix; all24records; no oracle session selection; crop query tail between recalls',
                rag='MiniLM mean-pooled normalized embeddings of supplied original question keys; top1 full session+QA record; base Llama generates answer',
                rag_secondary='Context-only dense retrieval across all user+assistant turn chunks;192wordpieces,32overlap; top3chunks; no stored questions or answers',
                own='Reuse verified independent answer-plus-EOS dense deltas; fresh normalized activation scores and24single-question finals',
                generation='Greedy,64token cap, per-answer next-turn stopping; baselines chat prompt; own bare original write key',
                retrieval_labels='Source IDs and references are evaluation-only; ranking uses vector cosine and query text',
                cache_prefill_chunk=256,no_truncation=True,
                test_scope='Exploratory same24previously inspected original-question items; not unseen-query or generalization accuracy',
                accuracy='Normalized whole delivered and emitted answer exact match, verbatim exact, mean normalized tokenF1, token-bounded containment',
                memory='Separate retained bank representation payload, physical storage tier, measured inference peak allocated/reserved GPU bytes, decoding KV bytes, retriever parameter bytes',
                persistent_payload_excludes='Common LLM weights; RAG retriever weights reported separately and included in actual GPU peaks',
                cache_note='All3methods use transient decoding KV. KVbaseline additionally retains full memory bank KV across questions.',
                expected_memory='Exact model-config KV bytes/token times encoded prefix length; dense delta bytes/session; actual serialized RAG corpus+index',
                accuracy_target='100% aspiration; no unmeasured baseline accuracy forecasts',
                references=dict(kv='https://huggingface.co/docs/transformers/kv_cache',
                                embedding='https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
                                rag='https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html'))
    out.mkdir(parents=True);save(out,'items.json',items);save(out,'bank.json',records);save(out,'design.json',design)
    save(out,'manifest_sha256.json',{n:sha256(out/n) for n in ['items.json','bank.json','design.json']})
    save(out,'completion.json',dict(completed=False,stage='prepared'))


def document(record):
    return record['record_id']+'\nConversation:\n'+record['conversation']+'\nStored question: '+record['stored_question']+'\nStored answer: '+record['stored_answer']


def metrics(trials):
    n=len(trials);summary=dict(n=n)
    for key in ['returned_exact','emitted_exact','verbatim_exact','contains_reference','hit_cap','eos']:
        count=sum(t[key] for t in trials);summary[key]=dict(count=count,rate=count/n,ci95=list(interval(count,n)))
    for key in ['retrieval_at1','retrieval_at3']:
        valid=[t[key] for t in trials if t[key] is not None]
        summary[key]=None if not valid else dict(count=sum(valid),rate=sum(valid)/len(valid),ci95=list(interval(sum(valid),len(valid))))
    summary.update(mean_token_f1=sum(t['token_f1'] for t in trials)/n,
                   mean_output_tokens=sum(t['final']['tokens'] for t in trials)/n,
                   mean_query_seconds=sum(t['query_seconds'] for t in trials)/n,
                   max_query_cache_payload_bytes=max(t['final'].get('max_query_cache_payload_bytes',0) for t in trials))
    return summary


def evaluated(item,final,seconds,*,sources=None,selected_sid=None):
    # Reference values and source labels enter only after inference returns.
    sources=sources or []
    return dict(qid=item['qid'],question=item['question'],answer=item['answer'],final=final,
                query_seconds=seconds,selected_sid=selected_sid,reference_source_sid=item['sid'],retrieved_sids=sources,
                retrieval_at1=(sources[0]==item['sid']) if sources else (selected_sid==item['sid'] if selected_sid else None),
                retrieval_at3=(item['sid'] in sources[:3]) if sources else (selected_sid==item['sid'] if selected_sid else None),
                returned_exact=normalize(final['text'])==normalize(item['answer']),
                emitted_exact=normalize(final['emitted_text'])==normalize(item['answer']),
                verbatim_exact=final['text']==item['answer'],token_f1=token_f1(final['text'],item['answer']),
                contains_reference=reference_tokens_present(final['text'],item['answer']),
                hit_cap=final['hit_cap'],eos=final['finish_reason']=='eos')


def run(root,out):
    design=read(out/'design.json');items=read(out/'items.json');bank=read(out/'bank.json');source=Path(design['source'])
    frozen=read(out/'manifest_sha256.json')
    if frozen!={n:sha256(out/n) for n in frozen}:raise ValueError('Protocol changed')
    if read(out/'completion.json').get('completed'):raise ValueError('Completed runs are immutable')
    for n,v in design['source_hashes'].items():
        if sha256(source/n)!=v:raise ValueError('Historical source changed')
    from zsre_routing_experiment import require_gpu1
    gpu=require_gpu1()
    busy=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
    if busy.stdout.strip():raise RuntimeError('PhysicalGPU1busy')
    import torch
    from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
    from AlphaEditSessionStore import SessionFFNMemory
    if torch.cuda.device_count()!=1:raise RuntimeError('Require onlyphysicalGPU1visible')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    codes=['experiment_memory_comparison.py','memory_comparison_core.py','answer_only_generation.py','experiment_answer_termination_exact.py','AlphaEditSessionStore.py']
    code_hashes={n:sha256(root/n) for n in codes}
    embpath=Path(design['embedding_model']);embedding_hashes={str(p.relative_to(embpath)):sha256(p) for p in embpath.rglob('*') if p.is_file() and '.cache' not in p.parts}
    saved_delta=read(source/'completion.json')['new_delta_sha256'];delta_root=Path(design['delta_root'])
    before={r['sid']:sha256(delta_root/r['sid']/'v1.pt') for r in items}
    if any(v!=saved_delta[sid] for sid,v in before.items()):raise ValueError('Historical delta changed')
    save(out,'provenance.json',dict(physical_gpu=1,gpu_uuid=gpu,code_sha256=code_hashes,embedding_sha256=embedding_hashes,
                                   torch_version=torch.__version__,cuda_version=torch.version.cuda,tf32=torch.backends.cuda.matmul.allow_tf32))
    begin=time.monotonic();model=None;store=None;encoder=None;cache=None;sampler=None;results={};measurements={}
    def status(stage,**kw):save(out,'status.json',dict(stage=stage,gpu_uuid=gpu,elapsed_seconds=time.monotonic()-begin,**kw))
    def stage_start(stage):
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        status(stage);return time.monotonic()
    def stage_end(stage,started,**extra):
        torch.cuda.synchronize()
        row=dict(elapsed_seconds=time.monotonic()-started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 end_allocated_bytes=torch.cuda.memory_allocated(),**extra)
        measurements[stage]=row;save(out,'measurements.json',measurements);return row
    save(out,'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    try:
        sampler=subprocess.Popen([sys.executable,str(root/'sample_session_qa_resources.py'),'--pid',str(__import__('os').getpid()),
                                  '--out',str(out),'--delta-root',str(delta_root)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        start=stage_start('llama_loading')
        tok=AutoTokenizer.from_pretrained(design['model'],local_files_only=True);tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(design['model'],torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
        base_bytes=sum(p.numel()*p.element_size() for p in model.parameters());stage_end('llama_loading',start,base_parameter_bytes=base_bytes)
        start=stage_start('rag_index_creation')
        etok=AutoTokenizer.from_pretrained(embpath,local_files_only=True)
        encoder=AutoModel.from_pretrained(embpath,local_files_only=True).to('cuda:0').float().eval()
        embbytes=sum(p.numel()*p.element_size() for p in encoder.parameters())
        qvectors=embed(encoder,etok,[r['stored_question'] for r in bank]);torch.save(qvectors,out/'qa_embeddings.pt')
        chunks=[]
        for r in bank:
            for turn in next(x for x in items if x['sid']==r['sid'])['context']:
                ids=etok.encode(turn['role']+': '+turn['content'],add_special_tokens=False)
                for offset in range(0,len(ids),160):
                    piece=ids[offset:offset+192]
                    if not piece:continue
                    chunks.append(dict(sid=r['sid'],text=etok.decode(piece,skip_special_tokens=True)))
                    if offset+192>=len(ids):break
        save(out,'context_chunks.json',chunks)
        cvectors=embed(encoder,etok,[r['text'] for r in chunks]);torch.save(cvectors,out/'context_embeddings.pt')
        stage_end('rag_index_creation',start,embedding_parameter_bytes=embbytes,qa_vectors=24,context_vectors=len(chunks))
        for method in ['rag_qa','rag_context_only']:
            start=stage_start(method);trials=[];max_cache=0
            for index,item in enumerate(items):
                status(method,done=index,total=24);clock=time.monotonic()
                q=embed(encoder,etok,[item['question']])[0]
                if method=='rag_qa':
                    ranked=rank_vectors(q,qvectors,1)
                    selected=[bank[i] for i,_ in ranked];docs=[document(r) for r in selected];sids=[r['sid'] for r in selected]
                else:
                    ranked=rank_vectors(q,cvectors,3)
                    selected=[chunks[i] for i,_ in ranked];docs=['Conversation excerpt:\n'+r['text'] for r in selected];sids=[r['sid'] for r in selected]
                prefix=bank_prefix(tok,docs);suffix=query_suffix(tok,item['question'])
                pids=tok.encode(prefix,add_special_tokens=False);sids_tokens=tok.encode(suffix,add_special_tokens=False)
                if len(pids)+len(sids_tokens)+64>model.config.max_position_embeddings:raise ValueError('RAGcontext exceedscapacity')
                cache=prefill(model,pids,chunk_size=design['cache_prefill_chunk'])
                prefix_bytes=cache_payload_bytes(cache)
                final,cache=decode_cached(model,tok,cache,sids_tokens)
                max_cache=max(max_cache,final['max_query_cache_payload_bytes'])
                trial=evaluated(item,final,time.monotonic()-clock,sources=sids)
                trial.update(retrieval_scores=[score for _,score in ranked],context_tokens=len(pids),prefix_cache_bytes=prefix_bytes)
                trials.append(trial);cache=None;gc.collect()
                save(out,method+'.json',dict(trials=trials))
                print(method,index+1,'exact',trial['returned_exact'],flush=True)
            files=['bank.json','qa_embeddings.pt'] if method=='rag_qa' else ['context_chunks.json','context_embeddings.pt']
            persistent=sum((out/f).stat().st_size for f in files)
            resource=stage_end(method,start,persistent_payload_bytes=persistent,persistent_tier='CPU text/index files',
                               additional_model_bytes=embbytes,max_decode_kv_bytes=max_cache,storage_files=files)
            results[method]=dict(trials=trials,summary=metrics(trials),resources=resource)
            save(out,'evaluation_progress.json',results)
        encoder=None;etok=None;gc.collect();torch.cuda.empty_cache()
        # All memory records are encoded into one actual reusable cache, not an oracle session.
        start=stage_start('kv_full_qa_build')
        prefix=bank_prefix(tok,[document(r) for r in bank]);prefix_ids=tok.encode(prefix,add_special_tokens=False)
        suffixes={r['qid']:tok.encode(query_suffix(tok,r['question']),add_special_tokens=False) for r in items}
        if len(prefix_ids)+max(map(len,suffixes.values()))+64>model.config.max_position_embeddings:raise ValueError('Whole bank exceeds model context; no truncation')
        for r in items:
            if tok.encode(prefix+query_suffix(tok,r['question']),add_special_tokens=False)!=prefix_ids+suffixes[r['qid']]:
                raise ValueError('Cache prefix/suffix tokenization differs from full prompt')
        expected_cache=len(prefix_ids)*kv_bytes_per_token(model.config)
        save(out,'kv_plan.json',dict(prefix_tokens=len(prefix_ids),expected_payload_bytes=expected_cache,
                                      bytes_per_token=kv_bytes_per_token(model.config),max_context=model.config.max_position_embeddings))
        cache=prefill(model,prefix_ids,chunk_size=design['cache_prefill_chunk'],
                      progress=lambda done,total:status('kv_full_qa_build',done=done,total=total))
        actual_cache=cache_payload_bytes(cache)
        if actual_cache!=expected_cache:raise ValueError('Actual KV tensors disagree with config formula')
        stage_end('kv_full_qa_build',start,persistent_payload_bytes=actual_cache,prefix_tokens=len(prefix_ids),expected_payload_bytes=expected_cache)
        start=stage_start('kv_full_qa');trials=[];max_cache=0
        for index,item in enumerate(items):
            status('kv_full_qa',done=index,total=24);clock=time.monotonic()
            final,cache=decode_cached(model,tok,cache,suffixes[item['qid']])
            max_cache=max(max_cache,final['max_query_cache_payload_bytes'])
            trial=evaluated(item,final,time.monotonic()-clock)
            trial['context_tokens']=len(prefix_ids);trials.append(trial)
            save(out,'kv_full_qa.json',dict(trials=trials))
            print('kv_full_qa',index+1,'exact',trial['returned_exact'],flush=True)
        resource=stage_end('kv_full_qa',start,persistent_payload_bytes=actual_cache,persistent_tier='GPU reusable cache',
                           additional_model_bytes=0,max_decode_kv_bytes=max_cache,
                           preprocessing_peak_allocated_bytes=measurements['kv_full_qa_build']['peak_allocated_bytes'])
        results['kv_full_qa']=dict(trials=trials,summary=metrics(trials),resources=resource);save(out,'evaluation_progress.json',results)
        cache=None;gc.collect();torch.cuda.empty_cache()
        start=stage_start('our_delta')
        names=[f'model.layers.{l}.mlp.down_proj.weight' for l in design['layers']]
        store=SessionFFNMemory(model,names,delta_root)
        scores=activation_scores(model,tok,store,items,delta_root)
        routed=routes(items,scores);save(out,'our_routing.json',routed)
        trials=[];max_cache=0
        # Activation is batched across all questions; report its amortized cost separately.
        routing_seconds=time.monotonic()-start
        for index,(item,route) in enumerate(zip(items,routed)):
            status('our_delta',done=index,total=24);clock=time.monotonic()
            final=store.run(route['selected_sid'],1,lambda m:generate_answer_batch(m,tok,[item['question']],stop_at_turn=True))[0]
            input_tokens=len(tok.encode('Q: '+item['question']+'\nA:'))
            final['max_query_cache_payload_bytes']=(input_tokens+final['tokens']-1)*kv_bytes_per_token(model.config)
            final['cache_measurement']='Exact model-config formula; own generation API does not return its cache'
            max_cache=max(max_cache,final['max_query_cache_payload_bytes'])
            trial=evaluated(item,final,time.monotonic()-clock,selected_sid=route['selected_sid'])
            trial.update(rank=route['rank'],routing_correct=route['routing_correct']);trials.append(trial)
            save(out,'our_delta.json',dict(trials=trials));print('our_delta',index+1,'exact',trial['returned_exact'],flush=True)
        persistent=sum((delta_root/r['sid']/'v1.pt').stat().st_size for r in items)
        resource=stage_end('our_delta',start,persistent_payload_bytes=persistent,persistent_tier='Disk delta files; one loaded at a time',
                           additional_model_bytes=0,max_decode_kv_bytes=max_cache,routing_seconds=routing_seconds,
                           cpu_base_backup_bytes=sum(t.numel()*t.element_size() for t in store._base.values()),
                           write_cost_scope='Existing writes reused; historical write+immediate-check cost is not part of inference peak')
        results['our_delta']=dict(trials=trials,summary=metrics(trials),resources=resource)
        save(out,'evaluation.json',dict(results=results,matched_methods=['our_delta','rag_qa','kv_full_qa'],secondary='rag_context_only',scope=design['test_scope']))
        # Memory scaling points are calculations, not unrun accuracy experiments.
        scaling=[]
        for k in [1,8,16,24]:
            text=bank_prefix(tok,[document(r) for r in bank[:k]])
            kv_tokens=len(tok.encode(text,add_special_tokens=False))
            rag_text=len(json.dumps(bank[:k],indent=2,ensure_ascii=False).encode())
            scaling.append(dict(sessions=k,kv_prefix_tokens=kv_tokens,expected_kv_bytes=kv_tokens*kv_bytes_per_token(model.config),
                                expected_dense_delta_bytes=sum((delta_root/r['sid']/'v1.pt').stat().st_size for r in bank[:k]),
                                expected_rag_payload_bytes=rag_text+k*384*4,
                                note='Expected payload from actual encoded bank/config; only K24has measured answer accuracy'))
        save(out,'expected_memory_scaling.json',scaling)
        store.deactivate()
        for name,base in store._base.items():
            if not torch.equal(model.get_parameter(name).detach().cpu(),base):raise ValueError('Base weights not restored')
        status('integrity_checks')
        after={sid:sha256(delta_root/sid/'v1.pt') for sid in before}
        if after!=before:raise ValueError('Source deltas changed')
        for name,value in code_hashes.items():
            if sha256(root/name)!=value:raise ValueError('Code changed')
        for name,value in design['source_hashes'].items():
            if sha256(source/name)!=value:raise ValueError('Historical source changed')
        if frozen!={n:sha256(out/n) for n in frozen}:raise ValueError('Protocol changed')
        resource=dict(elapsed_seconds=time.monotonic()-begin,base_model_parameter_bytes=base_bytes,
                      embedding_parameter_bytes=embbytes,methods=measurements,
                      measurement_scope='Inference/prefill/index creation only; no new AlphaEdit writes; all methods float32, same Llama, batch1',
                      peak_definitions='PyTorch max_memory_allocated is live tensors; max_memory_reserved is allocator blocks. Bank payload bytes are separate and stored in different tiers.')
        save(out,'resource_summary.json',resource)
        save(out,'completion.json',dict(completed=True,physical_gpu=1,gpu_uuid=gpu,methods_completed=4,answers_per_method=24,
             sources_unchanged=True,delta_hashes_unchanged=True,base_weights_restored=True,
             kv_payload_matches_formula=True,kv_prefix_is_reused=True,cache_tail_cropped_between_queries=True,
             frozen_manifest=frozen,delta_sha256=after,test_is_fresh=False))
        status('completed');print('COMPLETED',json.dumps({k:v['summary'] for k,v in results.items()}),flush=True)
    except BaseException as exc:
        save(out,'completion.json',dict(completed=False,stage='failed',physical_gpu=1,error=repr(exc)));status('failed',error=repr(exc));raise
    finally:
        if store is not None:store.deactivate()
        if sampler is not None:
            sampler.terminate()
            try:sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:sampler.kill()
        cache=None;store=None;encoder=None;model=None;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('results/memory-comparison-v1'))
    p.add_argument('--source',type=Path,default=Path('results/answer-termination-exact-v1'));p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parent;out=args.out.resolve()
    if not (out/'design.json').exists():prepare(root,out,args.source.resolve())
    if args.prepare_only:print('PREPARED',out);return
    run(root,out)


if __name__=='__main__':main()
