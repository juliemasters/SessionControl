"""Measured real-session scaling on physical GPU 1, with validated compact deltas.

The added 448 conversations are untouched LongMemEval distractor transcripts.
Their new labels are uniquely identifiable extractive quote-completion Q/A,
not official benchmark questions. Evaluation uses eight original pilot questions.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

from experiment_utils import read, save, sha256
from fixed_query_scaling_core import SIZES, nested_bank, latency_summary, check_complete
from experiment_utils import document, evaluated, metrics
from memory_comparison_core import (bank_prefix, query_suffix, kv_bytes_per_token,
    cache_payload_bytes, prefill, decode_cached, embed, rank_vectors)
from answer_only_generation import generate_answer_batch
from question_key import prompt


def prepare(data_root, out):
    if out.exists():raise ValueError('New output directory required')
    source=data_root/'results/memory-scaling-v1'
    done=read(source/'completion.json')
    if not done.get('completed') or not done.get('base_weights_restored'):
        raise ValueError('Require completed historical bank')
    originals,panel=nested_bank(read(source/'items.json'),read(data_root/'results/session-qa-routing-v2/items.json'))
    raw=read(data_root/'data/longmemeval_s.json')
    official={r['question_id']:r for r in raw}
    sessions={}
    for record in raw:
        for sid,context in zip(record['haystack_session_ids'],record['haystack_sessions']):
            if sid in sessions and sessions[sid]!=context:raise ValueError('Inconsistent source session')
            sessions[sid]=context
    for r in originals:
        parent=official[r['qid']]
        if r['question']!=parent['question'] or r['answer']!=parent['answer'] or r['context']!=sessions[r['source_session_id']]:
            raise ValueError('Original question/answer/full session changed')
    reserved={sid for r in raw for sid in r['answer_session_ids']}
    seen_questions={r['question'] for r in originals};added=[]
    for sid in sorted(sessions,key=lambda s:hashlib.sha256(s.encode()).hexdigest()):
        if sid in reserved:continue
        context=sessions[sid]
        if sum(len(t['content']) for t in context)>32000:continue
        sentences=[s.strip() for t in context if t['role']=='user'
                   for s in re.split(r'(?<=[.!?])\s+|\n+',t['content'])]
        for sentence in sentences:
            words=sentence.split()
            if not 9<=len(words)<=18 or not re.match(r'^(I |My |We )',sentence) or '?' in sentence:
                continue
            prefix=' '.join(words[:6])
            if sum(s.startswith(prefix) for s in sentences)!=1:continue
            question='What was my complete statement beginning with '+json.dumps(prefix,ensure_ascii=False)+'?'
            if question in seen_questions:continue
            turn=next(i for i,t in enumerate(context) if t['role']=='user' and sentence in t['content'])
            suffix=hashlib.sha256(sid.encode()).hexdigest()[:16]
            added.append(dict(qid='extractive-'+suffix,sid='real-'+suffix,source_session_id=sid,
                question=question,answer=sentence,context=context,answer_session_ids=[sid],
                history='new_extractive_distractor',annotation=dict(kind='unique_user_quote_completion',
                user_turn=turn,quote_prefix=prefix,answer_is_verbatim_user_span=True,
                context_sha256=hashlib.sha256(json.dumps(context,ensure_ascii=False,sort_keys=True).encode()).hexdigest())))
            seen_questions.add(question);break
        if len(added)==448:break
    if len(added)!=448:raise ValueError('Insufficient verifiable real distractor sessions')
    items=originals+added
    if len({r['source_session_id'] for r in items})!=512:raise ValueError('Duplicate real sessions')
    design=dict(physical_gpu=1,source=str(source),data_root=str(data_root),sizes=list(SIZES),repeats=3,
        model=str(data_root/'models/Llama-3.1-8B-Instruct'),embedding_model=str(data_root/'models/all-MiniLM-L6-v2'),
        projection=read(source/'design.json')['projection'],layers=[13,14,15],dtype='float32',
        dataset_sha256=sha256(data_root/'data/longmemeval_s.json'),
        projection_sha256=read(source/'design.json')['projection_sha256'],
        source_hashes={n:sha256(source/n) for n in ('items.json','design.json','completion.json','evaluation.json')},
        panel_ids=[r['qid'] for r in panel],original_sessions=64,added_real_sessions=448,
        annotation='Deterministic extractive quote completion: unique six-word prefix identifies a verbatim user sentence. Full original session preserved. New write annotations, not official benchmark labels.',
        scope='Exploratory interference test on eight previously inspected original questions. Added question distribution differs from official single-session-user. No paraphrases, no claim of 512 official eligible questions or 512 independent test queries.',
        compact='Legacy64 rounded dense deltas approximated by rank-one factors. Validate all64 own-question emitted token sequences and panel top1 routing before adding memories. New one-key zero-accumulator writes use Sherman-Morrison; identical grounded50-step objective/L2=.1/EOS, different numerical solver.',
        validation_gate='Maximum relative Frobenius reconstruction error <=.002, identical all64 own-question token sequences, identical panel top1 at N8/16/32/64. Gate is empirical, not global losslessness.',
        router='Independent factor energy on base FFN last-token keys; no reference or question-key lookup used to route. Every factor evaluated on each query; lexical sid tie break.',
        latency='Single-question end-to-end timing, including base key forward, disk factor reads/scoring, selected-factor application/generation/base restoration. No shared query matrix. RAG includes embedding/search/tokenization/prefill/decode. KV prefix setup separately measured. First pass and two subsequent passes distinguished, OS cache not flushed.',
        memory='Actual compact files/dense64 files, text+vectors, retained KV tensor payload; tier-specific. Peak CUDA allocated/reserved per complete method including setup. Dense>64 is a counterfactual calculation, not stored. Infeasible KV has calculated required payload and no measured accuracy.',
        generation='Greedy64tokens, EOS/next-turn stopping, literal original questions, batch1, no labels at recall.',
        accuracy='Eight unique first-pass answers per N; repeated passes are timing/consistency samples only. Normalized whole-output exact, containment, tokenF1, source top1/rank. KV full-bank has no explicit source router.',
        expected_accuracy='100% aspirational target; historical K1-V1 containment100%/whole exact0%, different items/writer/16-token cap, reference only, not forecast.',
        bank_order='Eight real prior pilot items, remaining56 existing real items in original frozen order,448 unique additional real sessions in SHA256(source ID) order.')
    out.mkdir(parents=True)
    save(out,'items.json',items);save(out,'design.json',design)
    save(out,'historical_k1.json',read(data_root/'results/k1-question-keys-v2/summary.json')['V1'])
    save(out,'manifest_sha256.json',{n:sha256(out/n) for n in ('items.json','design.json','historical_k1.json')})
    save(out,'completion.json',dict(completed=False,stage='prepared'))
    print('PREPARED real sessions512, original fixed questions8',flush=True)


def records(items):
    return [dict(record_id='record-'+str(i),sid=r['sid'],stored_question=r['question'],
        stored_answer=r['answer'],conversation='\n'.join(t['role']+': '+t['content'] for t in r['context'])) for i,r in enumerate(items)]


def run(out):
    from experiment_utils import require_gpu1
    gpu=require_gpu1()
    busy=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
    if busy.stdout.strip():raise RuntimeError('GPU1 busy; no fallback')
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModel
    from AlphaEditSessionStore import SessionFFNMemory
    from compact_session_memory import CompactSessionMemory,fit_rank_one,capture_keys,factor_scores
    from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
    import AlphaEdit.AlphaEdit_main as writer
    from session_write import write_request
    if torch.cuda.device_count()!=1:raise RuntimeError('Require only physical GPU1')
    torch.set_num_threads(8);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    design=read(out/'design.json');items=read(out/'items.json');panel=items[:8];source=Path(design['source'])
    frozen=read(out/'manifest_sha256.json')
    if read(out/'completion.json').get('completed'):raise ValueError('Completed run immutable')
    for name,value in frozen.items():
        if sha256(out/name)!=value:raise ValueError('Protocol changed')
    for name,value in design['source_hashes'].items():
        if sha256(source/name)!=value:raise ValueError('Historical source changed')
    code_root=Path(__file__).resolve().parent
    code_hashes={n:sha256(code_root/n) for n in ('experiment_real512_compact.py','compact_session_memory.py',
        'fixed_query_scaling_core.py','experiment_utils.py','question_key.py','session_write.py','answer_only_generation.py','memory_comparison_core.py',
        'AlphaEdit/grounded_value.py','AlphaEdit/compute_z.py','AlphaEdit/compute_ks.py')}
    begin=time.monotonic();measurements={};results={};compact=None;legacy=None;model=None;encoder=None;cache=None
    def sync():torch.cuda.synchronize()
    def status(stage,**extra):
        save(out,'status.json',dict(stage=stage,elapsed_seconds=time.monotonic()-begin,**extra))
    def start(stage):
        gc.collect();torch.cuda.empty_cache();sync();torch.cuda.reset_peak_memory_stats();status(stage)
        return time.monotonic()
    def end(stage,clock,**extra):
        sync();r=dict(elapsed_seconds=time.monotonic()-clock,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),end_allocated_bytes=torch.cuda.memory_allocated(),**extra)
        measurements[stage]=r;save(out,'measurements.json',measurements);return r
    def timed(function):
        sync();t=time.monotonic();value=function();sync();return value,time.monotonic()-t
    save(out,'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    try:
        status('integrity_before')
        expected=read(source/'completion.json')['delta_sha256']
        before={r['sid']:sha256(source/'deltas'/r['sid']/'v1.pt') for r in items[:64]}
        if before!={sid:expected[sid] for sid in before}:raise ValueError('Original deltas changed')
        if sha256(Path(design['projection']))!=design['projection_sha256']:raise ValueError('Projection changed')
        save(out,'provenance.json',dict(physical_gpu=1,gpu_uuid=gpu,torch_version=torch.__version__,
            cuda_version=torch.version.cuda,code_sha256=code_hashes,original_delta_sha256=before))
        t=start('model_loading')
        tok=AutoTokenizer.from_pretrained(design['model'],local_files_only=True);tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(design['model'],torch_dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
        names=[f'model.layers.{l}.mlp.down_proj.weight' for l in design['layers']]
        compact=CompactSessionMemory(model,names,out/'factors')
        base_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        end('model_loading',t,base_parameter_bytes=base_bytes,
            cpu_base_backup_bytes=sum(v.numel()*v.element_size() for v in compact.base.values()))
        # Verify every new annotation again before any write.
        for r in items[64:]:
            a=r['annotation'];turn=r['context'][a['user_turn']]
            if turn['role']!='user' or r['answer'] not in turn['content']:raise ValueError('Unverified annotation')
            if len(tok.encode(' '+r['answer']+tok.eos_token,add_special_tokens=False))>64:raise ValueError('New target exceeds answer cap')
        t=start('compact_validation')
        legacy=SessionFFNMemory(model,names,source/'deltas')
        all_keys=capture_keys(model,tok,names,[r['question'] for r in items[:64]])
        dense_scores={r['qid']:{} for r in panel};compact_scores={r['qid']:{} for r in panel};audit=[]
        for index,r in enumerate(items[:64]):
            status('compact_validation',done=index,total=64,qid=r['qid'])
            dense=torch.load(source/'deltas'/r['sid']/'v1.pt',map_location='cpu',weights_only=True)
            payload={};errors={};numerator=torch.zeros(8,device=model.device,dtype=torch.float64);denominator=0.
            for name in names:
                matrix=dense[name].to(model.device)
                left,right,error=fit_rank_one(matrix)
                if error>.002:raise ValueError('Rank-one reconstruction error above frozen gate')
                payload[name]=(left,right);errors[name]=error
                numerator+=(matrix@all_keys[name][:8].T).double().square().sum(0)
                denominator+=matrix.double().square().sum().item();del matrix
            path=out/'factors'/(r['sid']+'.pt')
            if path.exists():
                loaded=compact.load(r['sid'])
                if any(not torch.equal(a,b) for n in names for a,b in zip(loaded[n],payload[n])):raise ValueError('Resumed factors differ')
            else:compact.save(r['sid'],payload)
            comp=factor_scores(payload,{n:k[:8] for n,k in all_keys.items()})
            for i,q in enumerate(panel):
                dense_scores[q['qid']][r['sid']]=numerator[i].item()/denominator
                compact_scores[q['qid']][r['sid']]=comp[i]
            old=legacy.run(r['sid'],1,lambda m:generate_answer_batch(m,tok,[r['question']]))[0]
            new=compact.run(r['sid'],lambda m:generate_answer_batch(m,tok,[r['question']]))[0]
            same=old['token_ids']==new['token_ids']
            audit.append(dict(qid=r['qid'],sid=r['sid'],relative_frobenius_errors=errors,
                tokens_identical=same,dense=old,compact=new,source_sha256=before[r['sid']]))
            save(out,'compact_validation.json',dict(own_question_pairs=audit,passed=False))
            print('VALIDATE',index+1,'token_identity',same,'max_error',max(errors.values()),flush=True)
            if not same:raise ValueError('Compact storage changes a legacy own-question token sequence')
            del dense
        routing_checks=[]
        for n in (8,16,32,64):
            allowed={r['sid'] for r in items[:n]}
            for q in panel:
                a=sorted(allowed,key=lambda sid:(-dense_scores[q['qid']][sid],sid))
                b=sorted(allowed,key=lambda sid:(-compact_scores[q['qid']][sid],sid))
                routing_checks.append(dict(n=n,qid=q['qid'],dense_top1=a[0],compact_top1=b[0],top1_identical=a[0]==b[0]))
        if not all(r['top1_identical'] for r in routing_checks):raise ValueError('Compact routing changes legacy panel top1')
        save(out,'compact_validation.json',dict(passed=True,own_question_pairs=audit,routing_checks=routing_checks,
            maximum_relative_frobenius_error=max(v for r in audit for v in r['relative_frobenius_errors'].values()),
            scope='Empirical equivalence on64original own questions and32panel routes; approximation, not lossless/global equivalence.'))
        all_keys=None;legacy=None;gc.collect();end('compact_validation',t)
        hp=AlphaEditHyperParams.from_json(str(code_root/'config/alphaedit-hparams.json'))
        hp.layers=design['layers'];hp.L2=.1;hp.v_num_grad_steps=50;writer.CONTEXT_TEMPLATES_CACHE=[['{}']]
        projector=torch.load(design['projection'],map_location='cpu',weights_only=True)
        t=start('additional_writes');writes=[]
        previous=read(out/'writes.json') if (out/'writes.json').exists() else []
        for index,r in enumerate(items[64:]):
            status('additional_writes',done=index,total=448,qid=r['qid'])
            path=out/'factors'/(r['sid']+'.pt')
            if path.exists():
                row=next((x for x in previous if x['sid']==r['sid']),None)
                if row is None or sha256(path)!=row['sha256']:raise ValueError('Unverified resumed write')
                compact.load(r['sid']);writes.append(row);continue
            request=write_request(r,index);request['target_new']['str']=r['answer']+tok.eos_token
            full=request['value_context_templates'][0][1].format(request['prompt']).format(request['subject'])+' '+request['target_new']['str']
            if len(tok.encode(full))+64>model.config.max_position_embeddings:raise ValueError('Write over model context limit')
            w=time.monotonic();compact.write(r['sid'],tok,request,hp,projector)
            final=compact.run(r['sid'],lambda m:generate_answer_batch(m,tok,[r['question']]))[0]
            row=evaluated(r,final,time.monotonic()-w,selected_sid=r['sid'])
            row.update(sid=r['sid'],sha256=sha256(path),bytes=path.stat().st_size)
            writes.append(row);save(out,'writes.json',writes)
            print('WRITE',index+1,'/448','exact',row['returned_exact'],'seconds',round(row['query_seconds'],2),flush=True)
        end('additional_writes',t,count=len(writes),actual_factor_payload_bytes=sum(r['bytes'] for r in writes))
        projector=None;compact.restore();gc.collect();torch.cuda.empty_cache()
        bank=records(items);save(out,'bank.json',bank)
        save(out,'written_bank_sha256.json',{r['sid']:sha256(out/'factors'/(r['sid']+'.pt')) for r in items})
        # Each query freshly captures base keys and reads/scores every factor.
        for n in SIZES:
            folder=out/('bank-'+str(n));folder.mkdir(exist_ok=True)
            t=start('our_delta_'+str(n));trials=[]
            for repeat in range(design['repeats']):
                order=list(panel);random.Random(100+repeat).shuffle(order)
                for q in order:
                    status('our_delta',n=n,repeat=repeat,qid=q['qid'])
                    def route():
                        keys=capture_keys(model,tok,names,[q['question']]);scores={}
                        for r in items[:n]:scores[r['sid']]=factor_scores(compact.load(r['sid']),keys)[0]
                        return sorted(scores,key=lambda sid:(-scores[sid],sid)),scores
                    (ranked,scores),route_seconds=timed(route)
                    final,answer_seconds=timed(lambda:compact.run(ranked[0],lambda m:generate_answer_batch(m,tok,[q['question']]))[0])
                    final['max_query_cache_payload_bytes']=(len(tok.encode(prompt(q['question'])))+final['tokens']-1)*kv_bytes_per_token(model.config)
                    final['cache_measurement']='Configuration-derived estimate for generate API transient cache'
                    trial=evaluated(q,final,route_seconds+answer_seconds,selected_sid=ranked[0])
                    trial.update(repeat=repeat,rank=ranked.index(q['sid'])+1,route_seconds=route_seconds,
                        answer_seconds=answer_seconds,activation_scores=scores)
                    trials.append(trial);save(folder,'our_delta.json',dict(trials=trials))
            resource=end('our_delta_'+str(n),t,persistent_payload_bytes=sum((out/'factors'/(r['sid']+'.pt')).stat().st_size for r in items[:n]),
                persistent_tier='Disk compact factors; streamed per query',additional_model_bytes=0)
            first=[r for r in trials if r['repeat']==0];summary=metrics(first)
            summary.update(mean_rank=sum(r['rank'] for r in first)/8,mrr=sum(1/r['rank'] for r in first)/8)
            results[str(n)]={'our_delta':dict(available=True,trials=trials,summary=summary,latency=latency_summary(trials),resources=resource)}
            save(out,'evaluation_progress.json',dict(results=results));print('OUR',n,'route',summary['retrieval_at1']['count'],'exact',summary['returned_exact']['count'],flush=True)
        compact.restore()
        t=start('rag_index_setup')
        etok=AutoTokenizer.from_pretrained(design['embedding_model'],local_files_only=True)
        encoder=AutoModel.from_pretrained(design['embedding_model'],local_files_only=True).to('cuda:0').float().eval()
        embbytes=sum(p.numel()*p.element_size() for p in encoder.parameters())
        vectors=embed(encoder,etok,[r['stored_question'] for r in bank]);torch.save(vectors,out/'qa_embeddings.pt')
        end('rag_index_setup',t,embedding_parameter_bytes=embbytes,vector_payload_bytes=vectors.numel()*vectors.element_size())
        for n in SIZES:
            folder=out/('bank-'+str(n));save(folder,'bank.json',bank[:n]);torch.save(vectors[:n].clone(),folder/'qa_embeddings.pt')
            t=start('rag_qa_'+str(n));trials=[]
            for repeat in range(design['repeats']):
                order=list(panel);random.Random(100+repeat).shuffle(order)
                for q in order:
                    status('rag_qa',n=n,repeat=repeat,qid=q['qid'])
                    query,embedding_seconds=timed(lambda:embed(encoder,etok,[q['question']])[0])
                    ranked,search_seconds=timed(lambda:rank_vectors(query,vectors[:n],1))
                    selected=bank[ranked[0][0]]
                    def tokens():
                        return (tok.encode(bank_prefix(tok,[document(selected)]),add_special_tokens=False),
                            tok.encode(query_suffix(tok,q['question']),add_special_tokens=False))
                    (pids,suffix),tokenization_seconds=timed(tokens)
                    if len(pids)+len(suffix)+64>model.config.max_position_embeddings:raise ValueError('RAG context limit')
                    cache,prefill_seconds=timed(lambda:prefill(model,pids,chunk_size=256))
                    (final,cache),decode_seconds=timed(lambda:decode_cached(model,tok,cache,suffix))
                    seconds=embedding_seconds+search_seconds+tokenization_seconds+prefill_seconds+decode_seconds
                    trial=evaluated(q,final,seconds,sources=[selected['sid']])
                    trial.update(repeat=repeat,embedding_seconds=embedding_seconds,search_seconds=search_seconds,
                        tokenization_seconds=tokenization_seconds,prefill_seconds=prefill_seconds,decode_seconds=decode_seconds,
                        context_tokens=len(pids),retrieval_scores=[ranked[0][1]])
                    trials.append(trial);cache=None;save(folder,'rag_qa.json',dict(trials=trials))
            resource=end('rag_qa_'+str(n),t,persistent_payload_bytes=sum((folder/f).stat().st_size for f in ('bank.json','qa_embeddings.pt')),
                persistent_tier='CPU/disk text and vector index',additional_model_bytes=embbytes)
            results[str(n)]['rag_qa']=dict(available=True,trials=trials,summary=metrics([r for r in trials if r['repeat']==0]),
                latency=latency_summary(trials),resources=resource)
            save(out,'evaluation_progress.json',dict(results=results));print('RAG',n,'exact',results[str(n)]['rag_qa']['summary']['returned_exact']['count'],flush=True)
        encoder=None;etok=None;vectors=None;gc.collect();torch.cuda.empty_cache()
        for n in SIZES:
            folder=out/('bank-'+str(n))
            pids=tok.encode(bank_prefix(tok,[document(r) for r in bank[:n]]),add_special_tokens=False)
            suffixes={q['qid']:tok.encode(query_suffix(tok,q['question']),add_special_tokens=False) for q in panel}
            required=len(pids)+max(map(len,suffixes.values()))+64
            plan=dict(prefix_tokens=len(pids),required_tokens=required,model_context_limit=model.config.max_position_embeddings,
                bytes_per_token=kv_bytes_per_token(model.config),calculated_prefix_payload_bytes=len(pids)*kv_bytes_per_token(model.config))
            save(folder,'kv_capacity.json',plan)
            if required>model.config.max_position_embeddings:
                results[str(n)]['kv_full_qa']=dict(available=False,reason='context_limit',plan=plan,trials=[],summary=None)
                save(out,'evaluation_progress.json',dict(results=results));print('KV',n,'context_limit',required,flush=True);continue
            t=start('kv_setup_'+str(n))
            try:
                cache=prefill(model,pids,chunk_size=256,progress=lambda done,total:status('kv_setup',n=n,done=done,total=total))
            except torch.OutOfMemoryError as exc:
                resource=end('kv_failed_setup_'+str(n),t)
                results[str(n)]['kv_full_qa']=dict(available=False,reason='gpu_limit',error=str(exc),plan=plan,trials=[],summary=None,resources=resource)
                cache=None;gc.collect();torch.cuda.empty_cache();save(out,'evaluation_progress.json',dict(results=results));continue
            actual=cache_payload_bytes(cache)
            if actual!=plan['calculated_prefix_payload_bytes']:raise ValueError('KV measured payload differs from formula')
            setup=end('kv_setup_'+str(n),t,persistent_payload_bytes=actual)
            t=start('kv_full_qa_'+str(n));trials=[]
            for repeat in range(design['repeats']):
                order=list(panel);random.Random(100+repeat).shuffle(order)
                for q in order:
                    status('kv_full_qa',n=n,repeat=repeat,qid=q['qid'])
                    (final,cache),seconds=timed(lambda:decode_cached(model,tok,cache,suffixes[q['qid']]))
                    trial=evaluated(q,final,seconds);trial.update(repeat=repeat,context_tokens=len(pids))
                    trials.append(trial);save(folder,'kv_full_qa.json',dict(trials=trials))
            resource=end('kv_full_qa_'+str(n),t,persistent_payload_bytes=actual,persistent_tier='GPU full-bank prefix cache',
                setup_seconds=setup['elapsed_seconds'],setup_peak_allocated_bytes=setup['peak_allocated_bytes'],
                setup_peak_reserved_bytes=setup['peak_reserved_bytes'],additional_model_bytes=0)
            results[str(n)]['kv_full_qa']=dict(available=True,plan=plan,trials=trials,summary=metrics([r for r in trials if r['repeat']==0]),
                latency=latency_summary(trials),resources=resource)
            cache=None;gc.collect();torch.cuda.empty_cache();save(out,'evaluation_progress.json',dict(results=results))
            print('KV',n,'exact',results[str(n)]['kv_full_qa']['summary']['returned_exact']['count'],flush=True)
        check_complete(results,design['panel_ids'],design['repeats'])
        status('integrity_after');compact.restore()
        if any(not torch.equal(model.get_parameter(n).detach().cpu(),v) for n,v in compact.base.items()):raise ValueError('Base weights not restored')
        after={r['sid']:sha256(source/'deltas'/r['sid']/'v1.pt') for r in items[:64]}
        if after!=before:raise ValueError('Historical deltas changed')
        factor_hashes={r['sid']:sha256(out/'factors'/(r['sid']+'.pt')) for r in items}
        if factor_hashes!=read(out/'written_bank_sha256.json'):raise ValueError('Written factor bank changed')
        for name,value in code_hashes.items():
            if sha256(code_root/name)!=value:raise ValueError('Code changed during run')
        for name,value in frozen.items():
            if sha256(out/name)!=value:raise ValueError('Protocol changed')
        for name,value in design['source_hashes'].items():
            if sha256(source/name)!=value:raise ValueError('Historical source changed')
        if sha256(Path(design['data_root'])/'data/longmemeval_s.json')!=design['dataset_sha256']:raise ValueError('Dataset changed')
        dense_sizes=[(source/'deltas'/r['sid']/'v1.pt').stat().st_size for r in items[:64]]
        save(out,'evaluation.json',dict(results=results,sizes=list(SIZES),panel_ids=design['panel_ids'],scope=design['scope']))
        save(out,'resource_summary.json',dict(elapsed_seconds=time.monotonic()-begin,base_model_parameter_bytes=base_bytes,
            embedding_parameter_bytes=embbytes,measurements=measurements,
            compact512_actual_bytes=sum((out/'factors'/(r['sid']+'.pt')).stat().st_size for r in items),
            dense64_actual_bytes=sum(dense_sizes),dense_counterfactual_bytes_per_session=sum(dense_sizes)/64,
            cpu_base_backup_bytes=sum(v.numel()*v.element_size() for v in compact.base.values())))
        save(out,'completion.json',dict(completed=True,physical_gpu=1,gpu_uuid=gpu,real_sessions=512,
            original_annotated_sessions=64,new_extractive_annotations=448,fixed_original_queries=8,sizes=list(SIZES),
            compact_validation_passed=True,base_weights_restored=True,historical_deltas_unchanged=True,
            written_factors_unchanged=True,protocol_unchanged=True,code_unchanged=True,
            factor_sha256=factor_hashes,kv_status={str(n):'completed' if results[str(n)]['kv_full_qa']['available'] else results[str(n)]['kv_full_qa']['reason'] for n in SIZES}))
        status('completed');print('COMPLETED REAL512 COMPACT',flush=True)
    except BaseException as exc:
        save(out,'completion.json',dict(completed=False,stage='failed',error=repr(exc),physical_gpu=1))
        status('failed',error=repr(exc));raise
    finally:
        if compact is not None:compact.restore()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();out=args.out.resolve()
    if not out.exists():prepare(args.data_root.resolve(),out)
    if not args.prepare_only:run(out)


if __name__=='__main__':main()
