"""GPU-1 original-question recall repair: turn stopping and answer-plus-EOS writes.

No BERT, probability calibration, new questions, answer lookup at inference, or
changes to the historical experiment. Writer selection uses development only.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_larger_routing import read, save, sha256
from question_key_k1_core import normalize, interval, prompt
from session_qa_routing_core import write_request, candidate_text
from answer_only_generation import generate_answer_batch, reference_tokens_present, answer_boundary


def prepare(root, out, source):
    if out.exists():
        raise ValueError('Use a new output directory; historical runs are immutable')
    complete = read(source/'completion.json')
    if not complete.get('completed') or not complete.get('base_weights_restored'):
        raise ValueError('Require a successful baseline')
    prior = read(source/'items.json')
    # First eight training records exercise number/ambiguous/long-answer failures.
    # Fixed before any candidate write; no test examples select this subset.
    items = dict(train=prior['train'][:8], validation=prior['validation'], test=prior['test'])
    allrows = [r for rows in items.values() for r in rows]
    if len({r['source_session_id'] for r in allrows}) != len(allrows):
        raise ValueError('Overlapping source splits')
    official = {r['question_id']: r for r in read(root/'data/longmemeval_s.json')}
    for r in allrows:
        record = official[r['qid']]
        session = dict(zip(record['haystack_session_ids'], record['haystack_sessions']))[r['source_session_id']]
        if r['question'] != record['question'] or r['answer'] != record['answer'] or r['context'] != session:
            raise ValueError('Must use the literal official question/answer/full source session')
        if 'paraphrase' in r:
            raise ValueError('Original questions only')
    original_design = read(Path(read(source/'design.json')['source'])/'design.json')
    projection = Path(original_design['projection'])
    design = dict(source=str(source), physical_gpu=1, layers=[13,14,15], seed=0,
                  model=str(root/'models/Llama-3.1-8B-Instruct'), dtype='float32',
                  original_delta_root=read(source/'design.json')['delta_root'],
                  repaired_delta_root=str(out/'deltas'), projection=str(projection),
                  projection_sha256=sha256(projection), projection_metadata=read(projection.with_suffix('.json')),
                  source_hashes={name:sha256(source/name) for name in
                                 ['completion.json','design.json','items.json','banks.json','evaluation.json','fresh_answers.json']},
                  question_protocol='Literal original dataset question, identical Q: {question}\\nA: write and recall key',
                  full_context_at_write=True, reference_at_recall=False,
                  router='Normalized delta activation energy; lexicographic sid tie break; no text scorer or calibration',
                  activation_formula='sum_l ||delta_W_s,l @ k_l(q)||^2 / sum_l ||delta_W_s,l||_F^2',
                  candidate_writer=dict(name='answer_plus_eos', target='Supplied answer followed by tokenizer.eos_token',
                                        v_num_grad_steps=50, L2=.1, independent_base_writes=True,
                                        grounding='Full answer-bearing session, unchanged value templates',
                                        change='Add end-of-answer token to existing complete-answer loss'),
                  stopping='Per-row stop at generated newline Q:/A:/Question:/User:/Assistant:; return text before delimiter; preserve other text',
                  max_new_tokens=64, batch_size=8, development_train=8, validation=8, test=24,
                  unused='Eight other training and eight calibration records; no new calibration',
                  selection='Validation: candidate source routing and token-bounded containment must not decrease; then strictly improve returned exact, emitted exact, or mean tokens (in that order)',
                  test_scope='Exploratory paired repair on previously inspected24test questions; no unseen-question claim',
                  train_scope='Eight prior train examples only for write diagnostics; eight disjoint validation examples select the fixed candidate writer',
                  metrics='Normalized exact, token-bounded containment, emitted exact, online termination; no automated semantic truth label',
                  expected_accuracy='100% target plus historical K1; targets are not forecasts',
                  expected_memory='Dense float32 three-layer deltas:704644657bytes each; max40newdelta files; unchanged base LLM',
                  source_probes_reused=True, fresh_activation=True, fresh_selected_finals=True,
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    out.mkdir(parents=True)
    save(out,'items.json',items);save(out,'design.json',design)
    save(out,'manifest_sha256.json',{name:sha256(out/name) for name in ['items.json','design.json']})
    save(out,'completion.json',dict(completed=False,stage='prepared'))
    # Retain a transparent audit of the previously inspected six mismatches.
    notes = {
        'b86304ba': 'Correct reference sentence followed by an invented $120 sale/math exercise; user session gives only triple value.',
        'd52b4f67': 'Correct venue with New York City added; user evidence does not establish that city.',
        'a82c026e': 'Correct game with specific DLC name added; user evidence gives Dark Souls3DLC only.',
        '86f00804': 'Correct book title with author attribution added; extra attribution is not needed to answer the question.',
        '60d45044': 'Correct rice type followed by uncertainty and simulated dialogue.',
        '726462e0': 'Correct10%discount expressed in a longer sentence; reference value matches.'}
    baseline = next(r for r in read(source/'evaluation.json')['results'] if r['method']=='activation')
    audit=[]
    for t in baseline['trials']:
        if t['final_exact']: continue
        row=next(r for r in items['test'] if r['qid']==t['qid'])
        audit.append(dict(qid=t['qid'],question=t['question'],reference=t['answer'],
                          old_returned=t['final_answer'],old_full=t['final']['text'],
                          assessment=notes[t['qid']], reference_value_at_start=normalize(t['final_answer']).startswith(normalize(t['answer'])),
                          source_user_turns=[x['content'] for x in row['context'] if x['role']=='user']))
    save(out,'baseline_mismatch_audit.json',dict(scope='Manual descriptive audit; not a gold semantic grading model',cases=audit))


def activation_scores(model, tok, store, rows, delta_root):
    import torch
    store.deactivate();names=store.weight_names;current={};handles=[];keys={}
    for name in names:
        def hook(mod,args,output,name=name):current[name]=args[0][:,-1].detach().clone()
        handles.append(model.get_submodule(name.removesuffix('.weight')).register_forward_hook(hook))
    try:
        tok.padding_side='left'
        for start in range(0,len(rows),8):
            chunk=rows[start:start+8]
            encoded=tok([prompt(r['question']) for r in chunk],return_tensors='pt',padding=True,truncation=False).to(model.device)
            positions=encoded.attention_mask.cumsum(-1)-1;positions.masked_fill_(encoded.attention_mask==0,0)
            with torch.inference_mode():model.model(**encoded,position_ids=positions,use_cache=False)
            for i,r in enumerate(chunk):keys[r['qid']]={name:current[name][i].clone() for name in names}
    finally:
        for h in handles:h.remove()
    values={r['qid']:{} for r in rows}
    for session in rows:
        delta=torch.load(delta_root/session['sid']/'v1.pt',map_location='cpu',weights_only=True)
        numerator=torch.zeros(len(rows),device=model.device,dtype=torch.float64)
        denominator=torch.zeros((),device=model.device,dtype=torch.float64)
        with torch.inference_mode():
            for name in names:
                d=delta[name].to(model.device);key=torch.stack([keys[r['qid']][name] for r in rows],dim=1)
                numerator+=(d@key).double().square().sum(0);denominator+=d.double().square().sum()
                del d,key
        if denominator.item()<=0:raise ValueError('Empty delta')
        for i,r in enumerate(rows):values[r['qid']][session['sid']]=(numerator[i]/denominator).item()
        del delta
    return values


def routes(rows, values):
    result=[]
    for r in rows:
        ranked=sorted(values[r['qid']],key=lambda sid:(-values[r['qid']][sid],sid))
        result.append(dict(qid=r['qid'],question=r['question'],selected_sid=ranked[0],
                           selected_score=values[r['qid']][ranked[0]],own_sid=r['sid'],
                           rank=ranked.index(r['sid'])+1,routing_correct=ranked[0]==r['sid'],
                           activation_scores=values[r['qid']]))
    return result


def summaries(trials):
    n=len(trials)
    result=dict(n=n)
    for name in ['routing_correct','returned_exact','emitted_exact','extracted_exact','contains_reference','emitted_contains_reference','hit_cap','eos','next_turn']:
        count=sum(bool(r[name]) for r in trials)
        result[name]=dict(count=count,total=n,rate=count/n,ci95=list(interval(count,n)))
    result['mean_generated_tokens']=sum(r['final']['tokens'] for r in trials)/n
    result['mean_rank']=sum(r['rank'] for r in trials)/n
    result['mrr']=sum(1/r['rank'] for r in trials)/n
    return result


def finish_trials(rows, routed, answers):
    lookup={r['qid']:r for r in rows};trials=[]
    for route in routed:
        r=lookup[route['qid']];final=answers[r['qid']]
        returned=final['text'];emitted=final['emitted_text'];extracted=candidate_text(returned)
        trials.append(dict(route,answer=r['answer'],source_session_id=r['source_session_id'],
                           answer_session_ids=r['answer_session_ids'],final=final,
                           returned_exact=normalize(returned)==normalize(r['answer']),
                           emitted_exact=normalize(emitted)==normalize(r['answer']),
                           extracted_exact=normalize(extracted)==normalize(r['answer']),
                           contains_reference=reference_tokens_present(returned,r['answer']),
                           emitted_contains_reference=reference_tokens_present(emitted,r['answer']),
                           hit_cap=final['hit_cap'],eos=final['finish_reason']=='eos',
                           next_turn=final['finish_reason']=='next_turn'))
    return dict(trials=trials,summary=summaries(trials))


def generate_selected(model, tok, store, rows, routed, stop, status, label):
    answers={};actual=0;route_by_id={r['qid']:r for r in routed}
    for start in range(0,len(rows),8):
        chunk=rows[start:start+8]
        for sid in sorted({route_by_id[r['qid']]['selected_sid'] for r in chunk}):
            status(label,done=len(answers),total=len(rows),sid=sid)
            # Only question strings reach the generator. Companions keep original batching.
            result=store.run(sid,1,lambda m:generate_answer_batch(m,tok,[r['question'] for r in chunk],stop_at_turn=stop))
            actual+=len(result)
            for r,final in zip(chunk,result):
                if route_by_id[r['qid']]['selected_sid']==sid:
                    answers[r['qid']]=dict(final,batch_query_ids=[x['qid'] for x in chunk])
    return answers,actual


def run(root,out):
    design=read(out/'design.json');items=read(out/'items.json');source=Path(design['source'])
    frozen=read(out/'manifest_sha256.json')
    if frozen != {name:sha256(out/name) for name in ['items.json','design.json']}:
        raise ValueError('Frozen design changed')
    if read(out/'completion.json').get('completed'):raise ValueError('Completed runs are immutable')
    for name,value in design['source_hashes'].items():
        if sha256(source/name)!=value:raise ValueError('Baseline artifact changed')
    from zsre_routing_experiment import require_gpu1
    gpu=require_gpu1()
    busy=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
    if busy.stdout.strip():raise RuntimeError('PhysicalGPU1 busy')
    import torch
    from routing_ladder_models import AlphaEditAdapter
    from AlphaEditSessionStore import SessionFFNMemory
    if torch.cuda.device_count()!=1:raise RuntimeError('Only physicalGPU1 may be visible')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    if shutil.disk_usage(out).free < 40*704644657+1024**3:raise RuntimeError('Insufficient space for repaired bank')
    original_root=Path(design['original_delta_root']);delta_root=out/'deltas'
    baseline_hashes=read(source/'completion.json')['delta_sha256']
    used=[r for rows in items.values() for r in rows]
    before={r['sid']:sha256(original_root/r['sid']/'v1.pt') for r in used}
    if any(value!=baseline_hashes[sid] for sid,value in before.items()):raise ValueError('Original delta changed')
    code=['experiment_answer_termination_exact.py','answer_only_generation.py','routing_ladder_models.py',
          'AlphaEditSessionStore.py','AlphaEdit/grounded_value.py','AlphaEdit/AlphaEdit_main.py','session_qa_routing_core.py']
    code_hashes={name:sha256(root/name) for name in code}
    save(out,'execution_provenance.json',dict(physical_gpu=1,gpu_uuid=gpu,code_sha256=code_hashes,
         torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0),
         tf32_matmul=torch.backends.cuda.matmul.allow_tf32,dtype='float32'))
    begin=time.monotonic();adapter=None;oldstore=None;sampler=None;generations=0;write_rows=[];phases={}
    save(out,'completion.json',dict(completed=False,stage='running',gpu_uuid=gpu))
    def status(stage,**extra):
        save(out,'status.json',dict(stage=stage,gpu_uuid=gpu,elapsed_seconds=time.monotonic()-begin,**extra))
    try:
        sampler=subprocess.Popen([sys.executable,str(root/'sample_session_qa_resources.py'),'--pid',str(os.getpid()),
                                  '--out',str(out),'--delta-root',str(delta_root)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        status('model_loading')
        adapter=AlphaEditAdapter(root,delta_root,layers=design['layers'],projection_path=Path(design['projection']))
        oldstore=SessionFFNMemory(adapter.model,adapter.store.weight_names,original_root)
        eos_token=adapter.tok.eos_token; eos_id=adapter.tok.convert_tokens_to_ids(eos_token)
        if eos_id not in adapter.model.generation_config.eos_token_id:raise ValueError('Write target must be a generation stop token')
        save(out,'writer_config.json',dict(hparams=vars(adapter.hp),eos_token=eos_token,eos_token_id=eos_id))
        # Validate every grounded prefix in advance, with the new EOS target.
        for i,r in enumerate(used):
            req=write_request(r,i);req['target_new']['str']=r['answer']+eos_token
            target_ids=adapter.tok.encode(' '+req['target_new']['str'],add_special_tokens=False)
            if target_ids[-1]!=eos_id:raise ValueError('EOS token is not included in loss target')
            full=req['value_context_templates'][0][1].format(req['prompt']).format(req['subject'])+' '+req['target_new']['str']
            if len(adapter.tok.encode(full))+64>adapter.model.config.max_position_embeddings:raise ValueError('Full session exceeds capacity')
        def write(rows,split):
            nonlocal generations
            for r in rows:
                status('answer_plus_eos_writes',split=split,done=len(write_rows),qid=r['qid'])
                started=time.monotonic();req=write_request(r,len(write_rows));req['target_new']['str']=r['answer']+eos_token
                adapter.store.write_session(r['sid'],adapter.tok,[req],adapter.hp,
                                            torch.zeros_like(adapter.P),adapter.P,1)
                # Diagnostic target probabilities only after writing; references do not enter generation.
                final=adapter.store.run(r['sid'],1,lambda m:generate_answer_batch(m,adapter.tok,[r['question']],stop_at_turn=True))[0]
                generations+=1
                file=delta_root/r['sid']/'v1.pt'
                entry=dict(qid=r['qid'],sid=r['sid'],split=split,write_success=True,
                           sha256=sha256(file),bytes=file.stat().st_size,elapsed_seconds=time.monotonic()-started,
                           target_includes_eos=True,immediate=final,
                           immediate_exact=normalize(final['text'])==normalize(r['answer']),
                           immediate_contains=reference_tokens_present(final['text'],r['answer']))
                write_rows.append(entry);save(out,'writes.json',write_rows)
                print('WRITE',len(write_rows),split,r['qid'],'exact',entry['immediate_exact'],'finish',final['finish_reason'],flush=True)
        write(items['train'],'train');write(items['validation'],'validation')
        phases['development_write']=torch.cuda.max_memory_allocated()/1024**3
        # Score all eight validation deltas using bare questions and no answer labels.
        status('validation_activation')
        oldval=activation_scores(adapter.model,adapter.tok,oldstore,items['validation'],original_root)
        newval=activation_scores(adapter.model,adapter.tok,adapter.store,items['validation'],delta_root)
        val_routes={'original':routes(items['validation'],oldval),'answer_plus_eos':routes(items['validation'],newval)}
        validations={}
        for name,store in [('original',oldstore),('answer_plus_eos',adapter.store)]:
            result,count=generate_selected(adapter.model,adapter.tok,store,items['validation'],val_routes[name],True,status,'validation_'+name)
            generations+=count;validations[name]=finish_trials(items['validation'],val_routes[name],result)
        save(out,'validation.json',validations)
        base=validations['original']['summary'];new=validations['answer_plus_eos']['summary']
        safe=(new['routing_correct']['count']>=base['routing_correct']['count'] and
              new['contains_reference']['count']>=base['contains_reference']['count'])
        base_tuple=(base['returned_exact']['count'],base['emitted_exact']['count'],-base['mean_generated_tokens'])
        new_tuple=(new['returned_exact']['count'],new['emitted_exact']['count'],-new['mean_generated_tokens'])
        selected='answer_plus_eos' if safe and new_tuple>base_tuple else 'original'
        selection=dict(selected_writer=selected,validation_safety_pass=safe,
                       original=base,new=new,criterion=design['selection'],test_seen_for_selection=False,
                       policy='Turn-boundary stopping remains enabled for the selected writer')
        save(out,'selection.json',selection)
        locked=dict(selection_sha256=sha256(out/'selection.json'),validation_sha256=sha256(out/'validation.json'),
                    writer_config_sha256=sha256(out/'writer_config.json'),code_sha256=code_hashes,
                    development_delta_sha256={r['sid']:r['sha256'] for r in write_rows})
        save(out,'frozen_before_test.json',locked)
        print('VALIDATION_SELECTION',selected,json.dumps(selection),flush=True)
        if selected=='answer_plus_eos':write(items['test'],'test')
        phases['all_writing']=torch.cuda.max_memory_allocated()/1024**3
        adapter.store.deactivate();adapter.P=None;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        status('test_activation')
        old_scores=activation_scores(adapter.model,adapter.tok,oldstore,items['test'],original_root)
        original_routes=routes(items['test'],old_scores)
        new_routes=original_routes
        if selected=='answer_plus_eos':
            new_scores=activation_scores(adapter.model,adapter.tok,adapter.store,items['test'],delta_root)
            new_routes=routes(items['test'],new_scores)
        save(out,'routing.json',dict(original=original_routes,selected=new_routes))
        # Fresh baseline controls under identical batches, with and without online stopping.
        cells={}
        for stop,label in [(False,'original_unbounded'),(True,'original_stopped')]:
            result,count=generate_selected(adapter.model,adapter.tok,oldstore,items['test'],original_routes,stop,status,label)
            generations+=count;cells[label]=finish_trials(items['test'],original_routes,result)
            save(out,'evaluation_progress.json',cells)
        old_eval=next(r for r in read(source/'evaluation.json')['results'] if r['method']=='activation')
        old_by_q={r['qid']:r for r in old_eval['trials']}
        for trial in cells['original_unbounded']['trials']:
            old=old_by_q[trial['qid']]
            if trial['selected_sid']!=old['selected_sid'] or trial['final']['text']!=old['final']['text']:
                raise ValueError('Fresh baseline failed to reproduce the original exact-question run')
        # Probe every selected-bank delta independently; restore base weights for every call.
        store=adapter.store if selected=='answer_plus_eos' else oldstore
        status('fresh_full_bank_probes');probes=[]
        for session in items['test']:
            for start in range(0,24,8):
                chunk=items['test'][start:start+8]
                result=store.run(session['sid'],1,lambda m:generate_answer_batch(m,adapter.tok,[r['question'] for r in chunk],stop_at_turn=True))
                generations+=len(result)
                for r,final in zip(chunk,result):probes.append(dict(qid=r['qid'],sid=session['sid'],source_session_id=session['source_session_id'],probe=final))
            save(out,'test_probes.json',probes)
            status('fresh_full_bank_probes',done=len(probes),total=576)
            print('PROBES',len(probes),576,flush=True)
        if len(probes)!=576 or len({(r['qid'],r['sid']) for r in probes})!=576:
            raise ValueError('Incomplete independent probe bank')
        result,count=generate_selected(adapter.model,adapter.tok,store,items['test'],new_routes,True,status,'selected_final_generation')
        generations+=count;cells['selected_repair']=finish_trials(items['test'],new_routes,result)
        by_pair={(r['qid'],r['sid']):r['probe'] for r in probes}
        for trial in cells['selected_repair']['trials']:
            if trial['final']['text']!=by_pair[(trial['qid'],trial['selected_sid'])]['text']:
                raise ValueError('Fresh selected final failed to reproduce its independent probe')
        evaluation=dict(selected_writer=selected,bank_size=24,question_protocol=design['question_protocol'],
                        test_is_fresh=False,scope=design['test_scope'],cells=cells,
                        metrics_note='Returned exact excludes only intercepted turn scaffolding. Emitted exact retains every generated delimiter. Containment uses token boundaries, not substring matching.')
        save(out,'evaluation.json',evaluation)
        phases['recall']=torch.cuda.max_memory_allocated()/1024**3
        # Immutable source, code, projection, selection, and every saved delta verification.
        status('integrity_checks')
        if sha256(Path(design['projection']))!=design['projection_sha256']:raise ValueError('Null-space projection changed')
        if locked['selection_sha256']!=sha256(out/'selection.json') or locked['validation_sha256']!=sha256(out/'validation.json'):
            raise ValueError('Development selection changed after test')
        for name,value in code_hashes.items():
            if sha256(root/name)!=value:raise ValueError('Experiment code changed midrun')
        for name,value in design['source_hashes'].items():
            if sha256(source/name)!=value:raise ValueError('Historical artifact changed')
        after={sid:sha256(original_root/sid/'v1.pt') for sid in before}
        if after!=before:raise ValueError('Historical delta changed')
        repaired={r['sid']:sha256(delta_root/r['sid']/'v1.pt') for r in write_rows}
        if repaired!={r['sid']:r['sha256'] for r in write_rows}:raise ValueError('Repaired delta changed')
        adapter.store.deactivate()
        for name,base_weight in adapter.store._base.items():
            if not torch.equal(adapter.model.get_parameter(name).detach().cpu(),base_weight):
                raise ValueError('Base weights not restored')
        resources=[]
        if (out/'resource_usage.jsonl').exists():
            resources=[json.loads(line) for line in (out/'resource_usage.jsonl').read_text().splitlines()]
        resource=dict(elapsed_seconds=time.monotonic()-begin,phase_torch_peak_gib=phases,
                      peak_gpu_used_gib=max([r.get('gpu_used_mib',0) for r in resources],default=0)/1024,
                      peak_host_rss_gib=max([r.get('host_rss_mib',0) for r in resources],default=0)/1024,
                      peak_host_hwm_gib=max([r.get('host_peak_rss_mib',0) for r in resources],default=0)/1024,
                      base_weight_gib=sum(p.numel()*p.element_size() for p in adapter.model.parameters())/1024**3,
                      new_delta_bytes=sum(r['bytes'] for r in write_rows),new_delta_count=len(write_rows),
                      selected_test_delta_bytes=sum((store.root/r['sid']/'v1.pt').stat().st_size for r in items['test']),
                      old_delta_bytes=sum(p.stat().st_size for p in original_root.rglob('*.pt')),
                      actual_question_generations=generations,scope='Full development writes, validation selection, optional test writes, full-bank probes, fresh controls and finals; no BERT/calibration')
        save(out,'resource_summary.json',resource)
        save(out,'completion.json',dict(completed=True,physical_gpu=1,gpu_uuid=gpu,selected_writer=selected,
             original_baseline_reproduced=True,source_artifacts_unchanged=True,original_delta_hashes_unchanged=True,
             new_delta_hashes_unchanged=True,base_weights_restored=True,selection_frozen_before_test=True,
             fresh_full_bank_probes=576,fresh_selected_finals=24,fresh_final_probe_match=True,
             new_writes=len(write_rows),actual_question_generations=generations,original_delta_sha256=after,
             new_delta_sha256=repaired,protocol_frozen_sha256=frozen,test_is_fresh=False))
        status('completed')
        print('COMPLETED',json.dumps({label:cell['summary'] for label,cell in cells.items()}),flush=True)
    except BaseException as exc:
        save(out,'completion.json',dict(completed=False,stage='failed',error=repr(exc),physical_gpu=1))
        status('failed',error=repr(exc));raise
    finally:
        if adapter is not None:adapter.store.deactivate()
        if sampler is not None:
            sampler.terminate()
            try:sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:sampler.kill()
        oldstore=None;adapter=None;gc.collect()
        if 'torch' in sys.modules:sys.modules['torch'].cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/answer-termination-exact-v1'))
    parser.add_argument('--source',type=Path,default=Path('results/evidence-routing-exact-v1'))
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    out=args.out.resolve();source=args.source.resolve()
    if not (out/'design.json').exists():prepare(root,out,source)
    if args.prepare_only:print('PREPARED',out);return
    run(root,out)


if __name__=='__main__':main()
