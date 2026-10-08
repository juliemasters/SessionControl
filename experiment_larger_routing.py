"""Disjoint 24/8/24 connected AlphaEdit write/probe/BERT routing experiment.

Preparation is standard-library only. Physical GPU 1 is masked before importing
torch. Test answers are used at the authorized memory write and at evaluation;
they are never supplied to the routing scorers or BERT training/calibration.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from question_key_k1_core import prompt, fired, normalize, interval
from session_qa_routing_core import write_request, candidate_text
from short_answer_judge_data import eligible_records, build_examples, question_key


# Authored from questions only, before any new delta/probe/scorer output.
PARAPHRASES = {
    'c19f7a0b': 'On a typical weekday evening, at what time do I arrive home after work?',
    'c960da58': 'What is the total number of playlists in my Spotify account?',
    '3b6f954b': 'Which place did I go to for my study abroad program?',
    '94f70d80': 'How much time did I need to put together my IKEA bookshelf?',
    'dccbc061': 'What view did I used to hold about spirituality?',
    '311778f1': 'During last month, how many hours did I watch Netflix documentaries?',
    'b86304ba': 'How much is my sunset painting worth relative to its purchase price?',
    '5d3d2817': 'What job did I have before my current occupation?',
    '37d43f65': 'What was the total RAM capacity of my laptop after the upgrade?',
    'd52b4f67': 'At what location did I go to my cousin\'s wedding?',
    '51a45a95': 'Where did I use the five-dollar coffee creamer coupon?',
    'a82c026e': 'Which game did I finish at last over the past weekend?',
    '6b168ec8': 'What is the number of bicycles that belong to me?',
    '86f00804': 'Which book have I been reading at present?',
    '60d45044': 'Which variety of rice do I like the most?',
    '7527f7e2': 'What amount did I pay for the designer handbag?',
    '8a137a7f': 'Which kind of light bulb did I change in the lamp by my bed?',
    '8e9d538c': 'What number of skeins of worsted-weight yarn did I discover in my yarn stash?',
    '58bf7951': 'Which production did I see at the local community theatre?',
    '19b5f2b3': 'What was the duration of my stay in Japan?',
    '75499fd8': 'Which dog breed does my dog belong to?',
    'ccb36322': 'Which music streaming platform have I used recently?',
    '726462e0': 'What discount did I receive on my first order from the new clothing label?',
    'f4f1d8a4': 'Which person gave me the new stand mixer for my birthday?',
}


def read(path):
    return json.loads(Path(path).read_text())


def save(out, name, value):
    path = out/name
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root, out):
    raw = read(root/'data/longmemeval_s.json')
    pilot = read(root/'results/session-qa-routing-v2/items.json')
    available, exclusions = eligible_records(raw, pilot)
    available.sort(key=lambda r: r['qid'])
    random.Random(0).shuffle(available)
    if len(available) != 56:
        raise ValueError('Expected 56 independent eligible non-pilot sessions')
    source = {r['question_id']: r for r in raw}
    groups = {}
    for split, records in [('train', available[:24]), ('calibration', available[24:32]), ('test', available[32:])]:
        items = []
        for row in records:
            record = source[row['qid']]
            session = dict(zip(record['haystack_session_ids'], record['haystack_sessions']))[row['source_session_id']]
            if not session:
                raise ValueError('Missing full answer-bearing conversation')
            item = dict(row, sid='qa-'+hashlib.sha256(row['qid'].encode()).hexdigest()[:16],
                        answer_session_ids=record['answer_session_ids'], context=session, split=split)
            if split == 'test':
                item['paraphrase'] = PARAPHRASES[item['qid']]
                if question_key(item['question']) == question_key(item['paraphrase']):
                    raise ValueError('Paraphrase must change question wording')
            items.append(item)
        groups[split] = items
    # Check all identifiers across all three splits and the old development pilot.
    sets = [[{str(r[k]) for r in rows} for k in ['qid', 'source_session_id']] +
            [{question_key(r['question']) for r in rows}] for rows in groups.values()]
    for i in range(3):
        for j in range(i):
            if any(a & b for a, b in zip(sets[i], sets[j])):
                raise ValueError('Split identity overlap')
    if set(PARAPHRASES) != {r['qid'] for r in groups['test']}:
        raise ValueError('Question-only paraphrase manifest must cover exactly the test set')
    examples = {s: build_examples(rows, s, seed=0) for s, rows in groups.items()}
    projection = Path('runs/prepared/projector.pt')
    corpus_hash = sha256(root/'data/longmem-k1-distractors.jsonl')
    meta = read(projection.with_suffix('.json'))
    if meta.get('corpus_sha256') != corpus_hash or meta.get('layers') != [13, 14, 15]:
        raise ValueError('Projection/corpus/layer mismatch')
    design = dict(seed=0, layers=[13, 14, 15], physical_gpu=1,
                  split_counts={s: len(rows) for s, rows in groups.items()},
                  dataset_sha256=sha256(root/'data/longmemeval_s.json'), corpus_sha256=corpus_hash,
                  projection=str(projection), max_new_tokens=64,
                  delta_storage_path=str(out/'deltas'), storage_volatile=False,
                  context='full answer-bearing session', key='Q: {question}\nA:', value='answer',
                  writer='Unchanged pilot writer: grounded value, bare question keys; layers13–15,50steps,L2=.1',
                  split_rule='Sort eligible qids; Random(0).shuffle; first24train,next8calibration,last24test',
                  prior_pilot_excluded=True, eligible_count=56, exclusions=exclusions,
                  initial_judge=str(root/'results/bert-routing-judge-v1/checkpoint'),
                  training={'answerability_epochs':4,'answerability_lr':2e-5,'batch_size':16,
                            'listwise_epochs':3,'listwise_lr':1e-5,'checkpoint_selection':'fixed final epoch'},
                  calibration='One positive listwise temperature minimizing gold-session NLL on8 calibration query banks only',
                  methods=['original_bert','retrained_bert','calibrated_bert','activation'],
                  primary='calibrated_bert', bank_sizes=[8,16,24],
                  bank_rule='Per question: own delta plus seeded nested distractors;24queries at every K',
                  tie_break='Lexicographically smallest delta ID',
                  activation_formula='sum_l ||delta_W_s,l @ k_l(q)||^2 / sum_l ||delta_W_s,l||_F^2',
                  activation_key='Last bare question-key token captured under base Llama weights',
                  paraphrases='One manually authored question-only paraphrase/item; frozen before model run',
                  probe_batch_size=8,
                  probe_protocol='Fixed batches of8left-padded independent questions per loaded delta; restore base before loading each delta',
                  final_protocol='Fresh generation for each unique query/selected-delta pair using the identical fixed8question batch; shared across matching method/K selections',
                  expected_probes=1792, expected_writes=56,
                  expected_accuracy='100% is an aspirational target, not a forecast; prior filtered K1 is historical reference only')
    out.mkdir(parents=True, exist_ok=False)
    save(out, 'design.json', design)
    save(out, 'items.json', groups)
    for split, rows in examples.items():
        save(out, split+'_answerability.json', rows)
    save(out, 'manifest_sha256.json', {'design':sha256(out/'design.json'),'items':sha256(out/'items.json'),
                                      'train_answerability':sha256(out/'train_answerability.json')})
    save(out, 'completion.json', dict(completed=False, stage='prepared'))
    save(out, 'status.json', dict(stage='prepared'))
    return design, groups


def queries(items, variant='exact'):
    return [dict(query_id=i['qid']+(':'+variant if variant != 'exact' else ''),
                 qid=i['qid'], variant=variant, question=i['question'] if variant == 'exact' else i['paraphrase'],
                 answer=i['answer'], answer_session_ids=i['answer_session_ids'], own_sid=i['sid']) for i in items]


def smaller_bank(trial, k):
    own = trial['own_sid']
    rest = sorted(c['sid'] for c in trial['candidates'] if c['sid'] != own)
    seed = int(hashlib.sha256(('bank0:'+trial['qid']).encode()).hexdigest()[:16],16)
    random.Random(seed).shuffle(rest)
    return sorted([own]+rest[:k-1])


def generate_batch(model, tokenizer, rows):
    """Independent greedy questions with fixed left-padding; no answer/context."""
    import torch
    tokenizer.padding_side='left'
    encoded=tokenizer([prompt(r['question']) for r in rows],return_tensors='pt',padding=True,truncation=False).to('cuda:0')
    width=encoded.input_ids.shape[1]
    with torch.inference_mode():
        generated=model.generate(**encoded,do_sample=False,max_new_tokens=64,pad_token_id=tokenizer.pad_token_id)
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    result=[]
    for tokens in generated[:,width:].cpu().tolist():
        length=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens))
        result.append(dict(text=tokenizer.decode(tokens[:length],skip_special_tokens=True).strip(),
                           tokens=length,hit_cap=length>=64,
                           batch_query_ids=[r['query_id'] for r in rows]))
    return result


def evaluate_routes(test_trials, scores, activation_scores, temperature):
    result = []
    for variant in ['exact','paraphrase']:
        rows = [t for t in test_trials if t['variant'] == variant]
        if len(rows) != 24:
            raise ValueError('Each condition requires24independent test sessions')
        for k in [8,16,24]:
            for method in ['original_bert','retrained_bert','calibrated_bert','activation']:
                trials=[]
                for t in rows:
                    bank=smaller_bank(t,k)
                    if method == 'activation':
                        values=activation_scores[t['query_id']]
                    else:
                        values=scores[t['query_id']]['original' if method=='original_bert' else 'retrained']
                    # Positive temperature preserves ranks; do not round sigmoid probabilities.
                    ranked=sorted(bank,key=lambda s:(-values[s],s))
                    chosen=ranked[0]
                    candidates={c['sid']:c for c in t['candidates']}
                    own=candidates[t['own_sid']]
                    trials.append(dict(t, candidates=[candidates[s] for s in bank],
                                       selected_sid=chosen,selected_source_session_id=candidates[chosen]['source_session_id'],
                                       routing_correct=chosen==t['own_sid'],rank=ranked.index(t['own_sid'])+1,
                                       margin=values[ranked[0]]-values[ranked[1]],
                                       selected_score=values[chosen],own_contains=fired(own['candidate'],t['answer']),
                                       own_exact=normalize(own['candidate'])==normalize(t['answer'])))
                result.append(dict(condition=variant,bank_size=k,method=method,trials=trials))
    # Mechanism invariant: calibration alone cannot change selections.
    for i in range(0,len(result),4):
        if [r['selected_sid'] for r in result[i+1]['trials']] != [r['selected_sid'] for r in result[i+2]['trials']]:
            raise AssertionError('Positive temperature changed routing order')
    return dict(results=result,temperature=temperature)


def summarize(evaluation):
    for cell in evaluation['results']:
        rows=cell['trials'];n=len(rows)
        summary=dict(n_items=n,chance_routing_accuracy=1/cell['bank_size'],
                     mean_reciprocal_rank=sum(1/r['rank'] for r in rows)/n)
        for name,key in [('routing_accuracy','routing_correct'),('answer_containment','final_contains'),
                         ('answer_exact','final_exact'),('raw_answer_exact','raw_final_exact'),
                         ('own_answer_containment','own_contains'),('own_answer_exact','own_exact')]:
            count=sum(bool(r[key]) for r in rows)
            summary[name]=count/n;summary[name+'_ci95']=interval(count,n)
        cell['summary']=summary


def run(root,out):
    design=read(out/'design.json');groups=read(out/'items.json')
    frozen=read(out/'manifest_sha256.json')
    if frozen!={'design':sha256(out/'design.json'),'items':sha256(out/'items.json'),
                'train_answerability':sha256(out/'train_answerability.json')}:
        raise ValueError('Frozen protocol changed')
    delta_dir=Path(design['delta_storage_path'])
    delta_dir.mkdir(parents=True,exist_ok=True)
    existing=sum(p.stat().st_size for p in delta_dir.rglob('*.pt'))
    need=56*704644657-existing+700*1024**2
    if shutil.disk_usage(out).free<need:
        raise RuntimeError(f'Need {need/1024**3:.2f}GiB additional space for durable bank+checkpoint')
    from zsre_routing_experiment import require_gpu1
    gpu=require_gpu1()
    busy=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
    if busy.stdout.strip():raise RuntimeError('Physical GPU1 busy')
    import torch
    from routing_ladder_models import AlphaEditAdapter
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    sampler=subprocess.Popen([sys.executable,str(root/'sample_session_qa_resources.py'),'--pid',str(os.getpid()),
                              '--out',str(out),'--delta-root',str(delta_dir)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    begin=time.monotonic();adapter=None;phase_peaks={}
    save(out,'completion.json',dict(completed=False,stage='running'))
    def status(stage,**extra):save(out,'status.json',dict(stage=stage,gpu_uuid=gpu,**extra))
    try:
        status('section_a_model_loading')
        adapter=AlphaEditAdapter(root,delta_dir,layers=[13,14,15],projection_path=Path(design['projection']))
        allitems=[i for rows in groups.values() for i in rows]
        # Validate every complete grounding session before the first write.
        lengths={}
        for index,item in enumerate(allitems):
            req=write_request(item,index)
            text=req['value_context_templates'][0][1].format(req['prompt']).format(req['subject'])+item['answer']
            lengths[item['qid']]=len(adapter.tok.encode(text))
            if lengths[item['qid']]+64>adapter.model.config.max_position_embeddings:
                raise ValueError('Grounding context exceeds model capacity; no silent truncation')
        save(out,'grounding_token_lengths.json',lengths)
        writes=read(out/'writes.json') if (out/'writes.json').exists() else []
        written={r['sid']:r for r in writes if r.get('write_success')}
        for index,item in enumerate(allitems):
            if item['sid'] in written:
                if sha256(delta_dir/item['sid']/'v1.pt')!=written[item['sid']]['sha256']:
                    raise ValueError('Previously written delta hash changed')
                continue
            status('section_a_writing',split=item['split'],done=len(written),total=56,qid=item['qid'])
            adapter.store.deactivate()
            base=adapter.generate(prompt(item['question']),64)
            started=time.monotonic()
            adapter.store.write_session(item['sid'],adapter.tok,[write_request(item,index)],adapter.hp,
                                        torch.zeros_like(adapter.P),adapter.P,1)
            immediate=adapter.store.run(item['sid'],1,lambda m:adapter.generate(prompt(item['question']),64))
            file=delta_dir/item['sid']/'v1.pt'
            row=dict(sid=item['sid'],qid=item['qid'],split=item['split'],write_success=True,
                     base=base,immediate=immediate,own_contains=fired(candidate_text(immediate['text']),item['answer']),
                     sha256=sha256(file),bytes=file.stat().st_size,elapsed_seconds=time.monotonic()-started)
            writes.append(row);written[item['sid']]=row;save(out,'writes.json',writes)
            print('WRITE',len(written),56,item['split'],item['qid'],row['own_contains'],flush=True)
        phase_peaks['write']=torch.cuda.max_memory_allocated()/1024**3
        # Writer/projection state is no longer needed during Section B.
        adapter.store.deactivate();adapter.P=None;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        banks={}
        for split in ['train','calibration','test']:
            filename=split+'_trials.json'
            toquery=queries(groups[split])+(queries(groups[split],'paraphrase') if split=='test' else [])
            saved=read(out/filename) if (out/filename).exists() else [dict(q,candidates=[]) for q in toquery]
            ready={t['query_id']:t for t in saved}
            if list(ready)!=[q['query_id'] for q in toquery]:
                raise ValueError('Saved query order differs from frozen batching')
            bank=sorted(groups[split],key=lambda r:r['sid'])
            for stored_index,stored in enumerate(bank):
                counts=[sum(c['sid']==stored['sid'] for c in ready[q['query_id']]['candidates']) for q in toquery]
                if all(count==1 for count in counts):
                    continue
                if any(count!=0 for count in counts):
                    raise ValueError('Saved delta/query batch incomplete or duplicated')
                status('section_b_probing',split=split,done=stored_index,total=len(bank),sid=stored['sid'])
                for start in range(0,len(toquery),8):
                    chunk=toquery[start:start+8]
                    probes=adapter.store.run(stored['sid'],1,lambda m:generate_batch(m,adapter.tok,chunk))
                    for q,probe in zip(chunk,probes):
                        ready[q['query_id']]['candidates'].append(dict(sid=stored['sid'],source_session_id=stored['source_session_id'],
                                                                     candidate=candidate_text(probe['text']),probe=probe))
                save(out,filename,saved)
                print('PROBE_DELTA',split,stored_index+1,len(bank),'queries',len(toquery),flush=True)
            if any(len(t['candidates'])!=len(bank) for t in saved):
                raise ValueError('Incomplete probe bank')
            banks[split]=saved
        phase_peaks['probe']=torch.cuda.max_memory_allocated()/1024**3
        activation_path=out/'activation_scores.json'
        if activation_path.exists():
            activation_scores=read(activation_path)['scores']
        else:
            status('activation_scoring')
            adapter.store.deactivate();names=adapter.store.weight_names
            current={};handles=[];keys={}
            for name in names:
                def hook(mod,args,output,name=name):current[name]=args[0][0,-1].detach().clone()
                handles.append(adapter.model.get_submodule(name.removesuffix('.weight')).register_forward_hook(hook))
            try:
                for trial in banks['test']:
                    x=adapter.tok(prompt(trial['question']),return_tensors='pt').to('cuda:0')
                    with torch.inference_mode():adapter.model.model(**x,use_cache=False)
                    keys[trial['query_id']]={name:current[name] for name in names}
            finally:
                for handle in handles:handle.remove()
            qids=[t['query_id'] for t in banks['test']]
            activation_scores={q:{} for q in qids}
            for stored in groups['test']:
                delta=torch.load(delta_dir/stored['sid']/'v1.pt',map_location='cpu',weights_only=True)
                numerator=torch.zeros(len(qids),device='cuda:0',dtype=torch.float64)
                denominator=torch.zeros((),device='cuda:0',dtype=torch.float64)
                with torch.inference_mode():
                    for name in names:
                        d=delta[name].to('cuda:0');k=torch.stack([keys[q][name] for q in qids],dim=1)
                        numerator+=(d@k).double().square().sum(0)
                        denominator+=d.double().square().sum()
                        del d,k
                if denominator.item()<=0:raise ValueError('Zero delta energy')
                for idx,q in enumerate(qids):activation_scores[q][stored['sid']]=(numerator[idx]/denominator).item()
                del delta
            save(out,'activation_scores.json',dict(scores=activation_scores,formula=design['activation_formula'],
                                                   question_key_under_base_model=True,ground_truth_used_for_scoring=False))
        # Release all Llama weights before BERT training.
        adapter.store.deactivate();del adapter;adapter=None;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        status('bert_training_and_calibration')
        from larger_bert_calibration import train_and_calibrate
        if (out/'bert_scores.json').exists() and (out/'heldout_calibration.json').exists():
            bert=read(out/'bert_scores.json');calibration=read(out/'calibration.json')
            scores=bert['scores'] if 'scores' in bert else bert
        else:
            trained=train_and_calibrate(out,banks['train'],banks['calibration'],banks['test'],
                                        read(out/'train_answerability.json'),Path(design['initial_judge']),seed=0)
            scores=trained['scores'];calibration=trained['calibration']
        phase_peaks['bert']=torch.cuda.max_memory_allocated()/1024**3
        evaluation=evaluate_routes(banks['test'],scores,activation_scores,calibration['temperature'])
        save(out,'routing_selections.json',evaluation)
        # Fresh final generation under the selected delta, identical protocol for every method/K.
        status('final_model_loading')
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from AlphaEditSessionStore import SessionFFNMemory
        tok=AutoTokenizer.from_pretrained(root/'models/Llama-3.1-8B-Instruct',local_files_only=True);tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(root/'models/Llama-3.1-8B-Instruct',torch_dtype=torch.float32,local_files_only=True).to('cuda:0').eval()
        names=[f'model.layers.{layer}.mlp.down_proj.weight' for layer in [13,14,15]]
        store=SessionFFNMemory(model,names,delta_dir)
        finals=read(out/'fresh_answers.json') if (out/'fresh_answers.json').exists() else []
        cache={(r['query_id'],r['sid']):r for r in finals}
        pairs={(r['query_id'],r['selected_sid']) for cell in evaluation['results'] for r in cell['trials']}
        torch.cuda.reset_peak_memory_stats()
        try:
            test_query_order=queries(groups['test'])+queries(groups['test'],'paraphrase')
            batch_index={q['query_id']:i//8 for i,q in enumerate(test_query_order)}
            needed_batches=sorted({(sid,batch_index[q]) for q,sid in pairs if (q,sid) not in cache})
            all_trial_by_q={t['query_id']:t for t in banks['test']}
            total_final_generations=0
            for sid,index in needed_batches:
                chunk=test_query_order[index*8:index*8+8]
                status('fresh_final_answers',done=len(cache),total=len(pairs),sid=sid,batch_index=index)
                generated=store.run(sid,1,lambda m:generate_batch(m,tok,chunk))
                total_final_generations+=len(generated)
                for q,final in zip(chunk,generated):
                    pair=(q['query_id'],sid)
                    if pair not in pairs or pair in cache:continue
                    selected=next(c for c in all_trial_by_q[q['query_id']]['candidates'] if c['sid']==sid)
                    if final['text']!=selected['probe']['text']:
                        raise ValueError('Fresh deterministic final differs from its identically batched probe')
                    saved=dict(query_id=q['query_id'],sid=sid,final=final,matches_probe=True)
                    finals.append(saved);cache[pair]=saved
                save(out,'fresh_answers.json',finals)
                print('FINAL_BATCH',len(cache),len(pairs),flush=True)
            for cell in evaluation['results']:
                for row in cell['trials']:
                    pair=(row['query_id'],row['selected_sid'])
                    final=cache[pair]['final'];text=candidate_text(final['text'])
                    row.update(final=final,final_answer=text,final_contains=fired(text,row['answer']),
                               final_exact=normalize(text)==normalize(row['answer']),
                               raw_final_exact=normalize(final['text'])==normalize(row['answer']))
        finally:
            store.deactivate()
        phase_peaks['final']=torch.cuda.max_memory_allocated()/1024**3
        summarize(evaluation);save(out,'evaluation.json',evaluation)
        before={r['sid']:r['sha256'] for r in writes}
        after={sid:sha256(delta_dir/sid/'v1.pt') for sid in before}
        if before!=after:raise ValueError('Delta hashes changed during recall')
        probe_count=sum(len(t['candidates']) for rows in banks.values() for t in rows)
        if probe_count!=1792 or len(writes)!=56 or set(cache)!=pairs:
            raise ValueError('Experiment completeness check failed')
        sampler.terminate();sampler.wait(timeout=15)
        samples=[json.loads(line) for line in (out/'resource_usage.jsonl').read_text().splitlines()]
        def peak(key):return max((s[key]/1024 for s in samples if key in s),default=0)
        storage=sum(p.stat().st_size for p in delta_dir.rglob('*.pt'))
        save(out,'resource_summary.json',dict(peak_gpu_used_gib=peak('gpu_used_mib'),peak_host_rss_gib=peak('host_rss_mib'),
             peak_host_hwm_gib=peak('host_peak_rss_mib'),delta_storage_bytes=storage,delta_storage_gib=storage/1024**3,
             delta_bytes_per_session=storage/56,base_weight_gib=29.915054321289062,deltas_storage_path=str(delta_dir),
             storage_volatile=False,torch_peak_gib=phase_peaks,sample_interval_seconds=10,
             elapsed_seconds=time.monotonic()-begin,scope='Successful process from model loading through final verification; externalGPUboard and experimentPIDRSS',
             checkpoint_bytes=sum(p.stat().st_size for p in (out/'checkpoint').rglob('*') if p.is_file())))
        save(out,'completion.json',dict(completed=True,writes=56,expected_writes=56,fresh_probes=probe_count,
             expected_probes=1792,fresh_final_count=len(cache),expected_fresh_final_count=len(pairs),
             final_batch_companion_generations=total_final_generations,
             final_results=576,hashes_unchanged=True,delta_sha256=after,gpu_uuid=gpu,
             fresh_finals_match_probes=True,protocol_frozen_sha256=frozen,storage_volatile=False))
        status('completed');print('COMPLETED',out,flush=True)
    except BaseException as exc:
        save(out,'completion.json',dict(completed=False,error=repr(exc)))
        status('failed',error=repr(exc));raise
    finally:
        if adapter is not None:adapter.store.deactivate()
        if sampler.poll() is None:
            sampler.terminate();sampler.wait(timeout=15)


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=root/'results/session-routing-scale-v1')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.out=a.out.resolve()
    if not a.out.exists():prepare(root,a.out)
    if a.prepare_only:
        print(a.out/'design.json');return
    if (a.out/'completion.json').exists() and read(a.out/'completion.json').get('completed'):
        raise ValueError('Completed experiment is immutable; choose a new output for another run')
    run(root,a.out)


if __name__=='__main__':main()
