"""Corrected, fully supervised Q/A memory protocol; no evidence-capsule targets."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import statistics
import sys

from comparison_core import digest,read,save

ROOT=Path(__file__).resolve().parent
CODE=Path('/home/julie/Desktop/draft_1')
LEGACY=Path('/home/julie/sessioncontrol-backups/cleanup-20261015T022610Z/runs/scaling-real512-compact-v1')
PRIOR=ROOT/'runs/official64-fourmethod-v1'
DATA=Path('/home/julie/alphaeditSessionControl/data/longmemeval_s.json')
SIZES=(8,16,32,64,128,256,500)
METHODS=('our_delta','plain_alphaedit','rag','kv_full')
RESERVE=2*2**30


def hashed(value):return hashlib.sha256(value.encode()).hexdigest()


def question_plan(sessions,queries,sizes=SIZES):
    plan=[]
    for n in sizes:
        present={r['sid'] for r in sessions[:n]}
        ids=[q['qid'] for q in queries if set(q['support_sids'])<=present]
        if not ids:raise ValueError('No complete official questions at checkpoint')
        plan.append(dict(n=n,qids=ids,questions=len(ids)))
    return plan


def prepare(out):
    if out.exists():raise FileExistsError('Use a fresh corrected-run directory')
    from transformers import AutoTokenizer
    prior=read(PRIOR/'design.json');official=read(DATA);old=read(LEGACY/'items.json')[:64]
    records={};queries=[]
    for row in official:
        qid=str(row['question_id']);sessions=dict(zip(row['haystack_session_ids'],row['haystack_sessions']))
        dates=dict(zip(row['haystack_session_ids'],row['haystack_dates']))
        queries.append(dict(qid=qid,question=row['question'],answer=str(row['answer']),
                            question_date=row['question_date'],question_type=row['question_type'],
                            support_sids=row['answer_session_ids'],abstention=qid.endswith('_abs'),
                            original64=qid in {x['qid'] for x in old}))
        for sid in row['answer_session_ids']:
            r=records.setdefault(sid,dict(sid=sid,context=sessions[sid],bindings=[]))
            if r['context']!=sessions[sid]:raise ValueError('Conflicting conversation text')
            r['bindings'].append(dict(qid=qid,question=row['question'],answer=str(row['answer']),date=dates[sid]))
    order=[r['source_session_id'] for r in old];seen=set(order)
    # Preserve the old 64-session trajectory exactly; append whole support groups.
    for q in sorted(queries,key=lambda r:hashed(r['qid'])):
        for sid in q['support_sids']:
            if sid not in seen:order.append(sid);seen.add(sid)
    bank=[records[sid] for sid in order[:500]]
    if len(bank)!=500 or len(seen)!=940:raise ValueError('Unexpected official source bank')
    for r,item in zip(bank[:64],old):
        if len(r['bindings'])!=1 or r['context']!=item['context'] or r['sid']!=item['source_session_id']:
            raise ValueError('Old conversation cannot be reused unchanged')
        b=r['bindings'][0]
        if (b['qid'],b['question'],b['answer'])!=(item['qid'],item['question'],item['answer']):
            raise ValueError('Old question/answer differs from official release')
    tok=AutoTokenizer.from_pretrained(prior['model'],local_files_only=True)
    max_target=max(len(tok.encode(q['answer'],add_special_tokens=False)) for q in queries)
    if max_target+2>128:raise ValueError('Reference exceeds fixed answer budget')
    plan=question_plan(bank,queries)
    out.mkdir(parents=True)
    for folder in ['independent','shared',*METHODS]:(out/folder).mkdir()
    (out/'independent/factors').mkdir();(out/'shared/post_keys').mkdir()
    source_hashes={};legacy_hashes=read(LEGACY/'written_bank_sha256.json');ledger=[]
    for i,(r,item) in enumerate(zip(bank[:64],old)):
        source=LEGACY/'factors'/(item['sid']+'.pt');h=legacy_hashes[item['sid']]
        if digest(source)!=h:raise ValueError('Saved answer delta changed')
        file=f'{i:04d}.pt';(out/'independent/factors'/file).symlink_to(source)
        source_hashes[str(source)]=h
        ledger.append(dict(index=i,sid=r['sid'],file=file,sha256=h,reused=True,
                           legacy_memory_id=item['sid'],source=str(source),bindings=1))
    save(out/'independent/ledger.json',ledger)
    shared=read(PRIOR/'shared/completion.json')
    for n in (8,16,32,64):
        source=PRIOR/'shared'/f'weights-{n}.pt';h=shared['patch_sha256'][str(n)]
        if digest(source)!=h:raise ValueError('Saved shared checkpoint changed')
        (out/'shared'/source.name).symlink_to(source);source_hashes[str(source)]=h
    old_edits=read(PRIOR/'shared/edits.json');shared_ledger=[]
    for i,r in enumerate(bank[:64]):
        source=PRIOR/'shared/post_update_keys'/f'{i}.pt';h=shared['post_key_sha256'][str(i)]
        if digest(source)!=h:raise ValueError('Saved preservation key changed')
        (out/'shared/post_keys'/f'{i}.pt').symlink_to(source);source_hashes[str(source)]=h
        shared_ledger.append(dict(index=i,sid=r['sid'],reused=True,cache_norm=old_edits[i]['cache_norm'],bindings=1))
    save(out/'shared/ledger.json',shared_ledger)
    save(out/'sessions.json',bank);save(out/'queries.json',queries)
    d=dict(name='qa500-checkpoints-tenpass-v1',physical_gpu=1,model=prior['model'],embedding_model=prior['embedding_model'],
           projection=prior['projection'],projection_sha256=prior['projection_sha256'],dataset=str(DATA),
           dataset_sha256=digest(DATA),layers=[13,14,15],dtype='float32',l2=.1,write_steps=50,
           sizes=list(SIZES),plan=plan,repeats=10,answer_cap=128,prefill_chunk=1024,
           reused_sessions=64,new_sessions=436,total_sessions=500,official_release_questions=500,
           eligible_final_questions=plan[-1]['questions'],max_reference_tokens=max_target,
           source_hashes=source_hashes,old_single_session_qids=[r['qid'] for r in old],
           write_protocol='FULL source conversation + literal official question + COMPLETE reference answer for every write, in both AlphaEdit conditions. '
           'Target is answer plus EOS. No capsule extraction, summaries, reference clipping or derived quote questions.',
           unit='One conversation per delta. Eight conversations in the full source pool have two question bindings; selected bank contains a subset. '
           'For a multi-session item its complete supervised answer is provided to each associated conversation write. '
           'This deliberately tests exact-question answer storage/routing, not derivation of the answer from partial evidence or held-out reasoning.',
           scope='Oracle supporting/related conversation bank from official metadata. All Q/A labels are intentionally visible at write time. '
           'Earlier 64 official items and their saved weights remain first, then deterministic SHA256(question ID) support groups. '
           'Stop at 500 conversations. This does not evaluate every one of the 500 release questions: only questions with their complete support set present are queried.',
           evaluation='Fresh measurements at EVERY checkpoint. Evaluate ALL eligible official questions, ten actual passes on one reused frozen bank, '
           'with deterministic greedy decoding and matched shuffled query order across methods within each pass. '
           'Also report the fixed original64 cohort where present, new questions, question-type and abstention breakdowns. '
           'Changing eligible cohorts changes difficulty; the original64 panel separately measures retention.',
           routing='Original question-only normalized activation update energy over all first-N deltas, layers 13-15; select ARGMAX and directly generate under that delta. '
           'Lexical session-ID tie break. No BERT, generated candidate scorer, context retrieval, reference lookup or evidence-fusion reader in our method.',
           baseline='One original dense AlphaEdit model with a continuous preservation accumulator, same Q/A/context inputs. '
           'Reuse saved patches through 64, reconstruct the exact accumulator from saved post-update keys, append new conversations and save each new checkpoint. '
           'Direct question-to-answer generation; no session control. Independent writing uses the compact zero-accumulator solver; this is a complete-system comparison.',
           rag='Matched question-key MiniLM retrieval, top-one stored conversation. Store full conversation AND all associated exact questions/reference answers. '
           'Supply the selected record to the same base LLM for generation. This is supervised Q/A-memory RAG, not context-only RAG.',
           kv='Full first-N raw conversation + Q/A records in one warm reusable prefix. Actual capacity checked independently each pass. '
           'No truncation or silent retrieval fallback; infeasible checkpoints have no F1/TTFT. Prefix-build time and payload recorded separately.',
           repetition='Ten inference measurement passes, NOT ten independently trained memories. Report means and SAMPLE SD across ten per-pass metrics. '
           'Greedy quality SD may be zero. Do not present repeated identical answers as ten independent training replicates.',
           metrics='Whole-answer custom normalized word-token F1, exact match, reference containment, source hit@1/rank/MRR and support recall@6, '
           'visible/generated TTFT, complete query latency, output length/cap rate, disk bank bytes, host RSS, CUDA allocated/reserved peaks. '
           'Primary timing is mean per-pass query-mean, with mean of per-pass medians and p95 also explicit. Lexical F1 is not semantic factuality.',
           timing='Query arrival through routing/delta load+apply/tokenization/prefill to first nonempty final answer. No reference enters inference. '
           'Model startup, bank writing and shared checkpoint loading are excluded from query TTFT. Two unrelated one-token warmups. '
           'No OS-cache flush. Method order rotates across passes; query order is matched. Query times include restoration after generation.',
           memory='One isolated inference process per method/pass. Reset CUDA peaks and sample host RSS for each N, including bank setup. '
           'Common model weights counted in GPU peaks; disk, CPU and GPU tiers stay separate. Prefix-only factor/index/cache payload at each N. '
           'Disk bytes measured at each pass and do not sum all retained research snapshots.',
           legacy='64 rounded dense answer deltas were previously approximated with rank-one factors and validated for own-question/pilot output parity. '
           'Reuse those exact compact files; do not describe them as globally lossless compression.',
           write_time='No measured write-time comparison; retain the existing theoretical complexity discussion.')
    save(out/'design.json',d)
    save(out/'manifest.json',{f:digest(out/f) for f in ['design.json','sessions.json','queries.json']})
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1))
    save(out/'reuse_audit.json',dict(answer_factors_reused=64,shared_checkpoints_reused=[8,16,32,64],
                                    preservation_keys_reused=64,evidence_only_deltas_reused=0,
                                    new_conversations_to_write=436,eligible_queries_by_checkpoint=plan))
    storage_check(out)
    print('Prepared checkpoints:',[(p['n'],p['questions']) for p in plan],flush=True)


def check(out):
    d=read(out/'design.json')
    for f,h in read(out/'manifest.json').items():
        if digest(out/f)!=h:raise ValueError('Corrected protocol changed: '+f)
    if digest(d['dataset'])!=d['dataset_sha256'] or digest(d['projection'])!=d['projection_sha256']:
        raise ValueError('Dataset/projector changed')
    return d


def storage_check(out):
    d=check(out);config=read(Path(d['model'])/'config.json');h=config['hidden_size'];m=config['intermediate_size'];l=len(d['layers'])
    bindings=sum(len(r['bindings']) for r in read(out/'sessions.json'))
    factors=l*(h+m)*4*bindings
    patch=l*h*m*4
    bound=factors+(len(SIZES)+2)*patch+l*m*4*bindings+2*2**30
    free=shutil.disk_usage(out).free
    result=dict(compact_factor_payload_bytes=factors,max_rank=2,maximum_factor_bytes=512*1024,
                dense_patch_bytes=patch,max_dense_files=len(SIZES)+2,
                dense_policy='One snapshot per seven checkpoints and an atomic rolling restart patch, never a dense delta per session.',
                conservative_new_disk_bytes=bound,reserve_bytes=RESERVE,free_bytes=free,fits=free>=bound+RESERVE)
    save(out/'storage_budget.json',result)
    if not result['fits']:raise RuntimeError('Insufficient storage plus 2 GiB reserve')
    return result


def require_disk(out,size):
    if shutil.disk_usage(out).free<size+RESERVE:raise RuntimeError('Write would violate disk reserve')


def now():return datetime.now(timezone.utc).isoformat()


def status(folder,stage,**fields):save(folder/'status.json',dict(stage=stage,updated_utc=now(),**fields))


def load(out,folder):
    d=check(out);folder.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(CODE));os.chdir(CODE)
    from experiment_utils import require_gpu1
    import subprocess
    gpu=require_gpu1()
    busy=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
    if busy.stdout.strip():raise RuntimeError('GPU1 occupied; no fallback')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if torch.cuda.device_count()!=1:raise RuntimeError('GPU1 must be the only visible GPU')
    torch.set_num_threads(8);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.backends.cuda.matmul.allow_tf32=False
    from official64_core import snapshot_code
    files=[ROOT/f for f in ['qa500_core.py','qa500_build.py','qa500_eval.py','comparison_core.py','full500_core.py']]
    files += [CODE/f for f in ['session_write.py','question_key.py','compact_session_memory.py','answer_only_generation.py',
                 'experiment_utils.py','memory_comparison_core.py','ttft_timing.py','AlphaEdit/AlphaEdit_main.py',
                 'AlphaEdit/compute_z.py','AlphaEdit/compute_ks.py','AlphaEdit/grounded_value.py','config/alphaedit-hparams.json']]
    hashes={str(f):digest(f) for f in files}
    if (folder/'provenance.json').exists():
        prov=read(folder/'provenance.json')
        if hashes!=prov['code_sha256']:raise ValueError('Worker code changed since checkpoint; do not mix executions')
    else:
        prov=snapshot_code(folder,files);prov.update(physical_gpu=1,gpu_uuid=gpu,torch_version=torch.__version__,started_utc=now())
        save(folder/'provenance.json',prov)
    status(folder,'loading_model')
    tok=AutoTokenizer.from_pretrained(d['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(d['model'],torch_dtype=torch.float32,attn_implementation='sdpa',
                                            local_files_only=True).to('cuda:0').eval()
    return d,torch,tok,model


def answer_requests(record,tok,index):
    from session_write import write_request
    result=[]
    for j,b in enumerate(record['bindings']):
        request=write_request(dict(context=record['context'],question=b['question'],answer=b['answer']),index*2+j)
        request['target_new']['str']=b['answer']+tok.eos_token
        assert request['target_new']['str']==b['answer']+tok.eos_token
        result.append(request)
    return result


def canonical(payload):
    return {n:(u[:,None] if u.ndim==1 else u,v[:,None] if v.ndim==1 else v) for n,(u,v) in payload.items()}


def score(payload,keys):
    if all(u.shape[1]==1 for u,v in payload.values()):
        from compact_session_memory import factor_scores
        return factor_scores({n:(u[:,0],v[:,0]) for n,(u,v) in payload.items()},keys)[0]
    from full500_core import factor_energy
    return float(factor_energy(payload,keys)[0])


def document(record):
    import json
    # All write-time information, including the complete official answer, is retained.
    return ('Session: '+record['sid']+'\nConversation:\n'+json.dumps(record['context'],ensure_ascii=False)+
            '\n'+'\n'.join('Stored question: '+b['question']+'\nStored answer: '+b['answer'] for b in record['bindings']))


def order(qids,repeat,n):return sorted(qids,key=lambda q:hashed(f'{repeat}/{n}/{q}'))


def mean_sd(values):
    if len(values)!=10:raise ValueError('Require ten actual passes')
    if any(v is None for v in values):return dict(mean=None,sd=None,n=0)
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),n=10)


def verify_reused_accumulator(out):
    """Check the saved preservation state before any new GPU writing starts."""
    import torch
    torch.set_num_threads(8)
    d=check(out);config=read(Path(d['model'])/'config.json');width=config['intermediate_size']
    accumulator=torch.zeros((len(d['layers']),width,width),dtype=torch.float32)
    for i in range(64):
        keys=torch.load(out/'shared/post_keys'/f'{i}.pt',map_location='cpu',weights_only=True)
        for layer,key in enumerate(keys):accumulator[layer]+=key[:,None]@key[None,:]
    norm=float(accumulator.norm());expected=read(out/'shared/ledger.json')[63]['cache_norm']
    if norm!=expected:raise ValueError('Old 64-session preservation accumulator cannot be exactly reconstructed')
    save(out/'shared/reuse_validation.json',dict(exact=True,conversations=64,actual_norm=norm,expected_norm=expected))
    return norm
