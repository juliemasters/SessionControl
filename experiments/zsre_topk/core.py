"""Query-only coarse search and unchanged delta-response fine scoring."""
from pathlib import Path
import hashlib
import os
from comparison_core import read,save,digest,durable_save_torch
from experiments.zsre_generated.core import now,summary

ROOT=Path(__file__).resolve().parents[2]
SIZES=[8,16,32,64,128,256,500,1000]
METHODS=['full_scan','topk_1','topk_8','topk_16','topk_32']
LABELS={'full_scan':'Full-scan activation','topk_1':'Top-k = 1','topk_8':'Top-k = 8',
        'topk_16':'Top-k = 16','topk_32':'Top-k = 32'}


def order(n,repeat):
    return sorted(range(n),key=lambda i:hashlib.sha256(f'{repeat}/{n}/{i}'.encode()).hexdigest())


def question_order(design,n,repeat):
    ids=next(p['case_ids'] for p in design['plan'] if p['n']==n)
    return sorted(ids,key=lambda i:hashlib.sha256(f'{repeat}/{n}/{i}'.encode()).hexdigest())


def status(folder,stage,**fields):save(folder/'status.json',dict(stage=stage,updated_utc=now(),**fields))


def normalized_key(keys,names):
    import torch
    if set(keys)!=set(names):raise ValueError('Missing or unexpected activation layers')
    rows=[]
    for name in names:
        value=keys[name]
        if value.ndim!=2 or value.shape[0]!=1 or value.shape[1]<1:raise ValueError('Require one question activation')
        rows.append(value[0].detach().to(device='cpu',dtype=torch.float32))
    vector=torch.cat(rows)
    if not torch.isfinite(vector).all():raise ValueError('Nonfinite activation key')
    norm=vector.norm()
    if not torch.isfinite(norm) or norm.item()<=0:raise ValueError('Empty activation key')
    return torch.nn.functional.normalize(vector,dim=0).contiguous()


class ActivationRouter:
    """Cosine shortlist over stored write questions, then native fine scores.

    index_sids permits several question keys for one conversation delta. Coarse
    similarity is their maximum, then k DISTINCT deltas are ranked. The router
    accepts activations, never an evaluation case ID, answer or support label.
    """
    def __init__(self,bank,names,index_matrix,sids,*,index_sids=None,scorer=None):
        import torch
        if not bank or len(sids)!=len(set(sids)) or set(sids)!=set(bank):raise ValueError('Bank/SID mismatch')
        self.bank=bank;self.names=tuple(names);self.sids=tuple(sids);self.matrix=index_matrix
        self.index_sids=tuple(index_sids if index_sids is not None else sids)
        if scorer is None:
            from compact_session_memory import factor_scores
            scorer=lambda payload,keys:factor_scores(payload,keys)[0]
        self.scorer=scorer
        if index_matrix is not None:
            if index_matrix.device.type!='cpu' or index_matrix.dtype!=torch.float32 or index_matrix.ndim!=2:
                raise ValueError('Require one CPU FP32 key matrix')
            if len(self.index_sids)!=index_matrix.shape[0] or set(self.index_sids)!=set(sids):raise ValueError('Index/SID mismatch')
            if index_matrix.shape[1]<1 or not torch.isfinite(index_matrix).all():raise ValueError('Invalid activation index')
            if not torch.allclose(index_matrix.norm(dim=1),torch.ones(index_matrix.shape[0]),atol=1e-5,rtol=1e-5):
                raise ValueError('Stored activation keys must be normalized')

    def route(self,keys,k=None):
        import math
        if k is None:
            candidates=list(self.sids);coarse=None;short=None
        else:
            if type(k) is not int or k<=0:raise ValueError('k must be a positive integer')
            if self.matrix is None:raise ValueError('Top-k needs a stored activation index')
            vector=normalized_key(keys,self.names)
            if vector.numel()!=self.matrix.shape[1]:raise ValueError('Query/index width mismatch')
            similarities=(self.matrix@vector).tolist()
            if any(not math.isfinite(v) for v in similarities):raise ValueError('Nonfinite cosine score')
            pooled={sid:float('-inf') for sid in self.sids}
            for sid,value in zip(self.index_sids,similarities):pooled[sid]=max(pooled[sid],value)
            coarse=sorted(self.sids,key=lambda sid:(-pooled[sid],sid))
            candidates=coarse[:min(k,len(coarse))];short=list(candidates)
        scores=[(float(self.scorer(self.bank[sid],keys)),sid) for sid in candidates]
        if any(not math.isfinite(score) for score,_ in scores):raise ValueError('Nonfinite native fine score')
        scores.sort(key=lambda item:(-item[0],item[1]))
        return dict(selected=scores[0][1],score=scores[0][0],
                    margin=scores[0][0]-scores[1][0] if len(scores)>1 else None,
                    shortlist=short,coarse_order=coarse,fine_order=[sid for _,sid in scores])


def check(out):
    d=read(out/'design.json')
    for name,want in read(out/'input_manifest.json').items():
        if digest(out/name)!=want:raise ValueError('Frozen input changed: '+name)
    for name,want in d['code_sha256'].items():
        if digest(ROOT/name)!=want:raise ValueError('Frozen execution source changed: '+name)
    source=Path(d['source_run'])
    if not read(source/'completion.json')['completed']:raise ValueError('Original experiment no longer complete')
    if digest(source/d['source_manifest_name'])!=d['source_manifest_sha256']:raise ValueError('Original input manifest changed')
    return d


def load(out,folder):
    d=check(out)
    from experiment_utils import require_gpu1
    uuid=require_gpu1()
    import torch,numpy as np,random
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if torch.cuda.device_count()!=1:raise ValueError('Only physical GPU 1 may be visible')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(8);torch.manual_seed(d['seed']);np.random.seed(d['seed']);random.seed(d['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    folder.mkdir(parents=True,exist_ok=True)
    prov=folder/'provenance.json'
    if prov.exists():
        if read(prov)['code_sha256']!=d['code_sha256']:raise ValueError('Mixed inference sources')
    else:save(prov,dict(code_sha256=d['code_sha256'],physical_gpu=1,gpu_uuid=uuid,started_utc=now()))
    status(folder,'loading_model')
    tok=AutoTokenizer.from_pretrained(d['model'],local_files_only=True)
    tok.pad_token_id=tok.eos_token_id;tok.padding_side='left' if d['dataset']=='zsre' else 'right'
    model=AutoModelForCausalLM.from_pretrained(d['model'],torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
    return d,torch,tok,model


def capture(d,model,tok,names,query):
    if d['dataset']=='zsre':
        from experiments.session_zsre.core import capture_query_keys
        return capture_query_keys(model,tok,names,query)
    from compact_session_memory import capture_keys
    return capture_keys(model,tok,names,[query])


def generate(d,model,tok,query):
    if d['dataset']=='zsre':
        from experiments.zsre_generated.core import generate_raw
        return generate_raw(model,tok,query,d['answer_cap'])
    from answer_only_generation import generate_answer_batch
    return generate_answer_batch(model,tok,[query],cap=d['answer_cap'])[0]


def read_bank(d,torch,model,names,n):
    bank={};disk=0;payload_bytes=0
    for row in d['factor_files'][:n]:
        path=Path(row['path'])
        if digest(path)!=row['sha256']:raise ValueError('Saved delta changed')
        value=torch.load(path,map_location='cpu',weights_only=True)
        if set(value)!=set(names):raise ValueError('Factor layer mismatch')
        if d['dataset']=='longmemeval':
            from qa500_core import canonical
            value=canonical(value)
        for name,(u,v) in value.items():
            shape=model.get_parameter(name).shape
            if u.dtype!=torch.float32 or v.dtype!=torch.float32 or not torch.isfinite(u).all() or not torch.isfinite(v).all():raise ValueError('Invalid saved factors')
            if (u.shape[0],v.shape[0])!=tuple(shape):raise ValueError('Factor shape mismatch')
            if u.ndim==2 and (v.ndim!=2 or u.shape[1]!=v.shape[1]):raise ValueError('Factor rank mismatch')
            payload_bytes+=u.numel()*u.element_size()+v.numel()*v.element_size()
        bank[row['sid']]=value;disk+=path.stat().st_size
    return bank,disk,payload_bytes


def scorer(d):
    if d['dataset']=='longmemeval':
        from qa500_core import score
        return score
    return None


def apply_payload(torch,model,payload):
    with torch.no_grad():
        for name,(u,v) in payload.items():
            if u.ndim==1:delta=u.to(model.device)[:,None]*v.to(model.device)[None,:]
            elif u.shape[1]==1:delta=u[:,0].to(model.device)[:,None]*v[:,0].to(model.device)[None,:]
            else:delta=u.to(model.device)@v.to(model.device).T
            model.get_parameter(name).add_(delta);del delta


def routing_summary(trials):
    valid=[t for t in trials if t['source_hit'] is not None]
    short=[t for t in trials if t['shortlist_hit'] is not None]
    return dict(accuracy=sum(t['source_hit'] for t in valid)/len(valid) if valid else None,
                correct=sum(t['source_hit'] for t in valid),questions=len(trials),routing_questions=len(valid),
                shortlist_recall=sum(t['shortlist_hit'] for t in short)/len(short) if short else None,
                shortlist_correct=sum(t['shortlist_hit'] for t in short) if short else None)
