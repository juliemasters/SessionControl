"""Frozen inputs, official WISE loading, complete-answer writes and timing metrics."""
from datetime import datetime,timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import threading
import types

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
VENDOR=ROOT/'third_party/wise'
SIZES=(8,16,32,64,128,256,500)


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def hashed(text):return hashlib.sha256(text.encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def status(out,stage,**kwargs):save(out/'status.json',dict(stage=stage,updated_utc=now(),**kwargs))
def order(qids,repeat,n):return sorted(qids,key=lambda q:hashed(f'{repeat}/{n}/{q}'))


def vendor_modules():
    """Import only unmodified WISE modules, without EasyEdit's unrelated editors."""
    upstream=read(VENDOR/'UPSTREAM.json')
    for f,h in upstream['files'].items():
        if digest(VENDOR/f)!=h:raise ValueError('Pinned official WISE source changed: '+f)
    for name,path in [('_sc_wise',VENDOR),('_sc_wise.models',VENDOR/'models'),
                      ('_sc_wise.models.wise',VENDOR/'models/wise'),('_sc_wise.util',VENDOR/'util')]:
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    importlib.import_module('transformers.models.gpt2.modeling_gpt2')
    return (importlib.import_module('_sc_wise.models.wise.WISE'),
            importlib.import_module('_sc_wise.models.wise.wise_hparams'))


def editor_class():
    official,_=vendor_modules()
    import torch
    class EfficientWISE(official.WISE):
        """Same masked token CE and gradients, projecting supervised positions only.

        Upstream constructs vocabulary logits and an unused model loss for every
        context/padding token. Excluding -100 positions before the LM head avoids
        that allocation. Upstream activation loss, SGD, masks and router stay intact.
        """
        def edit(self,*args,**kwargs):
            self._frozen_prefix=None
            try:return super().edit(*args,**kwargs)
            finally:self._frozen_prefix=None

        def _hidden(self,tokens):
            # All decoder layers before WISE's edited layer are frozen and in
            # eval mode. For one fixed write batch, their output is identical
            # across optimizer steps. Cache that output, never a post-edit KV.
            layer_index=int(self.layer.split('.')[2].strip('[]')) if '[' not in self.layer else int(self.layer.split('[')[1].split(']')[0])
            layers=self.model.model.layers
            cache=getattr(self,'_frozen_prefix',None)
            signature=tuple((tokens[k].data_ptr(),tuple(tokens[k].shape)) for k in ['input_ids','attention_mask'])
            if layer_index==0:
                return self.model.model(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask'],use_cache=False)
            if cache is None:
                def capture(module,args,output):self._frozen_prefix=(signature,output[0].detach().clone())
                handle=layers[layer_index-1].register_forward_hook(capture)
                try:return self.model.model(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask'],use_cache=False)
                finally:handle.remove()
            if cache[0]!=signature:raise ValueError('Frozen-prefix cache reused for a different write batch')
            originals=[]
            def unchanged(hidden_states,*args,**kwargs):return (hidden_states,)
            def boundary(hidden_states,*args,**kwargs):return (cache[1],)
            try:
                for i,layer in enumerate(layers[:layer_index]):
                    originals.append((layer,layer.forward));layer.forward=boundary if i==layer_index-1 else unchanged
                return self.model.model(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask'],use_cache=False)
            finally:
                for layer,forward in originals:layer.forward=forward

        def _cal_ft_loss(self,tokens):
            k=getattr(self.model.config,'batch_size',1)
            result=self._hidden(tokens)
            hidden=result.last_hidden_state[:-k,:-1,:]
            labels=tokens['labels'][:-k,1:];mask=labels!=-100
            indices=mask.nonzero(as_tuple=False)
            if not len(indices):raise ValueError('Missing complete-answer labels')
            logits=self.model.lm_head(hidden[mask]).float()
            values=torch.nn.functional.cross_entropy(logits,labels[mask],reduction='none')
            losses=torch.zeros(labels.shape[0],device=values.device,dtype=values.dtype)
            losses.scatter_add_(0,indices[:,0],values)
            loss=(losses/mask.sum(1).clamp_min(1)).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite WISE answer loss')
            self.last_ft_loss=float(loss.detach());return loss
    return EfficientWISE


def prepare(out,data,model,reference):
    if out.exists():raise FileExistsError('Use a fresh WISE run directory')
    reference=Path(reference).resolve();data=Path(data).resolve();model=Path(model).resolve()
    ref=read(reference/'design.json');done=read(reference/'completion.json')
    if not done.get('completed') or done['passes']!=10:raise ValueError('Need the completed ten-pass reference experiment')
    if digest(data)!=ref['dataset_sha256']:raise ValueError('Official dataset differs from the reference experiment')
    if model!=Path(ref['model']).resolve():raise ValueError('Use exactly the same local base model for this comparison')
    if shutil.disk_usage(out.parent).free<8*2**30:raise RuntimeError('Need 8 GiB free including safety margin')
    official=read(data);records={};queries=[]
    seed=read(HERE/'config/seed_session_ids.json');seed_set=set(seed)
    all_support={sid for row in official for sid in row['answer_session_ids']}
    locality={}
    for row in official:
        qid=str(row['question_id']);sessions=dict(zip(row['haystack_session_ids'],row['haystack_sessions']))
        dates=dict(zip(row['haystack_session_ids'],row['haystack_dates']))
        queries.append(dict(qid=qid,question=row['question'],answer=str(row['answer']),question_date=row['question_date'],
            question_type=row['question_type'],support_sids=row['answer_session_ids'],abstention=qid.endswith('_abs'),
            original64=bool(set(row['answer_session_ids'])&seed_set)))
        for sid in row['answer_session_ids']:
            r=records.setdefault(sid,dict(sid=sid,context=sessions[sid],bindings=[]))
            if r['context']!=sessions[sid]:raise ValueError('Conflicting official conversation')
            r['bindings'].append(dict(qid=qid,question=row['question'],answer=str(row['answer']),date=dates[sid]))
        for sid,session in sessions.items():
            if sid in all_support:continue
            for turn_index,turn in enumerate(session):
                text=turn.get('content','')
                if turn.get('role')=='user' and isinstance(text,str) and text.strip():
                    locality.setdefault(sid+':'+str(turn_index),text)
    bank_ids=list(seed);seen=set(seed)
    for q in sorted(queries,key=lambda q:hashed(q['qid'])):
        for sid in q['support_sids']:
            if sid not in seen:bank_ids.append(sid);seen.add(sid)
    bank=[records[sid] for sid in bank_ids[:500]]
    if bank!=read(reference/'sessions.json'):raise ValueError('The 500-conversation bank differs from the completed experiment')
    if queries!=read(reference/'queries.json'):raise ValueError('Official evaluation questions differ from the completed experiment')
    plan=[]
    for n in SIZES:
        present=set(bank_ids[:n]);ids=[q['qid'] for q in queries if set(q['support_sids'])<=present]
        plan.append(dict(n=n,qids=ids,questions=len(ids),bindings=sum(len(r['bindings']) for r in bank[:n])))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    loc=[]
    for key in sorted(locality,key=hashed)[:200]:
        # Cropping applies only to auxiliary locality data, never the stored session or answer.
        text=tok.decode(tok.encode(locality[key],add_special_tokens=False)[:128],clean_up_tokenization_spaces=False)
        loc.append(dict(id=key,text=text))
    if len(loc)!=200:raise ValueError('Require 200 disjoint official distractor turns for locality')
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'reference').mkdir()
    save(out/'sessions.json',bank);save(out/'queries.json',queries);save(out/'locality.json',loc)
    manifest=read(reference/'report_manifest.json')
    for name in ['comparison.json','comparison.csv']:
        if digest(reference/name)!=manifest[name]:raise ValueError('Reference metrics changed')
        shutil.copy2(reference/name,out/'reference'/name)
    d=dict(name='WISE-Retrieve matched full-Q/A comparison',model=str(model),dataset=str(data),dataset_sha256=digest(data),
        physical_gpu=1,precision='float32',sizes=list(SIZES),plan=plan,repeats=10,generation_cap=128,
        source_reference=str(reference),reference_finished_utc=done['finished_utc'],upstream=read(VENDOR/'UPSTREAM.json'),
        official_hparams_sha256=digest(HERE/'config/official-llama3.1-8b.yaml'),seed=0,
        writer='Official WISE-Retrieve sequential edits; layer 29, learning rate 0.9, 30 iterations, save_freq=500, merge_freq=1000, replay=False.',
        inputs='Each binding receives the complete conversation, literal question and complete official answer plus EOS. '
               'One question-only and one fully grounded template; one deterministic auxiliary distractor locality row. No session or answer truncation.',
        adaptation='Disable upstream chat wrapping to keep the exact Q: question / A: recall key used by the comparison. '
                   'Custom token assembly retains every answer token. An integration subclass projects only supervised hidden states through the LM head; '
                   'masked CE reduction and its gradients are tested against the original implementation. Within one write batch, cache only the unchanged '
                   'frozen decoder prefix before layer 29; recompute the edited layer and all downstream layers at every optimizer step. '
                   'Official SGD, activation loss, masks and router unchanged.',
        boundaries='Checkpoints count conversations, not individual bindings. 500 conversations contain 508 bindings; official save/merge frequencies count bindings. '
                   'No frequency tuning against evaluation answers. Native WISE routing is used, without SessionControl routing.',
        evaluation='Ten measured inference passes on one sequentially trained WISE trajectory. Same eligible questions and per-pass order as all baselines. '
                   'Whole final-answer token F1 and normalized exact match only. No answer extraction or source-routing accuracy.',
        timing='Same FirstAnswerClock and greedy decoding as the completed experiment; query arrival to first nonempty final output. '
               'Includes WISE routing in every relevant forward. Excludes model loading, checkpoint installation and writing. '
               'Baselines are the prior completed ten-pass run, not simultaneous interleaved measurements.',
        memory='Separate deployable side-memory checkpoint bytes, process host RSS, and GPU allocated/reserved peaks including common model weights. '
                'Unchanged original weights are reconstructed from the common base model. Training-only mask/RNG/history state excluded from deployed bank bytes.',
        limitations='Supervised exact-question mechanism check, not a blind LongMemEval benchmark. Full global reference answer is supplied for each supporting session. '
                    'Eligible cohorts change with N. Precision and data match, but each method retains its own layer/optimizer/router configuration.',prepared_utc=now())
    save(out/'design.json',d);save(out/'ledger.json',[])
    save(out/'input_manifest.json',{name:digest(out/name) for name in ['design.json','sessions.json','queries.json','locality.json','reference/comparison.json','reference/comparison.csv']})
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1))
    print('Prepared:',[(p['n'],p['questions'],p['bindings']) for p in plan],flush=True)


def check(out):
    for name,h in read(out/'input_manifest.json').items():
        if digest(out/name)!=h:raise ValueError('Frozen WISE input changed: '+name)
    return read(out/'design.json')


def load(out,folder):
    d=check(out)
    from experiment_utils import require_gpu1
    uuid=require_gpu1()
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    import numpy as np
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if torch.cuda.device_count()!=1:raise RuntimeError('Physical GPU 1 must be the only visible device')
    torch.set_num_threads(8);torch.manual_seed(0);torch.cuda.manual_seed_all(0);random.seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    folder.mkdir(parents=True,exist_ok=True)
    files=[HERE/'core.py',HERE/'worker.py',HERE/'config/official-llama3.1-8b.yaml',VENDOR/'UPSTREAM.json']
    files += [VENDOR/f for f in d['upstream']['files']]
    files += [ROOT/f for f in ['answer_only_generation.py','question_key.py','ttft_timing.py','memory_comparison_core.py','experiment_utils.py']]
    hashes={str(f.relative_to(ROOT)):digest(f) for f in files}
    if (folder/'provenance.json').exists():
        if read(folder/'provenance.json')['code_sha256']!=hashes:raise ValueError('WISE worker source changed; refuse mixed resume')
    else:
        for f in files:
            dest=folder/'code_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        save(folder/'provenance.json',dict(code_sha256=hashes,physical_gpu=1,gpu_uuid=uuid,torch_version=torch.__version__,started_utc=now()))
    _,hparams=vendor_modules()
    hp=hparams.WISEHyperParams.from_hparams(str(HERE/'config/official-llama3.1-8b.yaml'))
    hp.model_name=str(d['model']);hp.device=0;hp.use_chat_template=False;hp.sequential_edit=True
    tok=AutoTokenizer.from_pretrained(d['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='left'
    status(folder,'loading_model')
    model=AutoModelForCausalLM.from_pretrained(d['model'],torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
    editor=editor_class()(hp,model,torch.device('cuda:0'))
    return d,torch,tok,editor


def tokens_for_binding(tok,record,binding,locality,device):
    import torch
    from question_key import prompt
    key=prompt(binding['question'])
    context=json.dumps(record['context'],ensure_ascii=False)
    prefixes=[key,'Grounding conversation (data):\n'+context+'\nAnswer using this conversation.\n'+key]
    target=tok.encode(' '+binding['answer']+tok.eos_token,add_special_tokens=False)
    sequences=[];labels=[]
    for text in prefixes:
        prefix=tok.encode(text)
        sequences.append(prefix+target);labels.append([-100]*len(prefix)+target)
    sequences.append(tok.encode(locality));labels.append([-100]*len(sequences[-1]))
    maximum=max(map(len,sequences));ids=[];masks=[];targets=[]
    for x,y in zip(sequences,labels):
        pad=maximum-len(x);ids.append([tok.pad_token_id]*pad+x);masks.append([0]*pad+[1]*len(x));targets.append([-100]*pad+y)
    values={name:torch.tensor(data,device=device,dtype=torch.long) for name,data in [('input_ids',ids),('attention_mask',masks),('labels',targets)]}
    return values,[None],[None]


def atomic_torch(torch,path,state):
    path=Path(path)
    if shutil.disk_usage(path.parent).free<2*2**30+sum(t.numel()*t.element_size() for t in tensor_leaves(state)):
        raise RuntimeError('Checkpoint would violate the 2 GiB reserve')
    temp=path.with_suffix('.pt.tmp')
    with temp.open('wb') as handle:torch.save(state,handle);handle.flush();os.fsync(handle.fileno())
    temp.replace(path)


def tensor_leaves(value):
    if hasattr(value,'numel'):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from tensor_leaves(v)
    elif isinstance(value,(list,tuple)):
        for v in value:yield from tensor_leaves(v)


def export_state(editor,count):
    a=editor.get_adapter_layer()
    if not editor.config.retrieve or editor.config.replay:raise ValueError('Checkpoint codec is for native retrieve, no replay')
    if not a.layer.weight.equal(a.original_layer.weight):raise ValueError('Unexpected main-memory mutation')
    def activation(m):return dict(avg=m.avg,count=m.count,sum=m.sum,min_a=m.min_a)
    return dict(conversations=count,bindings=a.editing_total_cnt,new_weight=a.new_weight.detach().cpu().clone(),
                memory_weight=[w.detach().cpu().clone() for w in a.memory_weight],
                editing_mean_act=activation(a.editing_mean_act),memory_mean_act=[activation(m) for m in a.memory_mean_act],merge_cnt=a.merge_cnt)


def install_state(editor,state):
    import torch
    official,_=vendor_modules();a=editor.get_adapter_layer();device=a.original_layer.weight.device
    if not editor.config.retrieve:raise ValueError('Unsupported checkpoint codec for WISE-Merge')
    a.new_weight=torch.nn.Parameter(state['new_weight'].to(device),requires_grad=False)
    a.memory_weight=[w.to(device) for w in state['memory_weight']]
    def activation(data):
        m=official.EditingMeanAct()
        for k,v in data.items():setattr(m,k,v)
        return m
    a.editing_mean_act=activation(state['editing_mean_act'])
    a.memory_mean_act=[activation(data) for data in state['memory_mean_act']]
    a.editing_total_cnt=state['bindings'];a.merge_cnt=state['merge_cnt'];a.editing=False;a.training=False
    for name in ['original_layer_output','new_weight_layer_output']:
        if hasattr(a,name):delattr(a,name)


def rng_state(torch):
    import numpy as np
    n=np.random.get_state()
    return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]])


def restore_rng(torch,state):
    import numpy as np
    random.setstate(state['python']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))


def summary(trials):
    def mean(key):
        vals=[t[key] for t in trials if t[key] is not None]
        return statistics.mean(vals) if vals else None
    visible=sorted(t['ttft_seconds']*1000 for t in trials if t['ttft_seconds'] is not None)
    return dict(questions=len(trials),mean_f1=mean('token_f1'),exact_rate=mean('exact'),
                ttft_mean_ms=statistics.mean(visible) if visible else None,
                ttft_median_ms=statistics.median(visible) if visible else None,
                ttft_p95_ms=visible[math.ceil(.95*len(visible))-1] if visible else None,
                first_generated_mean_ms=1000*mean('first_generated_token_seconds') if mean('first_generated_token_seconds') is not None else None,
                query_latency_mean_ms=1000*mean('query_seconds'),output_tokens_mean=mean('output_tokens'),
                hit_cap_rate=mean('hit_cap'),visible_ttft_samples=len(visible),missing_visible_ttft=len(trials)-len(visible))


class HostPeak:
    def __enter__(self):
        def rss():return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        self.rss=rss;self.before=rss();self.peak=self.before;self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.02):self.peak=max(self.peak,rss())
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.after=self.rss();self.peak=max(self.peak,self.after);self.stop.set();self.thread.join()
    def result(self):return dict(host_rss_before_bytes=self.before,host_peak_rss_bytes=self.peak,host_rss_after_bytes=self.after,host_sampling_ms=20)
