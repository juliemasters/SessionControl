"""Freeze the authors' released data and reproduction protocol, without a GPU."""
import argparse
from pathlib import Path
import json
import hashlib
from datetime import datetime,timezone
from .native import ROOT,VENDOR,digest,modules


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def prepare(out):
    if out.exists():raise FileExistsError('Use a new paper-reproduction output directory')
    data=Path('/home/julie/alphaeditSessionControl/data/wise-paper-official')
    edit_file=data/'data/ZsRE/zsre_mend_edit.json';train_file=data/'data/ZsRE/zsre_mend_train.json'
    edits=json.loads(edit_file.read_text());train=json.loads(train_file.read_text())
    assert len(edits)==1037 and len(train)==163196
    requests=[]
    for i,(e,loc) in enumerate(zip(edits[:1000],train[:1000])):
        assert all(isinstance(e[k],str) and e[k].strip() for k in ['src','alt','rephrase','loc','loc_ans','subject'])
        assert e['subject'] in e['src'],(i,e['subject'],e['src'])
        requests.append(dict(case_id=i,prompt=e['src'],target_new=e['alt'],subject=e['subject'],
            rephrase_prompt=e['rephrase'],ground_truth='<|endoftext|>',
            loc_prompt=loc['loc']+' '+loc['loc_ans'],
            locality=dict(prompt=e['loc'],ground_truth=e['loc_ans'])))
    out.mkdir(parents=True);save(out/'requests.json',requests)
    source=json.loads((VENDOR/'UPSTREAM.json').read_text())
    code={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'experiments/wise_paper').glob('*.py')}
    hp=modules()['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
    design=dict(name='Native WISE reproduction on the authors\u2019 ZsRE release',
       paper='https://arxiv.org/abs/2405.14768',source_revision=source['revision'],
       dataset_source='https://drive.google.com/file/d/1YtQvv4WvTa4rJyDYQR2J-uK8rnrt0kTA/view',
       dataset_files={str(p):digest(p) for p in [data/'data.zip',edit_file,train_file]},
       requests_sha256=digest(out/'requests.json'),data_edit_count=1037,data_train_count=163196,
       selected_count=1000,sample_policy='First 1000 ordered records from the authors\u2019 designated edit file, paired with first 1000 training locality examples, as in released run_wise_editing.py. Exact paper random sample IDs are undisclosed.',
       model='/home/julie/alphaeditSessionControl/models/Mistral-7B-v0.1',
       model_repository='mistralai/Mistral-7B-v0.1',model_revision='27d67f1b5f57dc0953326b2601d68371d40ea8da',
       physical_gpu=1,seed=42,precision='float32',checkpoints=[1,10,100,1000],variants=['merge','retrieve'],
       hyperparameters=hp.__dict__,code_sha256=code,
       algorithm='Unmodified publication-era WISE.py, utils.py, wise_main.py, merging code, and official context-template generation. No custom training subclass, loss, masks, threshold recalibration, forced memory, or session routing.',
       metrics='Official released teacher-forced token accuracy for reliability/paraphrase generalization and token preservation against the unedited model for locality. Also report all-target-tokens-correct indicators. These are not free-generated-answer F1 or EM.',
       checkpoint_protocol='One continuous 1000-edit trajectory per variant. At each checkpoint, evaluate all first-T edits after T writes. Lower-T cells have T samples, not an undisclosed aggregation over independent edit sequences.',
       merge_protocol='Paper main tables use WISE-Merge (retrieve=False); separately run documented WISE-Retrieve (retrieve=True). All other official YAML values unchanged except local model path and logical device 0 mapped to physical GPU 1.',
       known_differences=['Exact paper subset/order and execution commit are not supplied; use released edit subset and an explicitly pinned publication-era revision.',
          'Official get_context_templates uses five length-5 and five length-10 generated prefixes, whereas paper describes ten length-10 prefixes; follow released code unchanged.',
          'Released metrics average teacher-forced target-token correctness; paper equations describe binary correctness. Report both token-average and all-token indicators, not interchangeable with whole generated-answer accuracy.',
          'This run uses PyTorch 2.6.0/Transformers 4.51.3 on A100, rather than claiming exact original environment/A800 parity.',
          'Only the paper\u2019s main ZsRE QA benchmark is reproduced here. Hallucination and Temporal are downloaded but not silently included.'],
       paper_mistral_merge_reference={str(k):dict(reliability=r,generalization=g,locality=1.) for k,r,g in [(1,.98,.97),(10,.92,.89),(100,.87,.80),(1000,.70,.67)]},
       prepared_utc=datetime.now(timezone.utc).isoformat())
    save(out/'design.json',design);save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1))
    for mode in design['variants']:(out/mode).mkdir()
    print('PREPARED',out,len(requests),design['checkpoints'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.out)
