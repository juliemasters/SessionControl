"""Load pinned, unmodified WISE code and official metric function bodies."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'third_party/wise_paper'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def modules():
    upstream=json.loads((VENDOR/'UPSTREAM.json').read_text())
    for path,sha in upstream['files'].items():
        if digest(VENDOR/path)!=sha:raise ValueError('Official source changed: '+path)
    base=VENDOR/'easyeditor'
    for name,path in [('_paper_easyeditor',base),('_paper_easyeditor.models',base/'models'),
                      ('_paper_easyeditor.models.wise',base/'models/wise'),('_paper_easyeditor.util',base/'util')]:
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    importlib.import_module('transformers.models.gpt2.modeling_gpt2')
    return {name:importlib.import_module('_paper_easyeditor.models.wise.'+name) for name in ['WISE','utils','wise_main','wise_hparams']}


def evaluator():
    """Compile exact upstream functions, avoiding unrelated trainer/editor imports.

    Neither function body, token slicing, device convention nor scoring rule is
    changed. This is the released teacher-forcing metric, not generated-answer EM.
    """
    import numpy as np
    import torch
    path=VENDOR/'easyeditor/evaluate/evaluate_utils.py'
    source=path.read_text();tree=ast.parse(source)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['slice_list','test_prediction_acc']]
    if len(selected)!=2:raise ValueError('Missing official metric functions')
    module=ast.Module(body=selected,type_ignores=[])
    scope=dict(torch=torch,np=np)
    exec(compile(module,str(path),'exec'),scope)
    return scope['test_prediction_acc']


def restore(editor, path, torch):
    """Restore precisely the state written by native WISE.save.

    Explicit weights_only=False permits the trusted local native config/stats
    objects with PyTorch 2.6; no untrusted downloaded checkpoint is loaded here.
    """
    mods=modules();state=torch.load(path,map_location='cpu',weights_only=False)
    adapter=editor.get_adapter_layer();device=adapter.original_layer.weight.device
    adapter.load_state_dict(state['adapter_state_dict'])
    for name,value in state['additional_info'].items():
        if torch.is_tensor(value):value=value.to(device)
        elif name=='memory_weight':value=[w.to(device) for w in value]
        setattr(adapter,name,value)
    adapter.editing=False;adapter.training=False
    mods['WISE'].edit_history=state['edit_history']
    mods['WISE'].merge_group_edit_history=state['merge_group_edit_history']
    return adapter.editing_total_cnt
