"""Explicit approximate rank-one storage and a one-key AlphaEdit writer.

Legacy rounded dense deltas are approximated, never described as losslessly
compressed. New independent writes use the Sherman-Morrison solution for
lambda*I + (P k) k.T, valid only for one key and a zero preservation accumulator.
The value optimization, grounded session, layers and EOS target are unchanged.
"""
from pathlib import Path


def fit_rank_one(matrix, *, iterations=8):
    import torch
    with torch.inference_mode():
        right = torch.ones(matrix.shape[1], device=matrix.device)
        right /= right.norm()
        for _ in range(iterations):
            left = matrix @ right
            if left.norm().item()==0:
                raise ValueError('Zero delta')
            left /= left.norm()
            right = matrix.T @ left
            right /= right.norm()
        left = matrix @ right
        residual = (matrix-left[:, None]*right[None, :]).double().square().sum()
        norm = matrix.double().square().sum()
        return left.cpu(), right.cpu(), (residual/norm).sqrt().item()


def one_key_factors(projector, key, residual, l2):
    import torch
    if key.ndim!=1 or residual.ndim!=1 or l2<=0:
        raise ValueError('Only one key, one residual and positive L2 supported')
    projected = projector @ key
    denominator = l2 + key @ projected
    if not torch.isfinite(denominator) or denominator.item()<=0:
        raise ValueError('Invalid Sherman-Morrison denominator')
    return residual.detach(), (projected/denominator).detach()


class CompactSessionMemory:
    def __init__(self, model, names, root):
        self.model=model; self.names=tuple(names); self.root=Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base={n:model.get_parameter(n).detach().cpu().clone() for n in names}

    def restore(self):
        import torch
        with torch.no_grad():
            for name, value in self.base.items():
                self.model.get_parameter(name).copy_(value.to(self.model.device))

    def load(self, sid):
        import torch
        payload=torch.load(self.root/(sid+'.pt'),map_location='cpu',weights_only=True)
        if set(payload)!=set(self.names):raise ValueError('Unexpected factor layers')
        for name,(left,right) in payload.items():
            shape=self.model.get_parameter(name).shape
            if left.shape!=(shape[0],) or right.shape!=(shape[1],):
                raise ValueError('Factor shape mismatch')
            if left.dtype!=torch.float32 or right.dtype!=torch.float32 or not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError('Require finite float32 factors')
        return payload

    def save(self, sid, payload):
        import torch
        path=self.root/(sid+'.pt')
        if path.exists():raise ValueError('Never overwrite saved factors')
        torch.save({n:(l.detach().cpu().contiguous(),r.detach().cpu().contiguous()) for n,(l,r) in payload.items()},path)

    def apply(self, payload):
        import torch
        with torch.no_grad():
            for name,(left,right) in payload.items():
                weight=self.model.get_parameter(name)
                weight.add_(left.to(weight.device)[:,None]*right.to(weight.device)[None,:])

    def run(self, sid, inference):
        self.restore()
        try:
            self.apply(self.load(sid))
            return inference(self.model)
        finally:
            self.restore()

    def write(self, sid, tok, request, hp, projector):
        import torch
        from copy import deepcopy
        from AlphaEdit.compute_z import compute_z, get_module_input_output_at_words
        from AlphaEdit.compute_ks import compute_ks
        request=deepcopy(request)
        request['target_new']['str']=' '+request['target_new']['str'].lstrip()
        self.restore(); factors={}
        try:
            target=compute_z(self.model,tok,request,hp,hp.layers[-1],[['{}']])
            for i,layer in enumerate(hp.layers):
                keys=compute_ks(self.model,tok,[request],hp,layer,[['{}']]).T
                if keys.shape[1]!=1:raise ValueError('Compact writer requires exactly one key')
                current=get_module_input_output_at_words(self.model,tok,hp.layers[-1],
                    context_templates=[request['prompt']],words=[request['subject']],
                    module_template=hp.layer_module_tmp,fact_token_strategy=hp.fact_token)[1].T
                residual=(target-current[:,0])/(len(hp.layers)-i)
                left,right=one_key_factors(projector[i].to(self.model.device),keys[:,0],residual,hp.L2)
                name=hp.rewrite_module_tmp.format(layer)+'.weight'
                factors[name]=(left.cpu(),right.cpu())
                self.apply({name:(left,right)})
            self.save(sid,factors)
        finally:
            self.restore()


def capture_keys(model, tok, names, questions):
    import torch
    from question_key import prompt
    current={};handles=[]
    for name in names:
        def hook(module,args,output,name=name):current[name]=args[0][:,-1].detach().clone()
        handles.append(model.get_submodule(name.removesuffix('.weight')).register_forward_hook(hook))
    try:
        tok.padding_side='left'
        encoded=tok([prompt(q) for q in questions],padding=True,return_tensors='pt',truncation=False).to(model.device)
        positions=encoded.attention_mask.cumsum(-1)-1
        positions.masked_fill_(encoded.attention_mask==0,0)
        with torch.inference_mode():model.model(**encoded,position_ids=positions,use_cache=False)
        return current
    finally:
        for handle in handles:handle.remove()


def factor_scores(payload, keys):
    import torch
    numerator=torch.zeros(next(iter(keys.values())).shape[0],dtype=torch.float64,device=next(iter(keys.values())).device)
    denominator=torch.zeros((),dtype=torch.float64,device=numerator.device)
    with torch.inference_mode():
        for name,(left,right) in payload.items():
            left=left.to(numerator.device);right=right.to(numerator.device)
            lnorm=left.double().square().sum();rnorm=right.double().square().sum()
            numerator+=(keys[name]@right).double().square()*lnorm
            denominator+=lnorm*rnorm
    if denominator.item()<=0:raise ValueError('Empty compact delta')
    return (numerator/denominator).cpu().tolist()
