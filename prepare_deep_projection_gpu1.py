"""Build layer-specific empirical AlphaEdit null-space projectors on GPU 1.

SVD of token keys gives the same eigenspaces as their uncentered second moment,
without eigendecomposing a 14336-square matrix. This small corpus is a pilot.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zsre_routing_experiment import require_gpu1

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--layers',default='13,14,15')
    p.add_argument('--corpus',default='data/preservation_calibration.example.jsonl')
    p.add_argument('--threshold',type=float,default=.02)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    layers=[int(x) for x in a.layers.split(',')]
    raw=Path(a.corpus).read_bytes();records=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    texts=[x if isinstance(x,str) else x['text'] for x in records]
    if a.threshold<=0:raise ValueError('Positive threshold required')
    gpu=require_gpu1()
    import torch
    # Projector validation needs full float32 matmul precision, not TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained('models/Llama-3.1-8B-Instruct')
    model=AutoModelForCausalLM.from_pretrained('models/Llama-3.1-8B-Instruct',torch_dtype=torch.float32).to('cuda:0').eval()
    keys={layer:[] for layer in layers};handles=[]
    for layer in layers:
        def capture(module,args,layer=layer):keys[layer].append(args[0][0].detach().float().cpu())
        handles.append(model.get_submodule(f'model.layers.{layer}.mlp.down_proj').register_forward_pre_hook(capture))
    try:
        for text in texts:
            x=tok(text,return_tensors='pt',truncation=False).to('cuda:0')
            with torch.inference_mode():model(**x,use_cache=False)
    finally:
        for h in handles:h.remove()
    projections=[];diagnostics=[]
    for layer in layers:
        matrix=torch.cat(keys[layer]).to('cuda:0');n,d=matrix.shape
        # C = X.T X / n; eigenvalues = singular_values(X)^2 / n.
        _,singular,vh=torch.linalg.svd(matrix,full_matrices=False)
        eigenvalues=singular.square()/n
        protected=vh[eigenvalues>=a.threshold].T
        protected=torch.linalg.qr(protected,mode='reduced').Q
        projection=torch.eye(d,device='cuda:0')-protected@protected.T
        # Small-vector numerical checks avoid another dense matrix product.
        residual=(projection@protected).norm().item()
        diagnostics.append(dict(layer=layer,texts=len(texts),tokens=n,protected_rank=protected.shape[1],
            null_dimension=d-protected.shape[1],protected_residual_norm=residual))
        print('PROJECTION',diagnostics[-1],flush=True)
        if residual>1e-3:raise ValueError('Projector residual too large')
        projections.append(projection.cpu());del matrix,singular,vh,protected,projection
        torch.cuda.empty_cache()
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(torch.stack(projections),out)
    out.with_suffix('.json').write_text(json.dumps(dict(physical_gpu=1,gpu_uuid=gpu,layers=layers,
        corpus=a.corpus,corpus_sha256=hashlib.sha256(raw).hexdigest(),threshold=a.threshold,
        method='Null space of empirical uncentered token second moment via thin SVD',
        limitation='Small calibration corpus, not the full AlphaEdit preservation benchmark',diagnostics=diagnostics),indent=2))
    print('COMPLETED projection',out,flush=True)

if __name__=='__main__':main()
