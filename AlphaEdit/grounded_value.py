"""Grounded value optimization with unchanged causal prefixes cached once."""
import torch
from grounding_core import subject_last_index
from util import nethook


def compute_grounded_z(model, tok, request, hp, layer):
    if hp.fact_token != 'subject_last':
        raise ValueError('Grounded writes require subject_last')
    target_ids = tok(request['target_new']['str'], return_tensors='pt').input_ids[0].to('cuda')
    if target_ids[0] in [tok.bos_token_id, tok.unk_token_id]:
        target_ids = target_ids[1:]
    templates = [c.format(request['prompt'])+tok.decode(target_ids[:-1])
                 for group in request['value_context_templates'] for c in group]
    templates += ['{} is a']
    nethook.set_requires_grad(False, model)
    entries = []
    for template in templates:
        idx = subject_last_index(tok, template, request['subject'])
        x = tok(template.format(request['subject']), return_tensors='pt').input_ids.to('cuda')
        if idx < 1 or idx >= x.shape[1]:
            raise ValueError('Invalid grounded lookup index')
        with torch.no_grad():
            prefix = model.model(input_ids=x[:,:idx],use_cache=True)
        entries.append(dict(cache=prefix.past_key_values, prefix_length=idx, tail=x[:,idx:]))
        del prefix
    delta = torch.zeros(model.config.hidden_size,requires_grad=True,device='cuda')
    opt = torch.optim.Adam([delta],lr=hp.v_lr)
    initial = None
    kl_initial = None
    loss_layer = max(hp.v_loss_layer,layer)
    edit_name = hp.layer_module_tmp.format(layer)
    loss_name = hp.layer_module_tmp.format(loss_layer)
    norm = nethook.get_module(model,hp.ln_f_module)
    head = nethook.get_module(model,hp.lm_head_module)
    for step in range(hp.v_num_grad_steps):
        opt.zero_grad()
        nll_parts = []
        for i,entry in enumerate(entries):
            entry['cache'].crop(entry['prefix_length'])
            def edit(output, name):
                nonlocal initial
                if name == edit_name:
                    hidden = output[0]
                    if initial is None:
                        initial = hidden[0,0].detach().clone()
                    hidden[0,0] += delta
                return output
            tail = entry['tail']
            with nethook.TraceDict(model,layers=[edit_name,loss_name],retain_output=True,edit_output=edit) as tr:
                output = model.model(input_ids=tail, past_key_values=entry['cache'],use_cache=True,
                                     attention_mask=torch.ones((1,entry['prefix_length']+tail.shape[1]),device='cuda',dtype=torch.long))
            if i < len(entries)-1:
                hidden = tr[loss_name].output[0][0,-len(target_ids):]
                if hidden.shape[0] != len(target_ids):
                    raise ValueError('Target alignment mismatch')
                logp = head(norm(hidden)).float().log_softmax(-1)
                nll_parts.append(-logp.gather(-1,target_ids[:,None]).mean())
            else:
                kl_logp = head(output.last_hidden_state[0,0]).float().log_softmax(-1)
                if kl_initial is None:
                    kl_initial = kl_logp.detach().clone()
            # Drop appended graph-bearing KV entries; keep the immutable prefix.
            entry['cache'].crop(entry['prefix_length'])
            for cache_layer in range(len(entry['cache'].key_cache)):
                entry['cache'].key_cache[cache_layer] = entry['cache'].key_cache[cache_layer].detach()
                entry['cache'].value_cache[cache_layer] = entry['cache'].value_cache[cache_layer].detach()
        nll_each = torch.stack(nll_parts)
        nll = nll_each.mean()
        # Match the original KL batchmean reduction (one KL prompt).
        kl = hp.kl_factor*torch.nn.functional.kl_div(kl_initial[None,:],kl_logp[None,:],log_target=True,reduction='batchmean')
        decay = hp.v_weight_decay*delta.norm()/initial.norm()**2
        loss = nll+kl+decay
        print(f'grounded loss {step+1}: {loss.item():.4f}; nll={nll.item():.4f}; kl={kl.item():.4f}',flush=True)
        if loss < .05 or step == hp.v_num_grad_steps-1:
            break
        loss.backward()
        opt.step()
        maximum = hp.clamp_norm_factor*initial.norm()
        if delta.norm() > maximum:
            with torch.no_grad():
                delta.mul_(maximum/delta.norm())
    return initial+delta
