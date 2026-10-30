"""Completion-gated audit and report of native WISE's paper-dataset reproduction."""
import argparse
import csv
import html
import json
from pathlib import Path
import statistics
from .prepare import save
from .native import digest
from .worker import check,now


def build(out):
    d=check(out);rows=[]
    pre=json.loads((out/'baseline.json').read_text())
    if len(pre['rows'])!=1000:raise ValueError('Incomplete unedited baseline')
    for mode in d['variants']:
        done=json.loads((out/mode/'completion.json').read_text())
        if not done.get('full_run') or done['edits']!=1000:raise ValueError('Incomplete sequential run')
        for n in d['checkpoints']:
            p=out/mode/f'checkpoint-{n}.json';cell=json.loads(p.read_text());trials=cell['rows']
            if not cell.get('completed') or len(trials)!=n or [t['case_id'] for t in trials]!=list(range(n)):
                raise ValueError('Incomplete checkpoint evaluation')
            for t in trials:
                old=pre['rows'][t['case_id']]['locality_tokens'];new=t['locality_tokens']
                expected=sum(a==b for a,b in zip(old,new))/len(old)
                if len(old)!=len(new) or expected!=t['locality_token_preservation']:raise ValueError('Locality summary mismatch')
            keys=['reliability_token_accuracy','generalization_token_accuracy','locality_token_preservation',
                  'reliability_all_tokens_correct','generalization_all_tokens_correct','locality_all_tokens_preserved']
            values={k:statistics.mean(t[k] for t in trials) for k in keys}
            for k,v in values.items():
                if abs(cell['summary'][k]-v)>1e-12:raise ValueError('Metric mean mismatch')
            rows.append(dict(variant=mode,n=n,questions=n,**values,
                average_token_metric=statistics.mean(values[k] for k in keys[:3]),
                peak_gpu_allocated_gib=cell['peak_gpu_allocated_bytes']/2**30))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'results.json',dict(completed=True,summary=rows,protocol=d,
        note='Official teacher-forced token metrics, not freely generated-answer F1. Single sequential training trajectory per variant.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    keys=['reliability_token_accuracy','generalization_token_accuracy','locality_token_preservation']
    for ax,key,title in zip(axes,keys,['Reliability','Paraphrase generalization','Locality preservation']):
        for mode,color,marker in [('merge','#0077B6','o'),('retrieve','#555555','s')]:
            selected=[r for r in rows if r['variant']==mode]
            ax.plot([r['n'] for r in selected],[100*r[key] for r in selected],color=color,marker=marker,label='Measured WISE-'+mode.title())
        refkey={'reliability_token_accuracy':'reliability','generalization_token_accuracy':'generalization','locality_token_preservation':'locality'}[key]
        ax.plot(d['checkpoints'],[100*d['paper_mistral_merge_reference'][str(n)][refkey] for n in d['checkpoints']],
            color='#999999',ls='--',label='Paper WISE-Merge reference')
        ax.set_xscale('log');ax.set_xticks(d['checkpoints'],labels=d['checkpoints']);ax.set_ylim(0,103)
        ax.set_xlabel('Sequential edits');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Official token metric (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Native WISE on the authors\u2019 released ZsRE edit set · Mistral-7B',fontweight='bold')
    fig.subplots_adjust(bottom=.23,top=.80,wspace=.25)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'paper_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    text=['# WISE paper-dataset reproduction','',
      'Completed two native 1,000-edit trajectories, WISE-Merge and WISE-Retrieve, on physical GPU 1. No LongMemEval examples, custom loss, threshold changes, forced-memory routing or session controller were used.',
      '', '**Metric distinction:** reliability/generalization are the official released teacher-forced target-token accuracies. Locality compares post-edit token predictions with those of the unedited model. These are not free-generated-answer F1 or whole-answer exact match.', '',
      '| Variant | Edits | Questions | Reliability | Generalization | Locality | Average |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| WISE-{r['variant'].title()} | {r['n']} | {r['questions']} | {100*r[keys[0]]:.2f}% | {100*r[keys[1]]:.2f}% | {100*r[keys[2]]:.2f}% | {100*r['average_token_metric']:.2f}% |")
    text+=['','## Faithfulness and limits','',
      '- Source: unmodified publication-era EasyEdit revision `'+d['source_revision']+'`; every vendored source file is hashed.',
      '- Model: `'+d['model_repository']+'` at revision `'+d['model_revision']+'`, FP32. This is a model family used in the paper, rather than Llama-3.1.',
      '- Data: authors\u2019 official data.zip, designated ZsRE edit subset. '+d['sample_policy'],
      '- Sequential singleton updates; official SGD, layer 27, random gradient masks, activation-margin loss, threshold statistics, prefix augmentation and merge/retrieval policies.',
      '- Ground-truth paraphrases and evaluation-locality questions are not used as edit targets. Locality training examples come from the separate released training split.',
      '- All first-T questions are evaluated after T edits. This tests retention, rather than averaging only immediately post-write results.',
      '- Paper-reference dashed lines are context, not a claim of exact sample-order/environment parity. The lower-T cells here have only T questions.',
      '- One training seed/trajectory per variant. No confidence across independent training seeds is inferred.',
      '', *['- '+x for x in d['known_differences']], '',
      'The original LongMemEval results remain separate and unchanged. This experiment establishes a native WISE benchmark reference; it does not replace or retroactively correct the earlier task.', '',
      'Full all-target-token indicators, counts, per-question predictions, source/model/data checksums, native checkpoints and progress logs are saved alongside this report. Figures are generated only after both full runs pass the audit.']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    svg=(out/'paper_comparison.svg').read_text();svg='<svg'+svg.split('<svg',1)[1]
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>WISE paper reproduction</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:20px}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style><h1>Native WISE · ZsRE reproduction</h1>'+svg+'<pre>'+html.escape('\n'.join(text))+'</pre>')
    artifacts=['summary.csv','results.json','report.md','report.html','paper_comparison.png','paper_comparison.svg','paper_comparison.pdf']
    save(out/'report_manifest.json',{name:digest(out/name) for name in artifacts})
    save(out/'completion.json',dict(completed=True,physical_gpu=1,variants=d['variants'],edits_per_variant=1000,
        verified_checkpoint_requests=sum(d['checkpoints'])*2,report='report.html',finished_utc=now()))
    print('FINAL_REPORT_READY',out/'report.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out)
