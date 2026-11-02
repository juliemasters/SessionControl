"""Audit complete matched metrics and compare with the two frozen WISE runs."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
from experiments.wise_paper.native import digest
from experiments.wise_paper.prepare import save
from .prepare import check
from .worker import now


METRICS=['reliability_token_accuracy','generalization_token_accuracy','locality_token_preservation',
         'reliability_all_tokens_correct','generalization_all_tokens_correct','locality_all_tokens_preserved']


def build(out):
    d=check(out);ref=Path(d['reference_run'])
    done=json.loads((out/'worker_completion.json').read_text())
    if not done.get('full_run') or done['edits']!=1000:raise ValueError('Full requested bank is not complete')
    ledger=json.loads((out/'ledger.json').read_text())
    if [r['case_id'] for r in ledger]!=list(range(1000)):raise ValueError('Incomplete write ledger')
    for entry in ledger:
        if digest(out/'factors'/(entry['sid']+'.pt'))!=entry['sha256']:raise ValueError('Delta checksum failed')
    baseline=json.loads((ref/'baseline.json').read_text())['rows']
    rows=[]
    for n in d['checkpoints']:
        cell=json.loads((out/f'checkpoint-{n}.json').read_text());trials=cell['rows']
        if not cell.get('completed') or [r['case_id'] for r in trials]!=list(range(n)):
            raise ValueError('Incomplete checkpoint')
        if cell['bank_sha256']!={r['sid']:r['sha256'] for r in ledger[:n]}:raise ValueError('Checkpoint bank mismatch')
        for trial in trials:
            old=baseline[trial['case_id']]['locality_tokens'];new=trial['locality_tokens']
            if len(old)!=len(new) or trial['locality_token_preservation']!=sum(a==b for a,b in zip(old,new))/len(old):
                raise ValueError('Locality mismatch')
        for key in METRICS+['rewrite_routing_hit','paraphrase_routing_hit','rewrite_correct_delta_rank','paraphrase_correct_delta_rank']:
            if abs(statistics.mean(t[key] for t in trials)-cell['summary'][key])>1e-12:raise ValueError('Metric mean mismatch')
        rows.append(dict(variant='session_control',n=n,questions=n,**cell['summary'],
            peak_gpu_allocated_gib=cell['peak_gpu_allocated_bytes']/2**30,retained_bank_mib=cell['retained_bank_bytes']/2**20))
        for mode in ['merge','retrieve']:
            native=json.loads((ref/mode/f'checkpoint-{n}.json').read_text())
            if not native.get('completed') or [t['case_id'] for t in native['rows']]!=list(range(n)):raise ValueError('WISE checkpoint missing')
            rows.append(dict(variant='wise_'+mode,n=n,questions=n,**native['summary'],
                peak_gpu_allocated_gib=native['peak_gpu_allocated_bytes']/2**30,retained_bank_mib=None))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    save(out/'results.json',dict(completed=True,protocol=d,summary=rows,
        note='Same native teacher-forced metrics and all-first-T evaluation cohort. Not free-generated-answer F1 or TTFT.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    names={'session_control':'Our activation-routed deltas','wise_merge':'WISE-Merge','wise_retrieve':'WISE-Retrieve'}
    styles={'session_control':('#0077B6','o','-'),'wise_merge':('#444444','s','-'),'wise_retrieve':('#888888','^','--')}
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,key,title in zip(axes,METRICS[:3],['Reliability','Paraphrase generalization','Locality preservation']):
        for variant,(color,marker,line) in styles.items():
            selected=[r for r in rows if r['variant']==variant]
            ax.plot([r['n'] for r in selected],[100*r[key] for r in selected],color=color,marker=marker,ls=line,label=names[variant])
        ax.set_xscale('log');ax.set_xticks(d['checkpoints'],labels=d['checkpoints']);ax.set_ylim(0,103)
        ax.set_xlabel('Stored edits');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Official token metric (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Matched ZsRE · Mistral-7B · one fixed trajectory per method',fontweight='bold')
    fig.subplots_adjust(bottom=.23,top=.80,wspace=.25)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    ours=[r for r in rows if r['variant']=='session_control']
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for key,label,marker in [('rewrite_routing_hit','Original question','o'),('paraphrase_routing_hit','Held-out paraphrase','s')]:
        axes[0].plot([r['n'] for r in ours],[100*r[key] for r in ours],marker=marker,label=label)
    axes[0].set_ylim(0,103);axes[0].set_ylabel('Correct delta selected (%)');axes[0].legend(frameon=False)
    axes[1].plot([r['n'] for r in ours],[r['retained_bank_mib'] for r in ours],color='#0077B6',marker='o')
    axes[1].set_ylabel('Compact retained bank (MiB)')
    for ax in axes:ax.set_xscale('log');ax.set_xticks(d['checkpoints'],labels=d['checkpoints']);ax.set_xlabel('Stored edits');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Our method · source selection and retained delta storage',fontweight='bold');fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'routing_storage.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    text=['# Matched ZsRE: session-controlled AlphaEdit versus native WISE','',
          'Completed 1,000 independent compact writes and activation routing on physical GPU 1. '
          'Compared with the two previously completed, frozen native WISE trajectories. '
          'All methods use the same 1,000 ZsRE records, Mistral-7B revision, FP32 and checkpoint cohorts.',
          '', '**Primary metrics are teacher-forced target-token accuracy and locality token preservation. '
          'They are not free-generated-answer F1, exact match, semantic correctness or TTFT.**','',
          '| Method | Edits | Questions | Reliability | Generalization | Locality | Average |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {names[r['variant']]} | {r['n']} | {r['questions']} | {100*r[METRICS[0]]:.2f}% | {100*r[METRICS[1]]:.2f}% | {100*r[METRICS[2]]:.2f}% | {100*r['average_token_metric']:.2f}% |")
    text+=['','## All target tokens correct (still teacher-forced)','',
           '| Method | Edits | Original question | Paraphrase | All locality tokens preserved |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {names[r['variant']]} | {r['n']} | {100*r[METRICS[3]]:.2f}% | {100*r[METRICS[4]]:.2f}% | {100*r[METRICS[5]]:.2f}% |")
    text+=['','## Our routing and memory','',
           '| Edits | Original routing hit | Paraphrase routing hit | Original mean rank | Paraphrase mean rank | Bank MiB | Evaluation peak GPU GiB |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for r in ours:
        text.append(f"| {r['n']} | {100*r['rewrite_routing_hit']:.2f}% | {100*r['paraphrase_routing_hit']:.2f}% | {r['rewrite_correct_delta_rank']:.2f} | {r['paraphrase_correct_delta_rank']:.2f} | {r['retained_bank_mib']:.3f} | {r['peak_gpu_allocated_gib']:.3f} |")
    text+=['','## Protocol and limits','',
        '- ZsRE has Q&A edits, not conversations. One delta stores one exact original question and its complete counterfactual answer. This is not the earlier LongMemEval task.',
        '- Existing independent compact AlphaEdit writer and normalized activation update-energy scorer are reused. Layers 13-15, 50-step value optimization, EOS target and fixed hyperparameters remain unchanged. No result-dependent tuning.',
        '- The shared raw-question interface omits our old Q:/A: wrapper at both write and recall. No artificial context is invented. WISE retains its own native prefix augmentation and native answer target policy.',
        '- The Mistral-specific nullspace projector uses first 200 training-locality prompts only, with the earlier thin-SVD method and .02 threshold. Held-out paraphrases and evaluation-locality questions never enter writing or projector construction. This small-corpus procedure is not full Wikipedia AlphaEdit statistics.',
        '- The router receives the raw query alone. It selects a delta before the official evaluator receives any target tokens. Each paraphrase and locality question is routed independently. There is no oracle identity lookup or abstention gate.',
        '- The exact unchanged official evaluator bodies and the original 1,000-item unedited baseline are reused. Ten baseline items were rechecked for exact parity before writes.',
        '- At each T, all first-T original questions, paraphrases and locality questions are evaluated with only the first-T deltas available. Lower-T cells contain T examples. Changes across checkpoints can also reflect different cohort difficulty.',
        '- One seed and one fixed bank per method. This is a complete-system comparison of different write and routing mechanisms, not an isolated causal test of session management alone.',
        '- Native WISE results are reused unchanged, rather than rerun. Their publication-era revision, raw data and complete per-question outputs remain in the separate reference run.',
        '- Retained bank bytes count deployed factor files only. Projectors and source snapshots are construction/research artifacts, not deployed memories. GPU peaks include the common base model; CPU, disk and GPU tiers are separate. No TTFT or write-time benchmark is inferred from teacher-forced evaluation wall time.',
        '', 'All checkpoint-level per-question scores, selected memory IDs, score margins, correct-delta ranks, factor hashes and frozen source/data manifests are retained.']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    svg=(out/'comparison.svg').read_text();svg='<svg'+svg.split('<svg',1)[1]
    route=(out/'routing_storage.svg').read_text();route='<svg'+route.split('<svg',1)[1]
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched ZsRE comparison</title>'
        '<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:20px}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style>'
        '<h1>Our activation-routed memory versus native WISE</h1>'+svg+route+'<pre>'+html.escape('\n'.join(text))+'</pre>')
    artifacts=['summary.csv','results.json','report.md','report.html']+[f'{name}.{ext}' for name in ['comparison','routing_storage'] for ext in ['png','svg','pdf']]
    save(out/'report_manifest.json',{name:digest(out/name) for name in artifacts})
    save(out/'completion.json',dict(completed=True,physical_gpu=1,edits=1000,verified_checkpoint_queries=sum(d['checkpoints']),
         metrics_per_query=3,report='report.html',finished_utc=now()))
    print('FINAL_REPORT_READY',out/'report.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out.resolve())
