"""Add a two-method memory figure to an already completed, audited report.

No new inference, memory estimates, or changes to experimental measurements.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out):
    completion=read(out/'completion.json');manifest=read(out/'report_manifest.json')
    if not completion.get('completed') or completion['passes']!=10:
        raise ValueError('Only a successfully completed ten-pass experiment may be graphed')
    if digest(out/'comparison.json')!=manifest['comparison.json']:
        raise ValueError('Audited memory measurements changed')
    for name,h in manifest.items():
        if digest(out/name)!=h:raise ValueError('Existing report artifact changed: '+name)
    data=read(out/'comparison.json')['summary'];methods=['our_delta','rag']
    labels={'our_delta':'Our method (SessionControl)','rag':'RAG'}
    colors={'our_delta':'#0077b6','rag':'#555555'};markers={'our_delta':'o','rag':'s'}
    sizes=completion['checkpoints']
    fields=['extra_disk_bank_bytes','host_peak_rss_bytes','peak_gpu_allocated_bytes']
    selected=[]
    for n in sizes:
        for method in methods:
            row=next(r for r in data if r['group']=='all' and r['n']==n and r['method']==method)
            if row['status']!='measured' or row['passes']!=10:raise ValueError('Missing measured memory cell')
            selected.append({k:row[k] for k in ['method','n','passes','questions',*[f+s for f in fields for s in ['_mean','_sd']]]})
    revision=out/'report_revisions/before_focused_memory';revision.mkdir(parents=True,exist_ok=True)
    for name in ['report.html','report.md','report_manifest.json']:
        if not (revision/name).exists():shutil.copy2(out/name,revision/name)
    with (out/'memory_rag_vs_ours.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    panels=[('extra_disk_bank_bytes','Retained bank on disk','MiB',2**20),
            ('host_peak_rss_bytes','Peak process host RAM','GiB',2**30),
            ('peak_gpu_allocated_bytes','Peak GPU allocated memory','GiB',2**30)]
    for ax,(field,title,unit,divisor) in zip(axes,panels):
        ymax=0.
        for method in methods:
            rows=[r for r in selected if r['method']==method]
            y=[r[field+'_mean']/divisor for r in rows];sd=[r[field+'_sd']/divisor for r in rows]
            ax.errorbar(sizes,y,yerr=sd,marker=markers[method],color=colors[method],lw=2,capsize=3,label=labels[method])
            ymax=max(ymax,max(a+b for a,b in zip(y,sd)))
        ax.set_xscale('log',base=2);ax.set_xticks(sizes,labels=sizes)
        ax.set_ylim(0,ymax*1.15);ax.set_xlabel('Stored conversations');ax.set_ylabel(unit);ax.set_title(title)
        ax.grid(alpha=.2)
    fig.suptitle('Measured memory use: our method and RAG only',fontsize=16)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=10)
    fig.text(.5,.015,'Mean ± sample SD across ten new inference passes. All panels start at zero. GPU peaks include common base-model weights.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.83))
    for ext in ['svg','png','pdf']:fig.savefig(out/f'memory_rag_vs_ours.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    final={method:next(r for r in selected if r['method']==method and r['n']==500) for method in methods}
    ours=final['our_delta'];rag=final['rag']
    disk_ratio=ours['extra_disk_bank_bytes_mean']/rag['extra_disk_bank_bytes_mean']
    gpu_lower=100*(1-ours['peak_gpu_allocated_bytes_mean']/rag['peak_gpu_allocated_bytes_mean'])
    note=(f'At 500 conversations, RAG uses less disk storage and host RAM. Our method uses '
          f'{gpu_lower:.2f}% less peak allocated GPU memory, while its retained disk bank is {disk_ratio:.2f} times larger. '
          'The GPU totals include the common base model and method-specific query allocations; disk, host RAM and GPU memory are separate measures.')
    table='<table style="border-collapse:collapse"><tr><th>Method at 500 conversations</th><th>Disk bank (MiB)</th><th>Peak host RAM (GiB)</th><th>Peak GPU allocated (GiB)</th></tr>'
    markdown=['','## Memory focus: our method and RAG','',
              '![Measured memory use: our method and RAG only](memory_rag_vs_ours.png)','',
              '| Method at 500 conversations | Disk bank (MiB) | Peak host RAM (GiB) | Peak GPU allocated (GiB) |',
              '|---|---:|---:|---:|']
    for method in methods:
        r=final[method]
        values=[f"{r[field+'_mean']/divisor:.3f} ± {r[field+'_sd']/divisor:.3f}" for field,_,_,divisor in panels]
        table+='<tr><td>'+html.escape(labels[method])+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
        markdown.append('| '+labels[method]+' | '+' | '.join(values)+' |')
    table+='</table>'
    markdown+=['',note,'','All values are measured means ± sample SD over the ten additional passes. GPU peaks include base-model weights. '
               'The full eligible query set changes with the checkpoint. No tiers are summed into one memory number.','']
    svg='<svg'+(out/'memory_rag_vs_ours.svg').read_text().split('<svg',1)[1]
    section='<!-- focused-memory-start --><section id="memory-rag-vs-ours"><style>#memory-rag-vs-ours td,#memory-rag-vs-ours th{padding:10px;border:1px solid #ccc}</style><h2>Memory focus: our method and RAG</h2>'+svg+table+'<p>'+html.escape(note)+'</p><p>Download: <a href="memory_rag_vs_ours.png">PNG</a> · <a href="memory_rag_vs_ours.svg">SVG</a> · <a href="memory_rag_vs_ours.pdf">PDF</a> · <a href="memory_rag_vs_ours.csv">CSV</a></p></section><!-- focused-memory-end -->'
    page=(out/'report.html').read_text()
    if '<!-- focused-memory-start -->' in page:
        page=re.sub(r'<!-- focused-memory-start -->.*?<!-- focused-memory-end -->',lambda _:section,page,flags=re.S)
    else:
        original='<svg'+(out/'memory.svg').read_text().split('<svg',1)[1]
        if page.count(original)!=1:raise ValueError('Cannot identify the original memory figure unambiguously')
        page=page.replace(original,section,1)
    (out/'report.html').write_text(page)
    text=(out/'report.md').read_text();marker='\n## Memory focus: our method and RAG\n'
    if marker in text:text=text.split(marker,1)[0]
    (out/'report.md').write_text(text.rstrip()+'\n'+'\n'.join(markdown))
    save_path=out/'focused_memory_addendum.json'
    save_path.write_text(json.dumps(dict(source_comparison_sha256=digest(out/'comparison.json'),methods=methods,
        checkpoints=sizes,passes=10,measurements='Existing measured data only; no inference rerun.',
        previous_report_manifest_sha256=digest(revision/'report_manifest.json'),
        final_disk_ratio_ours_over_rag=disk_ratio,final_gpu_percent_lower_ours=gpu_lower),indent=2)+'\n')
    additions=['memory_rag_vs_ours.csv','memory_rag_vs_ours.svg','memory_rag_vs_ours.png','memory_rag_vs_ours.pdf','focused_memory_addendum.json']
    manifest.update({name:digest(out/name) for name in ['report.html','report.md',*additions]})
    (out/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(note)
    print('Updated existing report:',out/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run.resolve())
