"""Audit saved routing decisions and append a report after the frozen run completes.

This companion adds no inference calls and changes none of the running experiment's
frozen sources, inputs, outputs, generation settings or timing measurements.
"""
import argparse
import csv
import html
from pathlib import Path
import statistics
import subprocess
import sys
import time
from comparison_core import read, save, digest
from .core import ROOT, SIZES, check, order, now

DEFINITION = ('Routing accuracy = questions whose selected delta ID equals their '
              'written memory ID / all evaluated questions. Answer text, F1 and EOS '
              'do not affect this measure. Each checkpoint evaluates all first-N '
              'original questions against only the first-N saved deltas. Report the '
              'mean and sample SD of ten actual inference passes on one fixed bank.')


def audit_cell(cell, n, repeat):
    if not cell.get('completed') or not cell.get('available'):
        raise ValueError('Routing requires a complete available measurement cell')
    if (cell['method'], cell['n'], cell['repeat'], cell['physical_gpu']) != ('our_delta', n, repeat, 1):
        raise ValueError('Routing measurement identity mismatch')
    trials = cell['trials']
    if [t['case_id'] for t in trials] != order(n, repeat):
        raise ValueError('Routing requires every question in the matched cohort/order')
    bank = {f'case_{i:04d}' for i in range(n)}
    decisions = []
    for t in trials:
        expected = f"case_{t['case_id']:04d}"
        selected = t['selected']
        if selected not in bank:
            raise ValueError('Selected delta does not belong to this checkpoint')
        hit = selected == expected
        if type(t['source_hit']) is not bool or t['source_hit'] != hit:
            raise ValueError('Saved routing flag disagrees with selected delta ID')
        rank = t['correct_rank']
        if type(rank) is not int or not 1 <= rank <= n or (rank == 1) != hit:
            raise ValueError('Correct-delta rank disagrees with selected delta ID')
        decisions.append(dict(n=n, repeat=repeat, case_id=t['case_id'], expected_delta=expected,
                              selected_delta=selected, routing_correct=hit, correct_rank=rank))
    hits = sum(t['routing_correct'] for t in decisions)
    return dict(n=n, repeat=repeat, questions=n, correct=hits, routing_accuracy=hits/n), decisions


def aggregate_passes(rows, n):
    if len(rows) != 10 or {r['repeat'] for r in rows} != set(range(10)):
        raise ValueError('Require ten distinct actual inference passes')
    if any(r['n'] != n or r['questions'] != n for r in rows):
        raise ValueError('Routing denominators differ across passes')
    values = [r['routing_accuracy'] for r in rows]
    return dict(n=n, passes=10, questions_per_pass=n, total_decisions=10*n,
                total_correct=sum(r['correct'] for r in rows),
                correct_per_pass_min=min(r['correct'] for r in rows),
                correct_per_pass_max=max(r['correct'] for r in rows),
                routing_accuracy_mean=statistics.mean(values),
                routing_accuracy_sd=statistics.stdev(values))


def csvwrite(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def build(out):
    d = check(out)
    config = read(out/'routing_extension.json')
    if digest(__file__) != config['source_sha256']:
        raise ValueError('Routing extension source changed after launch')
    if digest(out/'input_manifest.json') != config['input_manifest_sha256']:
        raise ValueError('Frozen input manifest changed after routing launch')
    if not read(out/'completion.json').get('completed') or read(out/'driver_status.json')['stage'] != 'completed':
        raise ValueError('Final routing graphs require the full benchmark to complete successfully')
    folder = out/'routing'; folder.mkdir(exist_ok=True)
    done = folder/'completion.json'
    if done.exists() and read(done).get('completed'):
        for name, want in read(folder/'artifact_manifest.json').items():
            if digest(out/name) != want: raise ValueError('Completed routing artifact changed: '+name)
        return
    rows = []; perpass = []; decisions = []
    requests = read(out/'requests.json')
    for n in SIZES:
        cohort = []
        for repeat in range(10):
            source = out/'our_delta'/f'run-{repeat:02d}'
            if read(source/'provenance.json')['code_sha256'] != d['code_sha256']:
                raise ValueError('Mixed inference source versions')
            cell = read(source/f'bank-{n}.json')
            row, raw = audit_cell(cell, n, repeat)
            for t in cell['trials']:
                r = requests[t['case_id']]
                if (t['question'], t['reference']) != (r['prompt'], r['target_new']):
                    raise ValueError('Official question/memory binding mismatch')
            cohort.append(row); perpass.append(row); decisions.extend(raw)
        rows.append(aggregate_passes(cohort, n))
    # Preserve the original report and its audit before adding the routing section.
    archive = folder/'original_report_manifest.json'
    if not archive.exists():
        original = read(out/'report_manifest.json')
        for name, want in original.items():
            if digest(out/name) != want: raise ValueError('Original report artifact changed: '+name)
        for ext in ['md', 'html']:
            (folder/f'original_report.{ext}').write_bytes((out/f'report.{ext}').read_bytes())
        save(archive, original)
    original = read(archive)
    for ext in ['md', 'html']:
        if digest(folder/f'original_report.{ext}') != original[f'report.{ext}']:
            raise ValueError('Archived base report changed')
    csvwrite(folder/'accuracy.csv', rows)
    csvwrite(folder/'per_pass.csv', perpass)
    csvwrite(folder/'decisions.csv', decisions)
    save(folder/'accuracy.json', dict(completed=True, definition=DEFINITION, summary=rows,
         verified_decisions=len(decisions), unique_memories=max(SIZES), passes=10,
         inference_calls_added=0, source_sha256=config['source_sha256']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar([r['n'] for r in rows], [100*r['routing_accuracy_mean'] for r in rows],
                yerr=[100*r['routing_accuracy_sd'] for r in rows],
                color='#0077B6', marker='o', capsize=3, label='SessionControl')
    ax.set_xscale('log', base=2); ax.set_xticks(SIZES, labels=SIZES)
    ax.set_ylim(0, 103); ax.set_ylabel('Correct delta selected (%)'); ax.set_xlabel('Stored ZsRE Q/A items')
    ax.set_title('Activation routing accuracy · mean ± sample SD over ten passes')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for ext in ['png', 'svg', 'pdf']:fig.savefig(folder/f'accuracy.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['## SessionControl routing accuracy', '', DEFINITION, '',
             '| Stored items | Questions per pass | Passes | Routing accuracy (%) | Correct per pass (min–max) |',
             '|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['n']} | {r['questions_per_pass']} | 10 | "
                     f"{100*r['routing_accuracy_mean']:.2f} ± {100*r['routing_accuracy_sd']:.2f} | "
                     f"{r['correct_per_pass_min']}–{r['correct_per_pass_max']} |")
    lines += ['', 'Routing accuracy measures memory selection, separately from complete generated-answer F1/exact match. '
              'All questions count even if generation is empty, incorrect or reaches the token cap. '
              'Repeated inference on a fixed deterministic bank does not estimate independent training-seed uncertainty.']
    text = '\n'.join(lines)+'\n'
    (folder/'report.md').write_text(text)
    svg = '<svg'+(folder/'accuracy.svg').read_text().split('<svg', 1)[1]
    section = '<section id="routing-accuracy"><h2>SessionControl routing accuracy</h2>'+svg+'<pre>'+html.escape(text)+'</pre></section>'
    (out/'report.md').write_text((folder/'original_report.md').read_text()+'\n'+text)
    (out/'report.html').write_text((folder/'original_report.html').read_text()+section)
    (folder/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SessionControl routing accuracy</title>'
        '<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style>'+section)
    files = ['routing/'+name for name in ['accuracy.csv', 'per_pass.csv', 'decisions.csv', 'accuracy.json',
             'accuracy.png', 'accuracy.svg', 'accuracy.pdf', 'report.md', 'report.html',
             'original_report_manifest.json', 'original_report.md', 'original_report.html']]
    manifest = {name:digest(out/name) for name in files+['report.md', 'report.html']}
    save(folder/'artifact_manifest.json', manifest)
    save(out/'report_manifest.json', {**original, **manifest, 'routing/artifact_manifest.json':digest(folder/'artifact_manifest.json')})
    save(done, dict(completed=True, finished_utc=now(), verified_decisions=len(decisions),
                    sizes=SIZES, passes=10, inference_calls_added=0, report='report.html'))
    print('ROUTING_REPORT_READY', folder/'report.html', flush=True)


def watch(out):
    config = read(out/'routing_extension.json')
    try:
        while True:
            if digest(__file__) != config['source_sha256']:raise ValueError('Routing companion source changed')
            state = read(out/'completion.json')
            if state.get('completed') and read(out/'driver_status.json')['stage'] == 'completed':break
            if state.get('stage') == 'failed':raise RuntimeError('Primary experiment failed; no final routing graph produced')
            save(out/'routing_status.json', dict(stage='waiting_for_full_experiment', updated_utc=now()))
            time.sleep(30)
        build(out)
        save(out/'routing_status.json', dict(stage='completed', updated_utc=now()))
    except BaseException as error:
        save(out/'routing_status.json', dict(stage='failed', error=repr(error), updated_utc=now()))
        raise


def launch(out):
    check(out)
    meta = out/'routing_launch.json'
    if meta.exists():
        p = Path('/proc')/str(read(meta)['pid'])/'cmdline'
        if p.exists() and b'zsre_generated.routing_report' in p.read_bytes():raise RuntimeError('Routing companion already running')
    save(out/'routing_extension.json', dict(source_sha256=digest(__file__),
         input_manifest_sha256=digest(out/'input_manifest.json'), definition=DEFINITION,
         inference_calls_added=0, requested_utc=now()))
    command = [sys.executable, '-u', '-m', 'experiments.zsre_generated.routing_report', 'watch', '--out', str(out)]
    with (out/'routing.log').open('a') as log:
        p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             close_fds=True, start_new_session=True)
    save(meta, dict(pid=p.pid, command=command, started_utc=now(), detached=True, uses_gpu=False))
    print('DETACHED_ROUTING_REPORT_COMPANION', p.pid, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['launch', 'watch', 'build'])
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    globals()[args.action](args.out.resolve())
