"""Isolated, queued FP32 streaming-factor supplement to the frozen top-k run."""
import argparse
from collections.abc import Mapping
import copy
import gc
import html
from pathlib import Path
import statistics
import subprocess
import sys
import time
from experiments.zsre_topk import core as base

ROOT = base.ROOT
MODULE = 'experiments/topk_streaming.py'
METHODS = ['full_scan', 'topk_8']
DATASETS = ['zsre', 'longmemeval']
PASSES = 10
EXPECTED_REQUESTS = 25660
read, save, digest = base.read, base.save, base.digest


class LazyFactorBank(Mapping):
    """SID metadata only; each lookup deserializes one validated CPU payload.

    Counters describe filesystem/deserialize calls and serialized file sizes,
    not physical device reads. No payload is retained by this mapping.
    """
    def __init__(self, d, torch, model, names, n):
        self.d, self.torch = d, torch
        self.names = tuple(names)
        self.shapes = {name: tuple(model.get_parameter(name).shape) for name in names}
        self.files = {row['sid']: row for row in d['factor_files'][:n]}
        if len(self.files) != n or list(self.files) != d['bank_sids'][:n]:
            raise ValueError('Factor prefix/SID mismatch')
        self.reset_counters()

    def __iter__(self): return iter(self.files)
    def __len__(self): return len(self.files)
    def reset_counters(self): self.file_load_count = self.file_load_bytes = 0

    def __getitem__(self, sid):
        path = Path(self.files[sid]['path'])
        payload = self.torch.load(path, map_location='cpu', weights_only=True)
        self.file_load_count += 1
        self.file_load_bytes += path.stat().st_size
        if set(payload) != set(self.names): raise ValueError('Factor layer mismatch')
        if self.d['dataset'] == 'longmemeval':
            from qa500_core import canonical
            payload = canonical(payload)
        for name, (u, v) in payload.items():
            if (u.ndim not in (1, 2) or v.ndim != u.ndim or
                    u.dtype != self.torch.float32 or v.dtype != self.torch.float32 or
                    not self.torch.isfinite(u).all() or not self.torch.isfinite(v).all()):
                raise ValueError('Invalid FP32 saved factors')
            if (u.shape[0], v.shape[0]) != self.shapes[name]: raise ValueError('Factor shape mismatch')
            if u.ndim == 2 and (u.shape[1] < 1 or u.shape[1] != v.shape[1]):
                raise ValueError('Factor rank mismatch')
        return payload

    def validate_all(self):
        """Hash all files, then measure logical bytes one payload at a time."""
        disk = logical = largest = 0
        for sid, row in self.files.items():
            path = Path(row['path'])
            if digest(path) != row['sha256']: raise ValueError('Saved delta changed')
            disk += path.stat().st_size
            payload = self[sid]
            size = sum(t.numel() * t.element_size() for pair in payload.values() for t in pair)
            logical += size
            largest = max(largest, size)
            del payload
        self.largest_payload_bytes = largest
        self.reset_counters()
        return disk, logical


def check(out):
    d = base.check(out)
    for path, want in d['primary_frozen_sha256'].items():
        if digest(Path(path)) != want: raise ValueError('Frozen primary artifact changed: ' + path)
    return d


def check_root(root):
    for name, want in read(root / 'datasets.json').items():
        if digest(root / name / 'design.json') != want: raise ValueError('Supplement design changed')
        check(root / name)


def prepare(out, primary):
    out, primary = Path(out).resolve(), Path(primary).resolve()
    if out == primary or primary in out.parents or out in primary.parents:
        raise ValueError('Use a separate supplement root')
    if out.exists() and any(out.iterdir()): raise ValueError('Supplement output must be empty')
    manifests = {}
    for dataset in DATASETS:
        source, target = primary / dataset, out / dataset
        d = copy.deepcopy(base.check(source)); n = max(d['sizes'])
        target.mkdir(parents=True, exist_ok=True)
        frozen = [source / name for name in ('design.json', 'input_manifest.json', 'requests.json')]
        index = source / 'index' / f'bank-{n}.pt'
        meta = read(index.with_suffix('.json'))
        if digest(index) != meta['sha256']: raise ValueError('Primary activation index changed')
        frozen += [index, index.with_suffix('.json'), source / 'index/manifest.json', source / 'index/build.json']
        snapshot = {str(path): digest(path) for path in frozen}
        save(target / 'primary_snapshot.json', dict(primary_root=str(primary), sha256=snapshot,
             design=read(source / 'design.json'), input_manifest=read(source / 'input_manifest.json'), index_meta=meta))
        d.update(methods=METHODS, ks=[8], sizes=[n], plan=[d['plan'][-1]], expected_requests=2 * PASSES * len(d['plan'][-1]['case_ids']),
                 primary_root=str(primary), primary_frozen_sha256=snapshot, index_path=str(index), index_meta=meta,
                 memory='Disk-backed factors with no resident factor cache; one ephemeral payload per lookup.',
                 timing='Fresh streaming filesystem/deserialize calls; warm operating-system file cache allowed; no cache flush.',
                 supplement='Largest bank only, full scan versus top-k 8, ten actual FP32 inference passes.')
        d['code_sha256'][MODULE] = digest(ROOT / MODULE)
        ids = set(d['plan'][0]['case_ids'])
        requests = [{key: row[key] for key in ('case_id', 'prompt', 'target_new', 'answer_sids', 'abstention')}
                    for row in read(source / 'requests.json') if row['case_id'] in ids]
        if len(requests) != len(ids) or d['precision'] != 'float32' or d['answer_cap'] != 128:
            raise ValueError('Unexpected frozen cohort/precision/cap')
        if (n, len(requests), sum(row['abstention'] for row in requests)) != ((1000, 1000, 0) if dataset == 'zsre' else (500, 283, 15)):
            raise ValueError('Unexpected largest-bank denominator')
        save(target / 'design.json', d); save(target / 'requests.json', requests)
        save(target / 'input_manifest.json', {name: digest(target / name) for name in ('design.json', 'requests.json', 'primary_snapshot.json')})
        manifests[dataset] = digest(target / 'design.json')
    save(out / 'datasets.json', manifests)
    save(out / 'completion.json', dict(completed=False, stage='prepared', planned_requests=EXPECTED_REQUESTS))
    save(out / 'queue.json', dict(primary_root=str(primary), physical_gpu=1, methods=METHODS, passes=PASSES,
         planned_requests=EXPECTED_REQUESTS, prepared_utc=base.now()))
    check_root(out)


def evaluate(out, method, repeat):
    out = Path(out).resolve(); d = check(out)
    if method not in METHODS or repeat not in range(PASSES): raise ValueError('Invalid measurement arm/pass')
    n = d['sizes'][0]; folder = out / method / f'run-{repeat:02d}'; file = folder / f'bank-{n}.json'
    if file.exists() and read(file).get('completed'): return
    d, torch, tok, model = base.load(out, folder)
    torch.manual_seed(repeat); torch.cuda.manual_seed_all(repeat)
    from experiments.wise_qa500.core import HostPeak
    from memory_comparison_core import token_f1
    from question_key import normalize
    from ttft_timing import FirstAnswerClock, observe_generate
    names = [f'model.layers.{i}.mlp.down_proj.weight' for i in d['layers']]
    original = {name: model.get_parameter(name).detach().cpu().clone() for name in names}
    def restore():
        with torch.no_grad():
            for name, value in original.items(): model.get_parameter(name).copy_(value.to(model.device))
    warm = dict(d, answer_cap=1)
    for _ in range(2): base.generate(warm, model, tok, 'Answer briefly.')
    requests = {row['case_id']: row for row in read(out / 'requests.json')}
    ids = base.question_order(d, n, repeat); k = None if method == 'full_scan' else 8
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    trials = []
    try:
        with HostPeak() as host:
            base.status(folder, 'bank_setup', n=n, repeat=repeat)
            bank = LazyFactorBank(d, torch, model, names, n)
            disk, logical = bank.validate_all(); state = None; index_disk = index_bytes = 0
            if k is not None:
                state = torch.load(d['index_path'], map_location='cpu', weights_only=True)
                if state['n'] != n: raise ValueError('Wrong reused index prefix')
                index_disk, index_bytes = d['index_meta']['file_bytes'], d['index_meta']['payload_bytes']
            router = base.ActivationRouter(bank, names, None if state is None else state['vectors'], d['bank_sids'][:n],
                     index_sids=None if state is None else state['index_sids'], scorer=base.scorer(d))
            gc.collect(); torch.cuda.empty_cache()
            for qi, case_id in enumerate(ids):
                question = requests[case_id]['prompt']
                base.status(folder, 'query', n=n, repeat=repeat, completed=qi, total=len(ids), case_id=case_id)
                bank.reset_counters(); torch.cuda.synchronize(); arrival = time.monotonic(); observer = FirstAnswerClock(tok, arrival)
                if d['dataset'] == 'zsre': restore()
                keys = base.capture(d, model, tok, names, question); result = router.route(keys, k)
                routed = time.monotonic() - arrival
                score_loads, score_bytes = bank.file_load_count, bank.file_load_bytes
                if d['dataset'] == 'longmemeval': restore()
                payload = bank[result['selected']]
                try:
                    base.apply_payload(torch, model, payload)
                    with observe_generate(model, observer): final = base.generate(d, model, tok, question)
                finally: restore()
                torch.cuda.synchronize(); elapsed = time.monotonic() - arrival; del payload
                if score_loads != (n if k is None else min(k, n)) or bank.file_load_count != score_loads + 1:
                    raise ValueError('Streaming read-count invariant failed')
                label = requests[case_id]; support = set(label['answer_sids']); valid = not label['abstention']
                trials.append(dict(case_id=case_id, question=question, reference=label['target_new'], n=n, repeat=repeat,
                    final=final, token_f1=token_f1(final['text'], label['target_new']), exact=normalize(final['text']) == normalize(label['target_new']),
                    source_hit=result['selected'] in support if valid else None, selected=result['selected'],
                    shortlist_hit=bool(set(result['shortlist']) & support) if valid and result['shortlist'] is not None else None,
                    coarse_rank=min((i+1 for i,s in enumerate(result['coarse_order'] or []) if s in support), default=None) if valid else None,
                    fine_rank=min((i+1 for i,s in enumerate(result['fine_order']) if s in support), default=None) if valid else None,
                    shortlisted_sids=result['shortlist'], selected_score=result['score'], fine_margin=result['margin'],
                    scoring_file_load_count=score_loads, scoring_file_load_bytes=score_bytes,
                    apply_file_load_count=1, apply_file_load_bytes=bank.file_load_bytes-score_bytes,
                    factor_file_load_count=bank.file_load_count, factor_file_load_bytes=bank.file_load_bytes,
                    query_seconds=elapsed, routing_seconds=routed, output_tokens=final['tokens'], hit_cap=final['hit_cap'], **observer.result()))
                if (qi+1) % 10 == 0: save(file.with_suffix('.partial.json'), dict(completed=False, trials=trials))
            resources = dict(extra_disk_bank_bytes=disk+index_disk, factor_bank_disk_bytes=disk, key_index_disk_bytes=index_disk,
                logical_total_factor_payload_bytes=logical, resident_factor_cache_bytes=0, largest_ephemeral_factor_payload_bytes=bank.largest_payload_bytes,
                cpu_key_index_payload_bytes=index_bytes, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(), index_build_reused=True, index_build_new_writes=0)
        resources.update(host.result())
        save(file, dict(completed=True, available=True, physical_gpu=1, method=method, repeat=repeat, n=n, trials=trials,
             groups={'all': base.summary(trials)}, routing=base.routing_summary(trials), resources=resources, finished_utc=base.now()))
        file.with_suffix('.partial.json').unlink(missing_ok=True)
    finally: restore()


def gpu_free():
    result = subprocess.run(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True)
    return not result.stdout.strip()


def drive(root):
    root = Path(root).resolve(); check_root(root); primary = Path(read(root / 'queue.json')['primary_root'])
    try:
        while True:
            completion = read(primary / 'completion.json'); driver = read(primary / 'driver_status.json')
            status = read(primary / 'status.json') if (primary / 'status.json').exists() else {}
            if any(row.get('stage') in ('failed', 'blocked') for row in (completion, driver, status)):
                save(root / 'completion.json', dict(completed=False, stage='blocked', reason='Primary run failed; no GPU work started.'))
                save(root / 'driver_status.json', dict(stage='blocked', updated_utc=base.now()))
                return
            if completion.get('completed') is True and driver.get('stage') == 'completed' and gpu_free(): break
            base.status(root, 'queued_waiting_primary_and_gpu1'); time.sleep(30)
        check_root(root)
        for dataset in DATASETS:
            d = check(root / dataset)
            for name, want in d['model_manifest']['sha256'].items():
                if digest(Path(d['model']) / name) != want: raise ValueError('Frozen model changed')
        for repeat in range(PASSES):
            for dataset in DATASETS:
                for method in METHODS[repeat % 2:] + METHODS[:repeat % 2]:
                    check_root(root)
                    while not gpu_free(): base.status(root, 'waiting_for_gpu1'); time.sleep(30)
                    cmd = [sys.executable, '-u', '-m', 'experiments.topk_streaming', 'evaluate', '--out', str(root / dataset),
                           '--method', method, '--repeat', str(repeat)]
                    worker = subprocess.Popen(cmd, cwd=ROOT)
                    save(root / 'driver_status.json', dict(stage='running', dataset=dataset, method=method, repeat=repeat, worker_pid=worker.pid))
                    if worker.wait(): raise RuntimeError('Isolated streaming worker failed')
        report(root)
        save(root / 'driver_status.json', dict(stage='completed', finished_utc=base.now()))
    except BaseException as error:
        save(root / 'completion.json', dict(completed=False, stage='failed', error=repr(error), updated_utc=base.now()))
        save(root / 'driver_status.json', dict(stage='failed', error=repr(error), updated_utc=base.now()))
        raise


def launch(root):
    root = Path(root).resolve(); check_root(root)
    if read(root / 'completion.json').get('completed'): raise RuntimeError('Supplement already complete')
    marker = root / 'detached_launch.json'
    if marker.exists():
        proc = Path('/proc') / str(read(marker)['pid']) / 'cmdline'
        if proc.exists() and b'experiments.topk_streaming' in proc.read_bytes(): raise RuntimeError('Queue already running')
    cmd = [sys.executable, '-u', '-m', 'experiments.topk_streaming', 'drive', '--out', str(root)]
    with (root / 'driver.log').open('a') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 close_fds=True, start_new_session=True)
    save(marker, dict(pid=child.pid, command=cmd, detached=True, physical_gpu=1, started_utc=base.now()))
    save(root / 'driver_status.json', dict(stage='queued', pid=child.pid, updated_utc=base.now()))
    print('DETACHED_STREAMING_QUEUE', child.pid, flush=True)


def report(root):
    root = Path(root).resolve(); check_root(root)
    from experiments.zsre_topk.report import audit_trial, aggregate, csvwrite, GENERATION_METRICS, number, integer
    rows, passes, outputs = [], [], []
    memory = ['extra_disk_bank_bytes', 'factor_bank_disk_bytes', 'key_index_disk_bytes', 'logical_total_factor_payload_bytes',
              'resident_factor_cache_bytes', 'cpu_key_index_payload_bytes', 'host_peak_rss_bytes', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']
    for dataset in DATASETS:
        out = root / dataset; d = check(out); n = d['sizes'][0]
        requests = {r['case_id']: r for r in read(out / 'requests.json')}; sids = set(d['bank_sids'][:n])
        file_sizes = {r['sid']: Path(r['path']).stat().st_size for r in d['factor_files'][:n]}
        common_logical = None
        for method in METHODS:
            arm = []
            for repeat in range(PASSES):
                cell = read(out / method / f'run-{repeat:02d}' / f'bank-{n}.json'); trials = cell['trials']
                if (not cell['completed'] or cell['method'] != method or cell['repeat'] != repeat or cell['n'] != n or
                        [t['case_id'] for t in trials] != base.question_order(d, n, repeat)):
                    raise ValueError('Incomplete or mismatched streaming cell')
                for t in trials:
                    audit_trial(t, requests[t['case_id']], method, n, repeat, d['answer_cap'], sids)
                    count = n if method == 'full_scan' else min(8, n)
                    if (t['scoring_file_load_count'], t['apply_file_load_count'], t['factor_file_load_count']) != (count, 1, count+1):
                        raise ValueError('Raw streaming reads disagree with declared policy')
                    expected_score_bytes = sum(file_sizes[sid] for sid in (sids if method == 'full_scan' else t['shortlisted_sids']))
                    if (integer(t['scoring_file_load_bytes'], 'scoring_file_load_bytes') != expected_score_bytes or
                            integer(t['apply_file_load_bytes'], 'apply_file_load_bytes') != file_sizes[t['selected']] or
                            t['factor_file_load_bytes'] != t['scoring_file_load_bytes'] + t['apply_file_load_bytes']):
                        raise ValueError('Streaming byte accounting mismatch')
                    outputs.append(dict(dataset=dataset, method=method, **t))
                values = base.summary(trials); routing = base.routing_summary(trials); resources = cell['resources']
                if values != cell['groups']['all'] or routing != cell['routing'] or resources['resident_factor_cache_bytes'] != 0:
                    raise ValueError('Saved summary/cache invariant disagrees with raw cell')
                for key in memory: number(resources[key], key)
                expected_index = d['index_meta']['file_bytes'] if method != 'full_scan' else 0
                expected_index_payload = d['index_meta']['payload_bytes'] if method != 'full_scan' else 0
                if (resources['factor_bank_disk_bytes'] != sum(file_sizes.values()) or
                        resources['key_index_disk_bytes'] != expected_index or
                        resources['cpu_key_index_payload_bytes'] != expected_index_payload or
                        resources['extra_disk_bank_bytes'] != sum(file_sizes.values()) + expected_index or
                        resources['index_build_reused'] is not True or resources['index_build_new_writes'] != 0):
                    raise ValueError('Deployment factor/index accounting mismatch')
                logical = resources['logical_total_factor_payload_bytes']
                if common_logical is not None and logical != common_logical: raise ValueError('Factor payload differs across cells')
                common_logical = logical
                values.update(routing_accuracy=routing['accuracy'], shortlist_recall=routing['shortlist_recall'],
                    routing_mean_ms=1000*statistics.mean(t['routing_seconds'] for t in trials),
                    scoring_file_load_count=statistics.mean(t['scoring_file_load_count'] for t in trials),
                    factor_file_load_count=statistics.mean(t['factor_file_load_count'] for t in trials),
                    factor_file_load_bytes=statistics.mean(t['factor_file_load_bytes'] for t in trials))
                record = dict(dataset=dataset, method=method, n=n, repeat=repeat, **values, **{key: resources[key] for key in memory})
                passes.append(record); arm.append(record)
            row = dict(dataset=dataset, method=method, n=n, passes=PASSES, questions_per_pass=len(requests),
                       routing_questions_per_pass=sum(not r['abstention'] for r in requests.values()))
            for key in GENERATION_METRICS + ['routing_accuracy', 'shortlist_recall', 'routing_mean_ms', 'scoring_file_load_count', 'factor_file_load_count', 'factor_file_load_bytes'] + memory:
                row.update({key+'_'+suffix: value for suffix, value in aggregate([p[key] for p in arm]).items()})
            rows.append(row)
    if len(outputs) != EXPECTED_REQUESTS or len(passes) != 40: raise ValueError('Incomplete 25,660-request supplement')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = []
    for name, metrics in [('accuracy', [('mean_f1', 'Final-answer F1', 100), ('exact_rate', 'Final exact match', 100), ('routing_accuracy', 'Source routing', 100)]),
                          ('latency', [('ttft_mean_ms', 'Visible answer TTFT (ms)', 1), ('query_latency_mean_ms', 'Query latency (ms)', 1), ('routing_mean_ms', 'Routing latency (ms)', 1)]),
                          ('resources', [('extra_disk_bank_bytes', 'Factors + index disk (MiB)', 2**-20), ('host_peak_rss_bytes', 'Peak process RAM (GiB)', 2**-30), ('peak_gpu_allocated_bytes', 'Peak allocated GPU (GiB)', 2**-30)]),
                          ('reads', [('scoring_file_load_count', 'Scoring deserialize calls/query', 1), ('factor_file_load_count', 'Total factor calls/query', 1), ('factor_file_load_bytes', 'Serialized bytes/query (MiB)', 2**-20)])]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (key, title, scale) in zip(axes, metrics):
            for offset, method in enumerate(METHODS):
                selected = [next(r for r in rows if r['dataset'] == ds and r['method'] == method) for ds in DATASETS]
                ax.bar([i + offset*.36 for i in range(2)], [(r[key+'_mean'] or 0)*scale for r in selected], width=.36,
                       yerr=[(r[key+'_sd'] or 0)*scale for r in selected], capsize=3, label=method)
            ax.set_xticks([.18, 1.18], ['ZsRE N=1000', 'LongMemEval N=500']); ax.set_title(title); ax.grid(axis='y', alpha=.2)
        axes[0].legend(); fig.suptitle('Streaming factors: mean ± sample SD of ten fixed-bank inference passes'); fig.tight_layout()
        for extension in ('png', 'svg', 'pdf'): fig.savefig(root / f'{name}.{extension}', dpi=180)
        plt.close(fig); figures.append(name)
    csvwrite(root / 'comparison.csv', rows); csvwrite(root / 'per_pass.csv', passes); csvwrite(root / 'question_outputs.csv', outputs)
    save(root / 'comparison.json', dict(completed=True, summary=rows, verified_requests=len(outputs), verified_cells=len(passes),
         methods=METHODS, passes=PASSES, baseline_metrics_reused=False, aggregation='Mean ± sample SD of ten actual pass metrics'))
    text = ('# Streaming-factor supplement\n\nAll 40 cells and 25,660 fresh requests passed audit. '
            'ZsRE: 1,000 edits / 1,000 questions; LongMemEval: 500 conversations / 283 eligible questions, including 15 abstentions. '
            'Source routing excludes abstentions; final F1, exact match and latency include them. '
            'Each arm retains original FP32 model, factors, raw prompts, native scorer, 128-token decoder, restore placement and two warmups. '
            'Full scan deserializes N factors for scoring; top-k 8 deserializes min(8,N). Both reread the selected factor once to apply it.\n\n'
            'The factor mapping retains metadata and zero cached payload bytes. Logical total factor bytes describe the complete bank on disk, '
            'not resident tensor memory. At most one loaded factor payload is held by the scoring loop. Measured process RAM includes the model, '
            'original edited weights, allocator/runtime and operating-system effects; logical cache accounting is not an exact RSS attribution. '
            'Top-k adds the reused normalized CPU FP32 key index; deployment disk and CPU index bytes count it even though no new index is written.\n\n'
            'Timing includes capture, shortlist, filesystem/deserialization, unchanged fine scoring, selected-factor reload/application and generation. '
            'Setup and index construction are excluded; index construction is reused from the frozen primary. Warm operating-system file cache is allowed; '
            'no cache flush occurs. File call/byte counts are application-level filesystem/deserialization work, not measured physical cold-disk I/O. '
            'Ten passes share fixed factors and exact stored questions; their SD is inference variability, not training-seed or unseen-question uncertainty.\n')
    (root / 'summary.md').write_text(text)
    (root / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Streaming-factor supplement</title>'
        '<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}img{width:100%}pre{white-space:pre-wrap}</style>'
        '<pre>'+html.escape(text)+'</pre>'+''.join(f'<img src="{name}.svg" alt="{name}">' for name in figures))
    files = ['comparison.csv', 'comparison.json', 'per_pass.csv', 'question_outputs.csv', 'summary.md', 'report.html'] + [f'{name}.{ext}' for name in figures for ext in ('png', 'svg', 'pdf')]
    save(root / 'report_manifest.json', {name: digest(root / name) for name in files})
    save(root / 'completion.json', dict(completed=True, verified_requests=len(outputs), verified_cells=len(passes), planned_requests=EXPECTED_REQUESTS,
         physical_gpu=1, finished_utc=base.now(), report='report.html', report_manifest_sha256=digest(root / 'report_manifest.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'launch', 'drive', 'evaluate', 'report'])
    p.add_argument('--out', type=Path, required=True); p.add_argument('--primary', type=Path)
    p.add_argument('--method', choices=METHODS); p.add_argument('--repeat', type=int)
    a = p.parse_args()
    if a.action == 'prepare':
        if a.primary is None: p.error('prepare requires --primary')
        prepare(a.out, a.primary)
    elif a.action == 'evaluate': evaluate(a.out, a.method, a.repeat)
    else: globals()[a.action](a.out)
