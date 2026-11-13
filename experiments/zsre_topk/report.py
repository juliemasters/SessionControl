"""Audit all matched inference cells before producing the top-k comparison."""
import argparse
import csv
from datetime import datetime
import html
import math
from pathlib import Path
import statistics

from .core import ROOT, METHODS, LABELS, check, read, save, digest, now, question_order, summary


PASSES = 10
GENERATION_METRICS = [
    'mean_f1', 'exact_rate', 'ttft_mean_ms', 'ttft_median_ms', 'ttft_p95_ms',
    'first_generated_mean_ms', 'query_latency_mean_ms', 'output_tokens_mean', 'hit_cap_rate',
]
ROUTING_METRICS = ['routing_accuracy', 'shortlist_recall', 'routing_mean_ms', 'routing_p95_ms']
MEMORY = [
    'extra_disk_bank_bytes', 'factor_bank_disk_bytes', 'key_index_disk_bytes',
    'host_peak_rss_bytes', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes',
    'cpu_factor_payload_bytes', 'cpu_key_index_payload_bytes',
]
STYLES = {
    'full_scan': ('#777777', 's', '--'),
    'topk_1': ('#073B64', 'o', '-'),
    'topk_8': ('#076BA5', '^', '-'),
    'topk_16': ('#148FC1', 'D', '-'),
    'topk_32': ('#67AFCD', 'v', '-'),
}


def csvwrite(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values):
    """Describe ten pass means, retaining missing-visible-answer denominators."""
    if len(values) != PASSES:
        raise ValueError('Require exactly ten actual inference passes')
    present = [value for value in values if value is not None]
    return dict(
        mean=statistics.mean(present) if present else None,
        sd=statistics.stdev(present) if len(present) > 1 else None,
        valid_passes=len(present),
    )


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: expected a finite number')
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name}: invalid value {value!r}')
    return value


def integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name}: invalid integer {value!r}')
    return value


def instant(value, name):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f'{name}: invalid timestamp') from error
    if result.tzinfo is None:
        raise ValueError(f'{name}: timestamp requires a timezone')
    return result


def audit_trial(trial, request, method, n, repeat, answer_cap, valid_sids):
    from memory_comparison_core import token_f1
    from question_key import normalize

    if trial.get('repeat', repeat) != repeat or trial.get('n', n) != n:
        raise ValueError('Trial cell identity mismatch')
    if (trial['question'], trial['reference']) != (request['prompt'], request['target_new']):
        raise ValueError('Raw source question/reference mismatch')
    final = trial['final']
    if not isinstance(final['text'], str):
        raise ValueError('Missing final generated-answer text')
    actual_f1 = token_f1(final['text'], request['target_new'])
    if abs(number(trial['token_f1'], 'token_f1') - actual_f1) > 1e-12:
        raise ValueError('Complete final-answer F1 mismatch')
    actual_exact = normalize(final['text']) == normalize(request['target_new'])
    if type(trial['exact']) is not bool or trial['exact'] != actual_exact:
        raise ValueError('Complete final-answer exact-match mismatch')
    tokens = integer(trial['output_tokens'], 'output_tokens', 1, answer_cap)
    if tokens != final['tokens'] or len(final['token_ids']) != tokens:
        raise ValueError('Generated-token count mismatch')
    if (type(trial['hit_cap']) is not bool or trial['hit_cap'] != final['hit_cap']
            or final['finish_reason'] not in ('cap', 'eos', 'next_turn')
            or trial['hit_cap'] != (final['finish_reason'] == 'cap')):
        raise ValueError('Generation stopping metadata mismatch')
    if trial['hit_cap'] and tokens != answer_cap:
        raise ValueError('Generation marked capped before answer cap')

    elapsed = number(trial['query_seconds'], 'query_seconds')
    routing = number(trial['routing_seconds'], 'routing_seconds')
    generated = number(trial['first_generated_token_seconds'], 'first_generated_token_seconds')
    if not 0 <= routing <= generated <= elapsed or elapsed <= 0:
        raise ValueError('Invalid routing/first-generated/query timestamp order')
    visible = trial['ttft_seconds']
    if visible is not None and not generated <= number(visible, 'ttft_seconds') <= elapsed:
        raise ValueError('Invalid first-visible-answer timestamp')

    selected = trial['selected']
    if selected not in valid_sids:
        raise ValueError('Selected source lies outside the fixed bank prefix')
    shortlist = trial.get('shortlisted_sids')
    if method == 'full_scan':
        if shortlist is not None:
            raise ValueError('Full scan must not save a coarse shortlist')
    elif shortlist is not None:
        if (not isinstance(shortlist, list) or len(shortlist) != min(int(method.split('_')[1]), n)
                or len(shortlist) != len(set(shortlist)) or not set(shortlist) <= valid_sids
                or selected not in shortlist):
            raise ValueError('Invalid distinct-source coarse shortlist')
    if request['abstention']:
        if any(trial[key] is not None for key in ('source_hit', 'shortlist_hit', 'coarse_rank', 'fine_rank')):
            raise ValueError('Abstention questions must not receive source-routing labels')
        return
    support = set(request['answer_sids'])
    if not support or not support <= valid_sids:
        raise ValueError('Answerable question lacks its complete official support set')
    if type(trial['source_hit']) is not bool or trial['source_hit'] != (selected in support):
        raise ValueError('Routing hit flag disagrees with selected source ID')
    if method == 'full_scan':
        if trial['coarse_rank'] is not None or trial['shortlist_hit'] is not None:
            raise ValueError('Full scan must not claim a coarse shortlist')
        fine = integer(trial['fine_rank'], 'fine_rank', 1, n)
    else:
        k = min(int(method.split('_')[1]), n)
        coarse = integer(trial['coarse_rank'], 'coarse_rank', 1, n)
        if type(trial['shortlist_hit']) is not bool or trial['shortlist_hit'] != (coarse <= k):
            raise ValueError('Shortlist recall disagrees with the coarse source rank')
        if shortlist is not None and trial['shortlist_hit'] != bool(set(shortlist) & support):
            raise ValueError('Shortlist hit flag disagrees with original answer sources')
        if trial['shortlist_hit']:
            fine = integer(trial['fine_rank'], 'fine_rank', 1, k)
        else:
            if trial['fine_rank'] is not None or trial['source_hit']:
                raise ValueError('Source absent from shortlist cannot have a fine rank or hit')
            fine = None
    if trial['source_hit'] != (fine == 1):
        raise ValueError('Selected source disagrees with the source fine-score rank')
    # Some workers retain full rankings as well as minimum support ranks.
    for key, rank_key in (('coarse_order', 'coarse_rank'), ('fine_order', 'fine_rank')):
        ranking = trial.get(key)
        if ranking is None:
            continue
        if not isinstance(ranking, list) or len(ranking) != len(set(ranking)) or not set(ranking) <= valid_sids:
            raise ValueError('Invalid saved source ranking')
        expected_rank = min((index + 1 for index, sid in enumerate(ranking) if sid in support), default=None)
        if trial[rank_key] != expected_rank:
            raise ValueError('Saved rank does not match the first official supporting source')
        if key == 'fine_order' and (not ranking or ranking[0] != selected):
            raise ValueError('Selected source disagrees with the fine order')


def routing_summary(trials, method):
    count = len(trials)
    answerable = [trial for trial in trials if trial['source_hit'] is not None]
    correct = sum(trial['source_hit'] for trial in answerable)
    shortlisted = [trial['shortlist_hit'] for trial in trials if trial['shortlist_hit'] is not None]
    shortlist_correct = sum(shortlisted) if shortlisted else None
    return dict(
        accuracy=correct / len(answerable) if answerable else None,
        correct=correct, questions=count, routing_questions=len(answerable),
        shortlist_recall=None if not shortlisted else shortlist_correct / len(shortlisted),
        shortlist_correct=shortlist_correct,
    )


def audit_inputs(out, design):
    """Match normalized requests and eligible cohorts to the frozen source."""
    source = Path(design['source_run'])
    source = source if source.is_absolute() else ROOT / source
    if source.resolve() == out.resolve():
        raise ValueError('Source and new measurement run must be distinct')
    old_design = read(source / 'design.json')
    if not read(source / 'completion.json').get('completed'):
        raise ValueError('The reused factor source is not a completed experiment')
    if design['sizes'] != old_design['sizes']:
        raise ValueError('The comparison must retain all original bank checkpoints')
    for key in ('model', 'model_revision', 'answer_cap', 'layers'):
        if key in old_design and design[key] != old_design[key]:
            raise ValueError('Original model/generation/write configuration changed: ' + key)
    requests = read(out / 'requests.json')
    by_id = {request['case_id']: request for request in requests}
    if len(by_id) != len(requests):
        raise ValueError('Duplicate normalized question IDs')
    if design['dataset'] == 'zsre':
        source_requests = read(source / 'requests.json')
        originals = {row['case_id']: row for row in source_requests}
        expected_bank = [f'case_{index:04d}' for index in range(1000)]
        if len(requests) != 1000 or set(by_id) != set(originals):
            raise ValueError('Require the same 1,000 raw ZsRE questions and references')
        for case_id, request in by_id.items():
            original = originals[case_id]
            if ((request['prompt'], request['target_new']) != (original['prompt'], original['target_new'])
                    or request['answer_sids'] != [f'case_{case_id:04d}'] or request['abstention'] is not False):
                raise ValueError('Original ZsRE raw Q/A or source identity changed')
    else:
        originals = {row['qid']: row for row in read(source / 'queries.json')}
        expected_bank = [row['sid'] for row in read(source / 'sessions.json')]
        for case_id, request in by_id.items():
            original = originals[case_id]
            if ((request['prompt'], request['target_new']) != (original['question'], original['answer'])
                    or request['answer_sids'] != original['support_sids']
                    or request['abstention'] != original['abstention']):
                raise ValueError('Official LongMemEval question, answer or support metadata changed')
    if design['bank_sids'] != expected_bank or len(set(expected_bank)) != len(expected_bank):
        raise ValueError('Original ordered conversation/edit bank changed')
    if [plan['n'] for plan in design['plan']] != design['sizes']:
        raise ValueError('Measurement plan does not match bank sizes')
    old_plans = {plan['n']: plan for plan in old_design['plan']}
    for plan in design['plan']:
        n = plan['n']
        if design['dataset'] == 'zsre':
            expected_ids = list(range(n))
        else:
            expected_ids = old_plans[n]['qids']
        if plan['case_ids'] != expected_ids or not plan['case_ids'] or not set(expected_ids) <= set(by_id):
            raise ValueError('Eligible source question cohort changed')
        prefix = set(expected_bank[:n])
        for case_id in expected_ids:
            request = by_id[case_id]
            if not set(request['answer_sids']) <= prefix:
                raise ValueError('Question evaluated before its complete support set is present')
    return by_id


def audit_resources(resources, method):
    for key in MEMORY:
        integer(resources[key], key)
    if resources['factor_bank_disk_bytes'] <= 0 or resources['cpu_factor_payload_bytes'] <= 0:
        raise ValueError('Missing common preloaded FP32 factor bank')
    if resources['extra_disk_bank_bytes'] != resources['factor_bank_disk_bytes'] + resources['key_index_disk_bytes']:
        raise ValueError('Retained bank disk excludes factors or activation index')
    if method == 'full_scan':
        if resources['key_index_disk_bytes'] or resources['cpu_key_index_payload_bytes']:
            raise ValueError('Full scan must not hold the optional activation index')
    elif resources['key_index_disk_bytes'] <= 0 or resources['cpu_key_index_payload_bytes'] <= 0:
        raise ValueError('Top-k activation-index disk/RAM overhead is missing')
    if resources['host_peak_rss_bytes'] < resources['cpu_factor_payload_bytes'] + resources['cpu_key_index_payload_bytes']:
        raise ValueError('Process RAM cannot be smaller than its retained tensor payloads')
    if not 0 < resources['peak_gpu_allocated_bytes'] <= resources['peak_gpu_reserved_bytes']:
        raise ValueError('Invalid GPU allocated/reserved peak accounting')


def audit(out):
    """Read-only gate: no partial report is emitted when a cell is missing."""
    design = check(out)
    if (design['repeats'] != PASSES or design['methods'] != METHODS
            or design['dataset'] not in ('zsre', 'longmemeval') or design['physical_gpu'] != 1
            or design['precision'] != 'float32'):
        raise ValueError('Unexpected matched five-arm, ten-pass FP32 design')
    requests = audit_inputs(out, design)
    sizes = design['sizes']

    rows, perpass, outputs = [], [], []
    common_factor_sizes, common_index_sizes = {}, {}
    verified = 0
    for method in METHODS:
        for n in sizes:
            passes = []
            valid_sids = set(design['bank_sids'][:n])
            planned_ids = next(plan['case_ids'] for plan in design['plan'] if plan['n'] == n)
            question_count = len(planned_ids)
            for repeat in range(PASSES):
                folder = out / method / f'run-{repeat:02d}'
                provenance = read(folder / 'provenance.json')
                if provenance['physical_gpu'] != 1 or provenance['code_sha256'] != design['code_sha256']:
                    raise ValueError('Execution source/GPU provenance mismatch')
                cell = read(folder / f'bank-{n}.json')
                if (cell.get('completed') is not True or cell.get('available') is not True
                        or (cell['method'], cell['repeat'], cell['n'], cell['physical_gpu']) != (method, repeat, n, 1)):
                    raise ValueError('Every planned cell must be complete, measured and available')
                if instant(cell['finished_utc'], 'cell finished_utc') < instant(provenance['started_utc'], 'run started_utc'):
                    raise ValueError('Cell completion precedes measurement-process provenance')
                trials = cell['trials']
                if [trial['case_id'] for trial in trials] != question_order(design, n, repeat):
                    raise ValueError('Question cohort/order differs from the matched inference schedule')
                for trial in trials:
                    request = requests[trial['case_id']]
                    audit_trial(trial, request, method, n, repeat, design['answer_cap'], valid_sids)
                    outputs.append(dict(
                        method=method, n=n, repeat=repeat, case_id=trial['case_id'],
                        question=trial['question'], reference=trial['reference'], final_answer=trial['final']['text'],
                        answer_sids='|'.join(request['answer_sids']), abstention=request['abstention'],
                        token_f1=trial['token_f1'], exact=trial['exact'], selected=trial['selected'],
                        source_hit=trial['source_hit'], shortlist_hit=trial['shortlist_hit'],
                        coarse_rank=trial['coarse_rank'], fine_rank=trial['fine_rank'],
                        routing_seconds=trial['routing_seconds'], ttft_seconds=trial['ttft_seconds'],
                        first_generated_token_seconds=trial['first_generated_token_seconds'],
                        query_seconds=trial['query_seconds'], output_tokens=trial['output_tokens'],
                        hit_cap=trial['hit_cap'], finish_reason=trial['final']['finish_reason'],
                    ))
                    verified += 1
                values = summary(trials)
                route = routing_summary(trials, method)
                if values != cell['groups']['all'] or route != cell['routing']:
                    raise ValueError('Saved generation/routing summary disagrees with raw trials')
                audit_resources(cell['resources'], method)
                resources = cell['resources']
                factors = (resources['factor_bank_disk_bytes'], resources['cpu_factor_payload_bytes'])
                if factors != common_factor_sizes.setdefault(n, factors):
                    raise ValueError('Factor bank disk/CPU payload differs across matched arms or repeats')
                if method != 'full_scan':
                    index = (resources['key_index_disk_bytes'], resources['cpu_key_index_payload_bytes'])
                    if index != common_index_sizes.setdefault(n, index):
                        raise ValueError('Top-k arms do not share the same activation-index overhead')
                route_times = sorted(trial['routing_seconds'] * 1000 for trial in trials)
                values.update(
                    routing_accuracy=route['accuracy'], shortlist_recall=route['shortlist_recall'],
                    routing_mean_ms=statistics.mean(route_times), routing_p95_ms=route_times[math.ceil(.95 * question_count) - 1],
                )
                passrow = dict(method=method, n=n, repeat=repeat, **values,
                               routing_correct=route['correct'], shortlist_correct=route['shortlist_correct'],
                               routing_questions=route['routing_questions'],
                               **{key: resources[key] for key in MEMORY})
                perpass.append(passrow)
                passes.append(passrow)
            row = dict(method=method, label=LABELS[method], n=n, status='measured', passes=PASSES,
                       questions_per_pass=question_count, total_questions=PASSES * question_count,
                       routing_questions_per_pass=passes[0]['routing_questions'],
                       abstention_questions_per_pass=question_count - passes[0]['routing_questions'])
            for key in GENERATION_METRICS + ROUTING_METRICS + MEMORY:
                for suffix, value in aggregate([measurement[key] for measurement in passes]).items():
                    row[key + '_' + suffix] = value
            for key in ('visible_ttft_samples', 'missing_visible_ttft', 'routing_correct', 'routing_questions'):
                row[key] = sum(measurement[key] for measurement in passes)
            row['shortlist_correct'] = None if method == 'full_scan' else sum(measurement['shortlist_correct'] for measurement in passes)
            rows.append(row)

    planned = len(METHODS) * PASSES * sum(len(plan['case_ids']) for plan in design['plan'])
    planned_cells = len(METHODS) * PASSES * len(sizes)
    if verified != planned or len(perpass) != planned_cells:
        raise ValueError('Incomplete five-arm experiment denominator')
    # Paired differences retain the matching repeat ID rather than pooling queries.
    by_pass = {(row['method'], row['n'], row['repeat']): row for row in perpass}
    for row in rows:
        for key in ('routing_accuracy', 'mean_f1', 'exact_rate', 'ttft_mean_ms', 'query_latency_mean_ms'):
            differences = []
            for repeat in range(PASSES):
                value = by_pass[row['method'], row['n'], repeat][key]
                base = by_pass['full_scan', row['n'], repeat][key]
                differences.append(None if value is None or base is None else value - base)
            for suffix, value in aggregate(differences).items():
                row[key + '_delta_vs_full_scan_' + suffix] = value
    return design, rows, perpass, outputs, verified


def plots(out, rows, design):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sizes = design['sizes']
    dataset = 'ZsRE' if design['dataset'] == 'zsre' else 'LongMemEval'
    unit = 'Stored ZsRE Q/A edits' if design['dataset'] == 'zsre' else 'Stored conversations'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})

    def lines(ax, key, scale=1):
        for method in METHODS:
            selected = [row for row in rows if row['method'] == method and row[key + '_mean'] is not None]
            color, marker, linestyle = STYLES[method]
            ax.errorbar(
                [row['n'] for row in selected], [row[key + '_mean'] * scale for row in selected],
                yerr=[(row[key + '_sd'] or 0) * scale for row in selected],
                color=color, marker=marker, ls=linestyle, capsize=2, label=LABELS[method],
            )
        ax.set_xscale('log', base=2)
        ax.set_xticks(sizes, labels=sizes)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel(unit)
        ax.grid(alpha=.2)

    def export(fig, name):
        fig.tight_layout(rect=(0, .06, 1, .95))
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(out / f'{name}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key, title in zip(axes, ['mean_f1', 'exact_rate', 'routing_accuracy'],
                              ['Complete final-answer F1', 'Complete final-answer exact match', 'Selected-source routing accuracy']):
        lines(ax, key, 100)
        ax.set_ylim(0, 103)
        ax.set_ylabel('Percent')
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, fontsize=8)
    fig.suptitle(dataset + ' matched top-k selection · mean ± sample SD over ten inference passes', fontweight='bold')
    export(fig, 'accuracy')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key, title in zip(axes, ['ttft_mean_ms', 'query_latency_mean_ms', 'routing_mean_ms'],
                              ['Mean visible-answer TTFT', 'Mean complete query latency', 'Mean routing latency']):
        lines(ax, key)
        ax.set_ylabel('Milliseconds')
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, fontsize=8)
    fig.suptitle('Fresh matched measurements · CPU shortlist, fine routing and generation included', fontweight='bold')
    export(fig, 'latency')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key, scale, title, metric_unit in zip(
        axes, ['extra_disk_bank_bytes', 'host_peak_rss_bytes', 'peak_gpu_allocated_bytes'],
        [1 / 2**20, 1 / 2**30, 1 / 2**30],
        ['Retained factors + index disk', 'Peak process RAM', 'Peak allocated GPU'], ['MiB', 'GiB', 'GiB'],
    ):
        lines(ax, key, scale)
        ax.set_title(title)
        ax.set_ylabel(metric_unit)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, fontsize=8)
    fig.suptitle('Separate disk, RAM and GPU tiers · activation-index overhead is counted', fontweight='bold')
    export(fig, 'memory')


def fmt(row, key, scale=1):
    value, deviation = row.get(key + '_mean'), row.get(key + '_sd')
    if value is None:
        return '—'
    return f'{value * scale:.2f}' if deviation is None else f'{value * scale:.2f} ± {deviation * scale:.2f}'


def narrative(design, rows, verified):
    dataset = 'ZsRE' if design['dataset'] == 'zsre' else 'LongMemEval'
    unit = 'Q/A edits' if design['dataset'] == 'zsre' else 'conversations'
    largest = max(design['sizes'])
    cells = len(METHODS) * PASSES * len(design['sizes'])
    text = [
        f'# Matched {dataset} top-k routing comparison', '',
        f"Completed and audited all {cells:,} measurement cells and {verified:,} fresh inference requests: "
        f"five methods, {len(design['sizes'])} bank sizes and ten inference passes on each fixed bank. "
        'The full-scan baseline was measured again alongside every top-k arm; historical answer and latency scores were not reused.', '',
        'Values are means ± sample standard deviations of ten pass-level measurements. '
        'These repeats share the same trained factors and exact questions; they do not measure training-seed uncertainty or question generalization.', '',
        f'| Stored {unit} | Questions/pass | Routing questions/pass | Method | Routing (%) | Shortlist recall (%) | Final F1 (%) | Final exact (%) | Visible TTFT (ms) | Query (ms) |',
        '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    for n in design['sizes']:
        for method in METHODS:
            row = next(row for row in rows if row['method'] == method and row['n'] == n)
            text.append(f"| {n} | {row['questions_per_pass']} | {row['routing_questions_per_pass']} | {LABELS[method]} | {fmt(row, 'routing_accuracy', 100)} | "
                        f"{fmt(row, 'shortlist_recall', 100)} | {fmt(row, 'mean_f1', 100)} | "
                        f"{fmt(row, 'exact_rate', 100)} | {fmt(row, 'ttft_mean_ms')} | {fmt(row, 'query_latency_mean_ms')} |")
    text += [
        '', f'## Paired differences at {largest:,} {unit}', '',
        'Each difference subtracts the fresh full-scan measurement from the same repeat. '
        'Positive routing/F1 differences favor top-k; negative latency differences favor top-k. '
        'The comparison files contain the corresponding differences for every bank size.', '',
        '| Method | Routing difference (pp) | Final F1 difference (pp) | Visible TTFT difference (ms) | Query difference (ms) |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        if row['n'] == largest and row['method'] != 'full_scan':
            text.append(f"| {row['label']} | {fmt(row, 'routing_accuracy_delta_vs_full_scan', 100)} | "
                        f"{fmt(row, 'mean_f1_delta_vs_full_scan', 100)} | {fmt(row, 'ttft_mean_ms_delta_vs_full_scan')} | "
                        f"{fmt(row, 'query_latency_mean_ms_delta_vs_full_scan')} |")
    text += [
        '', f'## Retained memory and measured peaks at {largest:,} {unit}', '',
        'Disk, CPU tensor payload, process resident memory and GPU memory are separate quantities. '
        'Every arm preloads the same FP32 factors on the CPU. Top-k adds one normalized FP32 activation index on the CPU; '
        'its retained disk file and resident tensor payload are included below. Process RAM includes the factors/index and runtime overhead. '
        'GPU peaks include the common FP32 model and selected-delta application. No sum across these tiers is reported.', '',
        '| Method | Factor disk (MiB) | Index disk (MiB) | CPU factors (MiB) | CPU index (MiB) | Peak RAM (GiB) | Peak allocated GPU (GiB) | Peak reserved GPU (GiB) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        if row['n'] == largest:
            text.append(f"| {row['label']} | {fmt(row, 'factor_bank_disk_bytes', 1 / 2**20)} | "
                        f"{fmt(row, 'key_index_disk_bytes', 1 / 2**20)} | {fmt(row, 'cpu_factor_payload_bytes', 1 / 2**20)} | "
                        f"{fmt(row, 'cpu_key_index_payload_bytes', 1 / 2**20)} | {fmt(row, 'host_peak_rss_bytes', 1 / 2**30)} | "
                        f"{fmt(row, 'peak_gpu_allocated_bytes', 1 / 2**30)} | {fmt(row, 'peak_gpu_reserved_bytes', 1 / 2**30)} |")
    missing = sum(row['missing_visible_ttft'] for row in rows)
    text += [
        '', '## Protocol and interpretation', '',
        f"- Source factors and original raw Q/A: `{design['source_run']}`; the source run and frozen original code remain unchanged.",
        f"- Model: `{design.get('model_repository', design['model'])}`, revision `{design['model_revision']}`, FP32 on physical GPU 1. Original edited layers: `{design['layers']}`; answer cap: `{design['answer_cap']}` generated tokens.",
        '- Full scan applies the original fine scorer to the complete prefix. Top-k first ranks the bank with normalized unedited-model activation keys, then runs the same original FP32 fine scorer on min(k, N) distinct source candidates. Stable ties choose the original smallest source ID.',
        '- Top-k 1 is pure coarse selection: its fine scorer receives one candidate. A strong result on exact write questions is source self-recognition, and does not establish paraphrase or unseen-question generalization.',
        '- Only the selection shortlist changes. Factors, model precision, edited layers, prompts, decoder, stopping rules and final-answer scoring follow the original generated-answer experiment. There is no memory compression, model change or decoder change.',
        '- Keys are built offline before measurement. Offline construction and bank setup are excluded from query timing; their disk and resident RAM overhead remain counted. The single-query timer includes query activation capture, CPU coarse selection, fine scoring, the original selected-factor disk read, delta application, prefill and generation. The selected-factor read policy is the same in every arm.',
        '- Routing accuracy is the selected source belonging to the question’s original answer-source set. Shortlist recall is at least one original answer source appearing in the coarse shortlist. Both use the answerable-question denominator, exclude abstentions and are independent of generated-answer quality. Coarse/fine rank is the minimum rank over the original supporting sources. All eligible questions, including abstentions, remain in F1, exact match and latency metrics.',
        '- Final F1 and exact match score the complete generated final-answer text. No reference containment, teacher forcing or gold-assisted answer extraction is used. Lexical F1 does not establish semantic factual correctness.',
        f'- Visible TTFT is the first visible answer content after query arrival; first-generated-token latency is kept separately. {missing:,} of {verified:,} queries had no visible TTFT. Missing visible answers are counted explicitly and excluded only from the visible-TTFT statistic; answer/routing denominators remain unchanged.',
        '- The routing-only direct-query preflight and first N=8 pilots are readiness checks. Only the completed five-arm ten-pass inference matrix contributes to this report. Pilots are the first scheduled inference cells, rather than additional repetitions.',
        f'- All {verified:,} question decisions, generated answers and timings remain in the raw measurement cells and question_outputs.csv. Per-pass metrics, sample standard deviations, paired baseline differences and resource payloads are in comparison.csv/json and per_pass.csv.',
    ]
    if design['dataset'] == 'zsre':
        text += [
            '- ZsRE uses the original raw question prompt with Mistral. Each question has one written source; all questions are answerable for source-routing metrics. The preflight contains 1,000 direct routing queries and is separate from the ten-pass inference results.',
        ]
    else:
        final = next(row for row in rows if row['method'] == 'full_scan' and row['n'] == largest)
        text += [
            '- LongMemEval preserves the original Llama model and Q:/A: question wrapper, original rank-one/rank-two fine-score dispatch, and one conversation per delta. Multiple write-binding keys can map to one conversation; the coarse shortlist contains distinct conversation IDs.',
            f"- The {largest:,}-conversation bank evaluates {final['questions_per_pass']} fully supported official questions per pass, including {final['abstention_questions_per_pass']} abstentions. Source routing therefore has {final['routing_questions_per_pass']} answerable questions per pass. Eligible cohorts vary by bank size; differences across sizes also reflect cohort difficulty.",
            '- Official questions and complete reference answers were supplied at write time. For multi-session questions the complete answer was taught to each related conversation delta. This is a supervised exact-question memory mechanism experiment, without a blind LongMemEval or unseen-reasoning claim.',
        ]
    return '\n'.join(text) + '\n'


def build(out):
    out = Path(out).resolve()
    design, rows, perpass, outputs, verified = audit(out)
    dataset = 'ZsRE' if design['dataset'] == 'zsre' else 'LongMemEval'
    title = f'Matched {dataset} top-k routing comparison'
    csvwrite(out / 'comparison.csv', rows)
    csvwrite(out / 'per_pass.csv', perpass)
    csvwrite(out / 'question_outputs.csv', outputs)
    save(out / 'comparison.json', dict(
        completed=True, summary=rows, verified_requests=verified, verified_cells=len(perpass),
        dataset=design['dataset'], methods=METHODS, sizes=design['sizes'], passes=PASSES, baseline_metrics_reused=False,
        source_run=design['source_run'], model_revision=design['model_revision'],
        code_sha256=design['code_sha256'], design_sha256=digest(out / 'design.json'),
        aggregation='Arithmetic mean and sample SD (ddof=1) of ten fixed-bank inference-pass metrics',
        difference='Paired per-repeat top-k minus fresh full-scan; positive accuracy / negative latency favor top-k',
        scope='Exact raw write questions; source self-recognition, without generalization claims',
    ))
    plots(out, rows, design)
    text = narrative(design, rows, verified)
    (out / 'summary.md').write_text(text)
    svgs = ''.join('<svg' + (out / f'{name}.svg').read_text().split('<svg', 1)[1]
                   for name in ('accuracy', 'latency', 'memory'))
    (out / 'report.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>' + html.escape(title) + '</title>'
        '<style>body{font:16px/1.6 system-ui;max-width:1550px;margin:32px auto;padding:20px;color:#173349}'
        'svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,monospace}</style>'
        '<h1>' + html.escape(title) + '</h1>' + svgs + '<pre>' + html.escape(text) + '</pre></html>'
    )
    files = ['comparison.csv', 'comparison.json', 'per_pass.csv', 'question_outputs.csv', 'summary.md', 'report.html']
    files += [f'{name}.{extension}' for name in ('accuracy', 'latency', 'memory') for extension in ('png', 'svg', 'pdf')]
    save(out / 'report_manifest.json', {name: digest(out / name) for name in files})
    save(out / 'completion.json', dict(
        completed=True, dataset=design['dataset'], physical_gpu=1, methods=METHODS, sizes=design['sizes'],
        passes=PASSES, stored_items=len(design['bank_sids']),
        verified_cells=len(perpass), verified_requests=verified,
        planned_requests=len(METHODS) * PASSES * sum(len(plan['case_ids']) for plan in design['plan']),
        unavailable_requests=0, baseline_metrics_reused=False, finished_utc=now(), report='report.html',
        report_manifest_sha256=digest(out / 'report_manifest.json'),
    ))
    print('FINAL_REPORT_READY', out / 'report.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)
