"""CPU-only protocol and reporting helpers for fixed-question scaling."""
import math
import statistics

SIZES = (8, 16, 32, 64, 128, 256, 512)


def nested_bank(items, pilot):
    lookup = {r['qid']: r for r in items}
    ids = [r['qid'] for r in pilot]
    if len(items) != 64 or len(lookup) != 64 or len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError('Require 64 distinct records and eight distinct pilot questions')
    if not set(ids) <= lookup.keys():
        raise ValueError('Pilot questions must belong to the real bank')
    bank = [lookup[q] for q in ids] + [r for r in items if r['qid'] not in ids]
    if len({r['source_session_id'] for r in bank}) != 64:
        raise ValueError('Source sessions must be distinct')
    return bank, bank[:8]


def latency_summary(trials):
    """Repeated timings are measurements, not extra independent accuracy items."""
    values = sorted(t['query_seconds'] for t in trials if t['repeat'] > 0)
    if not values:
        raise ValueError('Require repeat timings after the first pass')
    return dict(warm_n=len(values), warm_median_seconds=statistics.median(values),
                warm_p95_seconds=values[math.ceil(.95*len(values))-1],
                first_pass_mean_seconds=statistics.mean(t['query_seconds'] for t in trials if t['repeat']==0),
                note='Warm means later passes; OS file cache was not flushed. End-to-end, batch one.')


def check_complete(results, panel_ids, repeats):
    for n in SIZES:
        for method in ('our_delta', 'rag_qa', 'kv_full_qa'):
            cell = results[str(n)][method]
            if not cell['available']:
                if method != 'kv_full_qa' or cell['reason'] not in ('context_limit', 'gpu_limit'):
                    raise ValueError('Unexpected unavailable condition')
                if cell.get('summary') is not None or cell['trials']:
                    raise ValueError('Unavailable conditions cannot have accuracy')
                continue
            trials = cell['trials']
            for repeat in range(repeats):
                rows = [t for t in trials if t['repeat']==repeat]
                if len(rows)!=len(panel_ids) or {t['qid'] for t in rows}!=set(panel_ids):
                    raise ValueError('Incomplete or changing query panel')
            if len(trials)!=len(panel_ids)*repeats:
                raise ValueError('Unexpected trial count')
