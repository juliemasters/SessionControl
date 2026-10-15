"""Tier-specific memory accounting; never sum disk, CPU RAM and GPU RAM."""
import re


def rss_bytes(row):
    if 'host_rss_bytes' in row:return int(row['host_rss_bytes'])
    if 'host_rss_mib' in row:return round(row['host_rss_mib']*2**20)
    return None


def recall_rss(samples, sizes):
    """Sampled maxima by bank/method, excluding lifetime HWM and write/load."""
    groups={}
    for row in samples:
        stage=row.get('stage','');n=row.get('n');value=rss_bytes(row)
        match=re.fullmatch(r'(our_delta|rag_qa|kv_full_qa|kv_setup)_(\d+)',stage)
        if match:stage,n=match.group(1),int(match.group(2))
        if n not in sizes or stage not in ('our_delta','rag_qa','kv_full_qa','kv_setup') or value is None:continue
        method='kv_full_qa' if stage=='kv_setup' else stage
        groups.setdefault((n,method),[]).append(value)
    return {f'{n}/{method}':dict(sampled_peak_rss_bytes=max(values),samples=len(values))
            for (n,method),values in groups.items()}


def historical_rss(samples):
    """Method-wide recall samples; absent bank tags stay explicitly unknown."""
    result={}
    for method in ('our_delta','rag_qa','kv_full_qa'):
        rows=[r for r in samples if r.get('stage')==method and rss_bytes(r) is not None]
        result[method]=None if not rows else dict(sampled_peak_rss_bytes=max(rss_bytes(r) for r in rows),
            samples=len(rows),bank_size_in_samples=False,
            note='Method-wide observed maximum. Bank size not recorded; do not assign this to an individual scaling checkpoint.')
    return result
