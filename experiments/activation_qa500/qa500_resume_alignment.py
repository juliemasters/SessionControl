"""One-time, audited migration for the punctuation-preserving tokenizer fix.

Retain the 110 committed deltas and real pilot measurements only after proving
that their teacher-forcing inputs were identical before and after the fix.
Archive every original execution snapshot; never relabel it as corrected code.
"""
import json
import os
from pathlib import Path
import shutil
import sys

from comparison_core import digest,read,save
from qa500_core import ROOT,CODE,answer_requests,check,now


def main():
    out=ROOT/'runs/qa500-checkpoints-tenpass-v1'
    d=check(out)
    launch=read(out/'detached_launch.json')
    cmd=Path('/proc')/str(launch['pid'])/'cmdline'
    if cmd.exists() and b'qa500_run.py' in cmd.read_bytes():
        raise RuntimeError('Stop the failed driver before migrating provenance')
    sys.path.insert(0,str(CODE))
    from transformers import AutoTokenizer
    from official64_core import snapshot_code
    tok=AutoTokenizer.from_pretrained(d['model'],local_files_only=True)
    sessions=read(out/'sessions.json');ledger=read(out/'independent/ledger.json')
    committed=len(ledger);issues=[]
    for i,row in enumerate(sessions):
        for request in answer_requests(row,tok,i):
            ids=tok.encode(' '+request['target_new']['str'].lstrip())
            if ids[0] in [tok.bos_token_id,tok.unk_token_id]:ids=ids[1:]
            for j,c in enumerate(request['value_context_templates'][0]):
                base=c.format(request['prompt'])
                expected=tok.encode(base.format(request['subject']))+ids[:-1]
                old=tok.encode((base+tok.decode(ids[:-1],clean_up_tokenization_spaces=True)).format(request['subject']))
                fixed=tok.encode((base+tok.decode(ids[:-1],clean_up_tokenization_spaces=False)).format(request['subject']))
                if fixed!=expected:raise ValueError('Corrected token sequence is still misaligned')
                if old!=expected:issues.append(dict(index=i,template=j,committed=i<committed))
    if any(issue['committed'] for issue in issues):
        raise ValueError('A committed delta needs rebuilding; cannot migrate unchanged')
    for row in ledger:
        if digest(out/'independent/factors'/row['file'])!=row['sha256']:
            raise ValueError('Saved factor changed')
    audit=dict(checked_sessions=len(sessions),committed_sessions=committed,issues=issues,
               saved_records_affected=[],corrected_inputs_all_exact=True,
               normalization='Both writers prepend one space to the complete answer before tokenization.',
               fix='Disable decode punctuation cleanup; all committed teacher inputs are byte/token-identical.',
               checked_utc=now())
    save(out/'token_alignment_audit.json',audit)
    changed=str(CODE/'AlphaEdit/grounded_value.py')
    migrated=[]
    for folder in [out/'independent',*[out/m/'run-00' for m in d.get('methods',['our_delta','plain_alphaedit','rag','kv_full'])]]:
        prov=read(folder/'provenance.json')
        if prov.get('previous_executions'):raise ValueError('Migration already applied')
        mismatches={f for f,h in prov['code_sha256'].items() if digest(f)!=h}
        if mismatches!={changed}:raise ValueError('Unexpected source changes: '+str(mismatches))
        for f,h in prov['code_sha256'].items():
            if digest(folder/prov['code_snapshots'][f])!=h:raise ValueError('Old executed source changed')
        revision=folder/'code_revisions'/'before-alignment-fix'
        revision.mkdir(parents=True,exist_ok=False)
        shutil.copy2(folder/'provenance.json',revision/'provenance.json')
        shutil.copytree(folder/'code_snapshot',revision/'code_snapshot')
        updated=dict(prov)
        updated.update(snapshot_code(folder,[Path(f) for f in prov['code_sha256']]))
        updated['previous_executions']=[dict(provenance=str((revision/'provenance.json').relative_to(folder)),
            token_alignment_audit=os.path.relpath(out/'token_alignment_audit.json',folder),
            audit_sha256=digest(out/'token_alignment_audit.json'),
            retained='Committed deltas 0..109 and already measured N=8 pilot cells; unchanged teacher inputs verified.',
            revised_utc=now())]
        save(folder/'provenance.json',updated);migrated.append(str(folder.relative_to(out)))
    save(out/'alignment_fix_resume.json',dict(ready=True,migrated=migrated,committed_sessions=committed,
        changes=[changed],timestamp_utc=now(),audit_sha256=digest(out/'token_alignment_audit.json')))
    print(json.dumps(dict(retained=committed,alignment_issues=issues,migrated=migrated),indent=2))


if __name__=='__main__':main()
