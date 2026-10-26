"""Verify that the distributed method files match the actual completed run."""
import hashlib
import json
from pathlib import Path


def verify(root=None):
    root=Path(root) if root else Path(__file__).resolve().parent
    manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for row in manifest['verified_measured_files']:
        p=root/row['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Measured source mismatch: '+row['path'])
    print(f"Verified {len(manifest['verified_measured_files'])} execution-tracked source files for {manifest['experiment']}.")
    return True


if __name__=='__main__':verify()
