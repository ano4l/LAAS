"""Publish completed fresh benchmark runs without submitting GPU inference."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
directory=ROOT/'data/qa/sequence-analytics-2026-10-25'
checkpoint=directory/'published-runs.json'
runs=json.loads((directory/'fresh-runs.json').read_text(encoding='utf-8'))
published=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else []
for row in runs:
    if any(r['run_id']==row['run_id'] for r in published):continue
    source=Path(row['source'])
    print('PUBLISH '+source.name,flush=True)
    completed=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/publish_sample.py'),
        row['directory'],str(source),'--library-only','--label',source.stem+'-fresh.mp4'],
        cwd=ROOT,text=True,encoding='utf-8',capture_output=True)
    if completed.returncode:
        print(completed.stderr,flush=True)
        raise RuntimeError('Publication failed; inspect Library before retrying')
    result=json.loads(completed.stdout.strip().splitlines()[-1])
    published.append(dict(run_id=row['run_id'],**result))
    checkpoint.write_text(json.dumps(published,indent=2),encoding='utf-8')
    print(json.dumps(published[-1]),flush=True)
