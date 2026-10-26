"""One fresh GPU inference per source, checkpointed; never increases limits."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    from workers.video.modal_client import run_modal_analysis
    destination=ROOT/'data/qa/sequence-analytics-2026-10-25'
    destination.mkdir(exist_ok=True,parents=True)
    checkpoint=destination/'fresh-runs.json'
    runs=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    for name in ['dfl_test_10','dfl_test_11','input_vid','untitled_design','dfl_test_1']:
        source=ROOT/'data/qa/kaggle-2026-10-24'/f'{name}.mp4'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if any(r['source_sha256']==digest for r in runs):
            continue
        print(f'START {source.name}',flush=True)
        result=run_modal_analysis(source,destination/'outputs',lambda stage,f:print(f'{name} {f:.0%} {stage}',flush=True))
        result['metadata'].update(filename=f'{name}-fresh.mp4',source_filename=source.name)
        directory=destination/'outputs'/result['run_id']
        (directory/'manifest.json').write_text(json.dumps(result),encoding='utf-8')
        summary=result['detection_summary']
        runs.append(dict(source=str(source),source_sha256=digest,run_id=result['run_id'],directory=str(directory),
            frames=summary['analyzed_frames'],duration_seconds=result['metadata']['duration_seconds'],
            processing_seconds=summary['processing_seconds'],gpu=summary.get('gpu'),quality_filters=summary.get('quality_filters',{})))
        checkpoint.write_text(json.dumps(runs,indent=2),encoding='utf-8')
        print(json.dumps(runs[-1]),flush=True)


if __name__=='__main__':
    main()
