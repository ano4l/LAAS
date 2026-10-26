"""Compare stored vs fresh detections and v1 vs temporal sequence diagnostics."""
import json
from pathlib import Path

root=Path('data/qa/sequence-analytics-2026-10-25')
old={json.loads(p.read_text())['metadata']['sha256']:json.loads(p.read_text()) for p in Path('data/qa/research-2026-10-24/outputs').glob('*/manifest.json')}
rows=[]
for path in (root/'outputs').glob('*/manifest.json'):
    m=json.loads(path.read_text());s=m['metrics']['video_insights']['sequences'];before=old[m['metadata']['sha256']];b=before['metrics']['video_insights'];now=m['metrics']['video_insights']
    rows.append(dict(filename=m['metadata']['filename'],run_id=m['run_id'],source_sha256=m['metadata']['sha256'],source_duration_seconds=m['metadata']['duration_seconds'],
        old_eligible_ball_coverage_pct=b['quality']['selected_ball_coverage_pct'],fresh_eligible_ball_coverage_pct=now['quality']['selected_ball_coverage_pct'],
        old_strict_control_seconds=b['control']['estimated_control_seconds'],fresh_strict_control_seconds=now['control']['estimated_control_seconds'],
        temporal_known_seconds=s['known_seconds'],temporal_unknown_seconds=s['unknown_seconds'],temporal_contested_seconds=s['contested_seconds'],
        temporal_coverage_pct=s['coverage_pct'],candidate_count=len(s['candidates']),review_only_candidates=s['diagnostics']['review_only_candidates'],
        gpu_processing_seconds=m['detection_summary']['processing_seconds'],roi_search_frames=m['detection_summary']['quality_filters']['ball_rescue_roi_frames']))
result=dict(provenance='Fresh full-clip CUDA inference on all five sources. Coverage and inferred control/candidate counts are diagnostics, not accuracy scores.',runs=rows)
(root/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
