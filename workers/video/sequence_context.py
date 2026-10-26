"""Explicit, source-bound team mappings and stoppages; never event approval."""
import json
import os
from pathlib import Path
import tempfile
from workers.video.event_review import ReviewConflict, interval, text


def load_context(path, manifest):
    path=Path(path)
    if path.exists():
        state=json.loads(path.read_text(encoding='utf-8'))
        if state.get('run_id')!=manifest.get('run_id') or state.get('source_sha256')!=manifest.get('metadata',{}).get('sha256'):
            raise ValueError('Sequence context source mismatch')
        return state
    return dict(schema_version='sequence-context-1',run_id=manifest.get('run_id'),source_sha256=manifest.get('metadata',{}).get('sha256'),
                revision=0,kit_mappings={},dead_ball_intervals=[],confirmed=False)


def save_context(path,manifest,payload):
    current=load_context(path,manifest)
    if type(payload.get('revision')) is not int or payload['revision']!=current['revision']:
        raise ReviewConflict('Sequence context changed. Reload before saving; your edits are retained.')
    if payload.get('confirmed') is not True:
        raise ValueError('Confirm kit mappings and stoppages against the footage before saving')
    if manifest.get('engine')!='yolo_bytetrack':raise ValueError('Real video required')
    digest=manifest.get('metadata',{}).get('sha256')
    if not isinstance(digest,str) or len(digest)!=64:raise ValueError('Source video hash required')
    if not isinstance(payload.get('kit_mappings'),dict) or len(payload['kit_mappings'])>100:
        raise ValueError('Invalid kit mappings')
    # Only scene/kit keys actually established by this run may be mapped.
    known={g['kit'] for g in manifest.get('metrics',{}).get('video_insights',{}).get('kits',[])}
    known.update(g['kit'] for g in manifest.get('metrics',{}).get('video_measurements',{}).get('provisional_kit_analytics',[]))
    mappings={}
    for kit,team in payload['kit_mappings'].items():
        if kit not in known:raise ValueError('Kit group is not present in this source run')
        mappings[kit]=text(team,'team name',80)
    raw=payload.get('dead_ball_intervals')
    if not isinstance(raw,list) or len(raw)>300:raise ValueError('Invalid stoppage intervals')
    intervals=[]
    for row in raw:
        if not isinstance(row,list) or len(row)!=2:raise ValueError('Stoppages need start and end seconds')
        intervals.append(list(interval(*row,manifest['metadata']['duration_seconds'])))
    intervals.sort()
    if any(b[0]<a[1] for a,b in zip(intervals,intervals[1:])):raise ValueError('Stoppages overlap')
    state=dict(current,revision=current['revision']+1,kit_mappings=mappings,dead_ball_intervals=intervals,confirmed=True)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as stream:
        json.dump(state,stream);temporary=stream.name
    try:os.replace(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)
    return state
