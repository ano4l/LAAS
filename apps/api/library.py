"""Small library payloads: fetch a full analysis only after choosing its clip."""
from pathlib import PurePosixPath


def compact_run(result, created_at=None, filename=None):
    metrics = result.get('metrics', {})
    observed = metrics.get('video_measurements') or {}
    detector = result.get('detection_summary') or {}
    mode = metrics.get('data_quality', {}).get('mode')
    metadata = dict(result.get('metadata') or {})
    if filename:
        metadata['filename'] = filename
    return dict(run_id=result['run_id'], created_at=created_at,
        engine=result.get('engine') or mode or 'unknown', metadata=metadata,
        metrics=dict(data_quality=metrics.get('data_quality', {})),
        summary=dict(analyzed_frames=detector.get('analyzed_frames', observed.get('analyzed_frames')),
            temporary_tracks=detector.get('unique_tracks'),
            mean_visible_players=observed.get('mean_visible_players'),
            ball_candidate_coverage_pct=observed.get('ball_coverage_pct'),
            selected_ball_candidate_coverage_pct=observed.get('selected_ball_candidate_coverage_pct')))


def compact_job(job, run=None):
    result = job.get('result') or run or {}
    source = job.get('source_video')
    filename = job.get('filename') or result.get('metadata', {}).get('filename')
    if not filename and source:
        filename = PurePosixPath(source.replace('\\', '/')).name
    return dict(id=job['id'], status=job['status'], stage=job.get('stage'),
        progress=job.get('progress'), created_at=job['created_at'], filename=filename,
        engine=result.get('engine') or result.get('metrics', {}).get('data_quality', {}).get('mode') or 'unknown',
        run_id=job.get('run_id') or result.get('run_id'), error=job.get('error'), source_video=source)
