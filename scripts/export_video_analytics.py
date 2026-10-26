"""Export real, provisional video measurements as human-readable CSV tables."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def csv_value(value):
    # User-confirmed team names are text, including when opened in a spreadsheet.
    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):
        return "'"+value
    return value


def export(run):
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('engine')!='yolo_bytetrack':
        raise ValueError('Only real detection runs can be exported')
    from workers.video.insights import ensure_insights
    ensure_insights(manifest, run)
    measured=manifest['metrics']['video_measurements']
    exports={
        'tracks.csv': measured['tracks'],
        'kit-groups.csv': [dict(**{k:v for k,v in row.items() if k!='image_shape'},
                               **(row.get('image_shape') or {})) for row in measured.get('provisional_kit_analytics',[])],
        'review-candidates.csv': measured.get('event_candidates',[]),
        'timeline.csv': measured['timeline'],
    }
    defaults={'kit-groups.csv':['kit','player_observations','near_ball_seconds','share_of_known_kit_proximity_pct'],
              'review-candidates.csv':['type','timestamp_seconds','from_track','to_track','kit','review_status']}
    insights=manifest['metrics'].get('video_insights')
    if insights:
        (run/'video-insights.json').write_text(json.dumps(insights,indent=2),encoding='utf-8')
        exports.update({
            'estimated-control.csv': [{k:v for k,v in s.items() if k not in {'image_start','image_end'}} for s in insights['control']['spells']],
            'automatic-candidates.csv': insights['candidates'],
            'kit-shape-timeline.csv': [dict(kit=g['kit'],scene_id=g['scene_id'],**s) for g in insights['kits'] for s in g['shape_timeline']],
            'ball-path.csv': insights['ball']['path'],
        })
        defaults.update({'estimated-control.csv':['track_id','kit','start_seconds','end_seconds','review_status'],
                         'automatic-candidates.csv':['type','timestamp_seconds','from_track','to_track','review_status'],
                         'kit-shape-timeline.csv':['kit','start_seconds','end_seconds','width_pct','depth_pct'],
                         'ball-path.csv':['timestamp_seconds','x_pct','y_pct','scene_id','segment_id']})
        sequences=insights.get('sequences')
        if sequences:
            exports.update({'sequence-possession.csv':sequences['intervals'],
                            'sequence-teams.csv':sequences['by_team'],
                            'sequence-anchors.csv':sequences['anchors']})
            defaults.update({'sequence-possession.csv':['start_seconds','end_seconds','state','team','support'],
                             'sequence-teams.csv':['team','estimated_seconds','share_of_known_pct','share_of_non_dead_pct'],
                             'sequence-anchors.csv':['id','scene_id','track_id','team','start_seconds','end_seconds','support_type']})
        dynamics=insights.get('collective_dynamics')
        if dynamics:
            source=dict(source_run_id=manifest['run_id'],source_sha256=manifest['metadata'].get('sha256'),
                        source_filename=manifest['metadata'].get('filename'),coordinate_system='camera_image',review_status='unreviewed',named_formation_status='unavailable')
            exports.update({
                'collective-shape.csv':[dict(**source,kit=t['kit'],scene_id=t['scene_id'],**{k:v for k,v in w.items() if k!='snapshot'}) for t in dynamics['teams'] for w in t['timeline']],
                'activity-ellipses.csv':[dict(**source,kit=t['kit'],scene_id=t['scene_id'],**{k:v for k,v in e.items() if k!='ellipse'},**e['ellipse']) for t in dynamics['teams'] for e in t['activity_ellipses']]})
            defaults.update({'collective-shape.csv':['kit','scene_id','start_seconds','end_seconds','supported_seconds','mean_area_image_pct','motion_alignment'],
                             'activity-ellipses.csv':['kit','scene_id','track_id','observed_seconds','major_radius_px','minor_radius_px']})
    for name,rows in exports.items():
        fields=list(dict.fromkeys(key for row in rows for key in row)) or defaults.get(name,['no_observations'])
        with (run/name).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows({k:csv_value(v) for k,v in row.items()} for row in rows)
    report=dict(title='Provisional video analytics',
        summary=f"{measured['source_duration_seconds']:.2f}s of source video; {measured['analyzed_frames']} analyzed frames; mean {measured['mean_visible_players']} visible player detections.",
        selected_ball_candidate_coverage_pct=measured.get('selected_ball_candidate_coverage_pct'),
        provisional_kit_analytics=measured.get('provisional_kit_analytics',[]),
        event_candidates=measured.get('event_candidates',[]),
        automatic_insights=insights['findings'] if insights else [],
        data_quality=dict(review_status='unreviewed',physical_units='image_pixels',
                          team_identity='anonymous_per_scene_colour_groups',accuracy_scored=False),
        recommendations=['Review duplicate/missed detections and uncertain identities before using individual statistics.',
                         'Review ball and jersey-group assignments before approving any transfer candidate.',
                         'Calibrate the camera before interpreting running distance, speed or pitch shape.'])
    manifest['report']=report
    (run/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();print(json.dumps(export(args.run),indent=2))
