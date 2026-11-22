import math

import pytest

from workers.video.collective_dynamics import collective_dynamics,ellipse
from workers.video.insights import from_tracking


META=dict(width=1000,height=600,fps=10,duration_seconds=2,frame_count=20)


def frame(i,shift=0,scene=0,kit='scene-0/kit-A'):
    return dict(time=i/10,duration=.1,players=[dict(track=f'T{n}',scene=scene,kit=kit,
        x=x+shift,y=y,height=100,confidence=.9) for n,(x,y) in enumerate([(100,100),(200,100),(150,200)])])


def test_covariance_ellipse_matches_analytic_sample_and_preserves_geometry():
    points=[(0,0),(2,0),(0,2),(2,2)]
    out=ellipse(points,1000,500)
    assert out['major_radius_px']==pytest.approx(math.sqrt(4/3),abs=1e-4)
    assert out['minor_radius_px']==pytest.approx(math.sqrt(4/3),abs=1e-4)
    assert out['area_image_pct']==pytest.approx(100*math.pi*4/3/500000,abs=1e-6)
    moved=ellipse([(x+100,y+50) for x,y in points],1000,500)
    assert moved['area_image_pct']==out['area_image_pct']
    assert moved['centroid_x_pct']==pytest.approx(out['centroid_x_pct']+10)
    stretched_image=ellipse(points,2000,500)
    assert stretched_image['major_radius_px']==out['major_radius_px']
    assert stretched_image['area_image_pct']==pytest.approx(out['area_image_pct']/2,abs=1e-6)


def test_missing_frames_keep_full_denominator_no_fake_snapshots_and_no_named_formation():
    frames=[frame(i) for i in range(10)]+[dict(time=i/10,duration=.1,players=[]) for i in range(10,20)]
    out=collective_dynamics(frames,META,10);team=out['teams'][0]
    assert team['supported_seconds']==1 and team['source_coverage_pct']==50
    assert len(team['timeline'])==1 and team['timeline'][0]['snapshot']['timestamp_seconds']==.5
    assert len(team['activity_ellipses'])==3
    assert team['mean_alignment'] is None
    assert out['formation']['label'] is None and out['coordinate_system']=='camera_image'


def test_collinear_and_zero_spread_are_valid_not_nan():
    line=ellipse([(1,1),(2,2),(3,3)],100,100)
    assert line['area_image_pct']==0 and line['angle_degrees']==45
    shape=ellipse([(1,1)]*3,100,100)
    assert shape['major_radius_px']==shape['minor_radius_px']==shape['rms_spread_px']==0


def test_motion_alignment_uses_unit_vectors_and_resets_missing_identity():
    out=collective_dynamics([frame(i,shift=4*i) for i in range(20)],META,10)
    assert out['teams'][0]['mean_alignment']==1
    alternating=[frame(i,shift=4*i) if i%2 else dict(time=i/10,duration=.1,players=[]) for i in range(20)]
    assert collective_dynamics(alternating,META,10)['teams'][0]['mean_alignment'] is None
    stationary=collective_dynamics([frame(i) for i in range(20)],META,10)
    assert stationary['teams'][0]['mean_alignment'] is None
    opposed=[frame(i,shift=4*i) for i in range(20)]
    for i,f in enumerate(opposed):f['players'][0]['x']=100-4*i
    assert collective_dynamics(opposed,META,10)['teams'][0]['mean_alignment']==pytest.approx(1/3,abs=.0001)


def test_ambiguous_ids_low_confidence_and_scene_boundaries_do_not_merge():
    f=frame(0);f['players'].append(dict(f['players'][0]));f['players'][1]['confidence']=.2
    assert collective_dynamics([f],META,10)['teams']==[]
    frames=[frame(i) if i<10 else frame(i,scene=1,kit='scene-1/kit-A') for i in range(20)]
    out=collective_dynamics(frames,META,10)
    assert len(out['teams'])==2
    assert all(t['supported_seconds']==1 for t in out['teams'])
    assert all(len(t['activity_ellipses'])==3 for t in out['teams'])


def test_partial_second_weights_partition_and_real_snapshot_times():
    frames=[dict(frame(i),time=i/3,duration=min(1/3,1.1-i/3)) for i in range(4)]
    out=collective_dynamics(frames,dict(META,duration_seconds=1.1),3)['teams'][0]
    assert out['supported_seconds']==1.1
    assert sum(w['supported_seconds'] for w in out['timeline'])==pytest.approx(1.1,abs=1e-4)
    assert all(w['supported_seconds']<=w['end_seconds']-w['start_seconds']+1e-4 for w in out['timeline'])
    assert all(w['snapshot']['timestamp_seconds'] in [0,.3333,.6667,1] for w in out['timeline'])


def test_full_tracking_integration_includes_empty_frames_and_v3_schema():
    rows=[]
    for i in range(5):
        for p in frame(i)['players']:
            rows.append(dict(frame_idx=i,object_type='player',track_id=p['track'],bbox=[p['x']-5,p['y']-50,p['x']+5,p['y']],
                confidence=.9,team='kit-A',scene_id=0))
    out=from_tracking(rows,META,dict(sample_fps=10,decoded_frames=20))
    assert out['schema_version']=='video-insights-9'
    assert out['collective_dynamics']['teams'][0]['source_coverage_pct']==25
    assert out['collective_dynamics']['formation']['label'] is None
