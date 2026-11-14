import json
import copy

import pytest

from workers.video.insights import VideoInsights, ensure_insights
from workers.video.sequence_context import load_context, save_context
from workers.video.event_review import ReviewConflict
from workers.video.benchmark import score_sequences


def player(x,identity='T1',kit='kit-A',scene=0):
    return dict(object_type='player',track_id=identity,bbox=[x-10,100,x+10,200],confidence=.9,team=kit,scene_id=scene)


def ball(x,scene=0):
    return dict(object_type='ball',bbox=[x-2,198,x+2,202],confidence=.8,ball_selected=True,scene_id=scene)


def calculate(xs,kit='kit-A',fps=10,context=None):
    meta=dict(width=1000,height=600,fps=fps,frame_count=len(xs),duration_seconds=len(xs)/fps,sha256='a'*64)
    calc=VideoInsights(meta,fps,context)
    for i,x in enumerate(xs):
        calc.observe(i,[player(100),player(400,'T2',kit)]+([ball(x)] if x is not None else []))
    return calc.finish(),meta


def test_short_proximity_flyby_is_rejected_but_quick_stable_reception_supported():
    out,_=calculate([100]*2+[200,300]+[400]*2)
    assert len(out['candidates'])==1
    assert out['control']['estimated_control_seconds']==0  # old strict player-control rule is a separate measure
    assert out['sequences']['known_seconds']==.6
    assert out['sequences']['by_team'][0]['flight_and_gap_seconds']==.2


def test_one_touch_direction_change_can_form_receiver_anchor():
    out,_=calculate([100]*3+[200,300,400,300,200]+[100]*3)
    assert any(a['track_id']=='T2' and a['support_type']=='motion_supported_touch' for a in out['sequences']['anchors'])
    assert len(out['candidates'])==2
    assert all(e['review_status']=='unreviewed' for e in out['candidates'])


def test_longer_pass_and_brief_gap_supported_with_explicit_missing_time():
    # No .24-second requirement at the receiver and no 1.2-second transit cap.
    travel=[110+10*i for i in range(29)]
    out,_=calculate([100]*3+travel+[400]*3)
    assert len(out['candidates'])==1
    assert out['candidates'][0]['end_seconds']-out['candidates'][0]['start_seconds']>1.2
    short,_=calculate([100]*4+[180,None,300]+[400]*4)
    assert len(short['candidates'])==1
    assert short['candidates'][0]['missing_ball_seconds']==.1
    assert short['sequences']['by_team'][0]['bounded_missing_ball_seconds']==.1
    # Ball path continues to contain observed points only.
    assert len(short['ball']['path'])==10


def test_long_missing_gap_remains_unknown_and_cannot_create_pass():
    out,_=calculate([100]*4+[None]*4+[400]*4)
    assert out['candidates']==[]
    assert out['sequences']['unknown_seconds']==.4
    assert out['sequences']['diagnostics']['rejected_transitions']['insufficient_ball_evidence']==1


def test_partial_tracking_candidate_is_review_only_and_never_inflates_possession():
    out,_=calculate([100]*4+[180,None,None,None,None,300]+[400]*4)
    assert len(out['candidates'])==1
    event=out['candidates'][0]
    assert event['evidence_level']=='partial_tracking_review_only'
    assert event['possession_included'] is False
    assert out['sequences']['unknown_seconds']==.6
    assert out['sequences']['by_team'][0]['flight_and_gap_seconds']==0


def test_turnover_does_not_assign_uncertain_flight_to_either_team():
    out,_=calculate([100]*4+[200,300]+[400]*4,kit='kit-B')
    assert out['candidates'][0]['type']=='possible_turnover'
    assert out['sequences']['known_seconds']==.8
    assert out['sequences']['unknown_seconds']==.2
    assert len(out['sequences']['by_team'])==2


def test_confirmed_stoppage_splits_exactly_and_blocks_candidates():
    context=dict(revision=1,kit_mappings={'scene-0/kit-A':'Blue'},dead_ball_intervals=[[.35,.65]])
    out,_=calculate([100]*4+[200,300]+[400]*4,context=context)
    seq=out['sequences']
    assert out['candidates']==[]
    assert seq['dead_ball_seconds']==.3
    assert seq['by_team'][0]['team']=='Blue'
    assert seq['context_revision']==1
    assert seq['mapping_status']=='user_confirmed'
    assert sum(r['end_seconds']-r['start_seconds'] for r in seq['intervals'])==pytest.approx(1)
    assert seq['known_seconds']+seq['unknown_seconds']+seq['contested_seconds']+seq['dead_ball_seconds']==pytest.approx(1)


def test_retroactive_group_support_never_merges_different_track_identities():
    meta=dict(width=1000,height=600,fps=10,frame_count=8,duration_seconds=.8)
    calc=VideoInsights(meta,10)
    for i in range(8):
        p=player(100,identity='T1' if i<4 else 'T2')
        if i<2:p['team']=None
        calc.observe(i,[p,ball(100)])
    seq=calc.finish()['sequences']
    assert {a['track_id'] for a in seq['anchors']}=={'T1','T2'}
    assert seq['candidates']==[]


def test_cuts_and_competing_opponents_remain_unknown_or_contested():
    meta=dict(width=1000,height=600,fps=10,frame_count=8,duration_seconds=.8)
    calc=VideoInsights(meta,10)
    for i in range(8):calc.observe(i,[player(100,scene=i//4),ball(100,scene=i//4)])
    out=calc.finish()
    assert out['candidates']==[]
    assert len(out['sequences']['anchors'])==2
    calc=VideoInsights(meta,10)
    for i in range(8):calc.observe(i,[player(100),player(102,'T2','kit-B'),ball(101)])
    seq=calc.finish()['sequences']
    assert seq['contested_seconds']==.8
    assert seq['known_seconds']==0


def manifest():
    return dict(run_id='b'*32,engine='yolo_bytetrack',metadata=dict(sha256='a'*64,duration_seconds=1,width=1000,height=600,fps=10,frame_count=10),
                detection_summary=dict(sample_fps=10,decoded_frames=10),metrics={'video_insights':{'kits':[{'kit':'scene-0/kit-A'}]}})


def test_context_source_binding_explicit_confirmation_validation_and_cas(tmp_path):
    m=manifest();path=tmp_path/'sequence-context.json'
    payload=dict(revision=0,kit_mappings={'scene-0/kit-A':'Blue'},dead_ball_intervals=[[.1,.2]],confirmed=True)
    assert load_context(path,m)['revision']==0
    saved=save_context(path,m,payload)
    assert saved['revision']==1 and saved['kit_mappings']==payload['kit_mappings']
    with pytest.raises(ReviewConflict):save_context(path,m,payload)
    for updates in [dict(confirmed=False),dict(kit_mappings={'other-kit':'Blue'}),dict(dead_ball_intervals=[[.2,.1]]),dict(dead_ball_intervals=[[0,.4],[.3,.6]])]:
        with pytest.raises(ValueError):save_context(path,m,dict(payload,revision=1,**updates))
    other=copy.deepcopy(m);other['metadata']['sha256']='c'*64
    with pytest.raises(ValueError):load_context(path,other)


def test_context_revision_invalidates_only_selected_saved_insights(tmp_path):
    m=manifest();m['metrics']={}
    (tmp_path/'tracking.json').write_text('[]')
    assert ensure_insights(m,tmp_path)
    assert not ensure_insights(m,tmp_path)
    save_context(tmp_path/'sequence-context.json',m,dict(revision=0,kit_mappings={},dead_ball_intervals=[[0,.5]],confirmed=True))
    assert ensure_insights(m,tmp_path)
    assert m['metrics']['video_insights']['sequences']['dead_ball_seconds']==.5


def labels(meta):
    return dict(schema_version='sequence-benchmark-1',source_sha256=meta['sha256'],duration_seconds=meta['duration_seconds'],
                annotation_source='human',reviewer='test human fixture',split='development',match_group='test-fixture',reviewed_windows=[[0,meta['duration_seconds']]],
                kit_mappings={'scene-0/kit-A':'Blue'},events=[],possession_intervals=[])


def test_benchmark_retains_unknown_possession_and_rejects_unreviewed_labels():
    out,meta=calculate([100]*4+[None]*6)
    truth=labels(meta);truth['possession_intervals']=[dict(start_seconds=0,end_seconds=1,team='Blue')]
    score=score_sequences(truth,out,meta)
    assert score['possession']['classified_coverage_pct']==40
    assert score['possession']['unclassified_seconds']==.6
    assert score['possession']['by_team'][0]['absolute_share_error_percentage_points']==60
    with pytest.raises(ValueError):score_sequences(dict(truth,annotation_source='unreviewed'),out,meta)
    with pytest.raises(ValueError):score_sequences(dict(truth,source_sha256='wrong'),out,meta)
    pilot=score_sequences(dict(truth,annotation_source='codex_visual_pilot'),out,meta,allow_visual_pilot=True)
    assert pilot['independent_human_labels'] is False


def test_event_matching_is_one_to_one_max_cardinality_and_window_scoped():
    _,meta=calculate([None]*20)
    truth=labels(meta);truth['reviewed_windows']=[[0,1.8]]
    truth['events']=[dict(id='a',type='pass',timestamp_seconds=.4),dict(id='b',type='pass',timestamp_seconds=1.)]
    out=dict(candidates=[dict(id='x',type='possible_pass',timestamp_seconds=.9),dict(id='y',type='possible_pass',timestamp_seconds=1.5),dict(id='outside',type='possible_pass',timestamp_seconds=1.9)])
    score=score_sequences(truth,out,meta,tolerance=.6)
    assert score['events']['true_positives']==2
    assert score['events']['false_positives']==0
    assert score['events']['recall']==1
    assert score['events']['precision']==1


def test_team_retention_bridges_same_team_touches_and_is_reported_apart():
    out,_=calculate([100]*4+[None]*4+[400]*4)
    s=out['sequences'];r=s['retention']
    assert s['known_seconds']==.8 and s['unknown_seconds']==.4  # supported totals are unchanged
    assert r['retained_seconds']==.4 and r['touch_pairs']==1 and r['known_with_retention_seconds']==1.2
    assert r['coverage_with_retention_pct']==100 and r['by_team'][0]['with_retention_seconds']==1.2
    retained=[i for i in s['intervals'] if i['state']=='team_retained']
    assert [(i['start_seconds'],i['end_seconds'],i['team'],i['track_id'],i['support']) for i in retained]==[(.4,.8,'scene-0/kit-A',None,'same_team_touch_before_and_after')]


def test_team_retention_is_not_assumed_across_a_change_of_team_a_long_gap_or_a_stoppage():
    other_team,_=calculate([100]*4+[None]*4+[400]*4,kit='kit-B')
    long_gap,_=calculate([100]*4+[None]*60+[400]*4)
    stoppage,_=calculate([100]*4+[None]*4+[400]*4,context=dict(revision=1,kit_mappings={},dead_ball_intervals=[[.5,.6]]))
    for out in (other_team,long_gap,stoppage):
        assert out['sequences']['retention']['retained_seconds']==0
        assert not any(i['state']=='team_retained' for i in out['sequences']['intervals'])
    assert long_gap['sequences']['retention']['max_gap_seconds']==5


def test_weak_ball_detection_counts_only_when_the_ball_path_supports_it():
    meta=dict(width=1000,height=600,fps=10,frame_count=2,duration_seconds=.2,sha256='a'*64)
    def seconds(**extra):
        calc=VideoInsights(meta,10)
        for i in range(2):
            calc.observe(i,[player(100),dict(ball(100),confidence=.15,**extra)])
        return calc.finish()['quality']['selected_ball_seconds']
    assert seconds()==0 and seconds(ball_path_support='weak_between_confident')==.2
