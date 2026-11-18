import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from apps.api.modal_gateway import create_gateway
from workers.video.event_review import empty_review,save_review,load_review,ReviewConflict

MANIFEST=dict(run_id='b'*32,engine='yolo_bytetrack',metadata=dict(duration_seconds=30,sha256='source-hash'))


def event(identity='pass-1',kind='pass',outcome='completed',status='approved',team='home',at=5):
    return dict(id=identity,type=kind,outcome=outcome,review_status=status,team=team,timestamp_seconds=at,player='Player 7',notes='')


def payload():return dict(revision=0,teams=dict(home='Home',away='Away'),events=[],coverage=[],possession_intervals=[])


def test_only_approved_records_count_unknown_outcomes_and_denominators_stay_explicit(tmp_path):
    path=tmp_path/'review.json';body=payload()
    assert empty_review(MANIFEST)['summary']['teams']['home'] is None
    body['events']=[event(),event('p2',outcome='unknown',at=8),event('draft',kind='shot',outcome='goal',status='draft')]
    body['coverage']=[[0,10],[5,15]]
    body['possession_intervals']=[dict(id='pos1',start_seconds=0,end_seconds=5,team='home',review_status='approved')]
    result=save_review(path,MANIFEST,body);summary=result['summary']
    assert summary['approved_events']==2 and summary['draft_events']==1
    assert summary['reviewed_seconds']==15 and summary['reviewed_video_pct']==50
    assert summary['teams']['home']['passes']==dict(recorded=2,completed=1,known_outcomes=1,outcome_coverage_pct=50,completion_pct=100)
    assert summary['teams']['home']['shots']['goals']==0
    assert summary['teams']['away'] is None
    assert summary['possession']['home_share_pct']==100
    assert summary['possession']['known_coverage_pct']==16.67
    assert summary['possession']['unknown_seconds']==25
    assert load_review(path,MANIFEST)['revision']==1
    with pytest.raises(ReviewConflict):save_review(path,MANIFEST,body)


def test_outcome_rates_use_known_outcomes_and_turnovers_and_player_outcomes_are_counted(tmp_path):
    body=payload()
    body['events']=[event('p1'),event('p2',outcome='incomplete'),event('p3',outcome='unknown'),
        event('s1','shot','goal'),event('s2','shot','on_target'),event('s3','shot','off_target'),event('s4','shot','blocked'),event('s5','shot','unknown'),
        event('t1','tackle','won'),event('t2','tackle','lost'),event('t3','tackle','unknown'),
        event('d1','dribble','successful'),event('d2','dribble','unsuccessful'),event('d3','dribble','unsuccessful'),
        event('u1','duel','won'),event('i1','interception','successful'),
        event('a1','tackle','won',team='away'),event('draft','duel','lost',status='draft')]
    summary=save_review(tmp_path/'review.json',MANIFEST,body)['summary']
    home=summary['teams']['home']
    assert summary['summary_version']==2 and {'scope','rates','shots','turnovers'}<=set(summary['definitions'])
    assert home['shots']==dict(recorded=5,on_target=2,goals=1,off_target=1,blocked=1,known_outcomes=4,on_target_pct=50,conversion_pct=25)
    assert home['tackles']==dict(recorded=3,won=1,lost=1,known_outcomes=2,win_pct=50)
    assert home['dribbles']==dict(recorded=3,successful=1,unsuccessful=2,known_outcomes=3,success_pct=33.33)
    # The draft lost duel is not counted, and one decided duel gives a rate of 100, not a guess.
    assert home['duels']==dict(recorded=1,won=1,lost=0,known_outcomes=1,win_pct=100)
    assert home['turnovers']==dict(conceded=3,incomplete_passes=1,unsuccessful_dribbles=2,won=2,tackles_won=1)
    away=summary['teams']['away']
    assert away['tackles']['win_pct']==100 and away['shots']['conversion_pct'] is None and away['dribbles']['success_pct'] is None
    player=next(p for p in summary['players'] if p['team']=='home')
    assert player['event_count']==16 and player['outcomes']['shot']==dict(goal=1,on_target=1,off_target=1,blocked=1,unknown=1)
    assert player['outcomes']['pass']==dict(completed=1,incomplete=1,unknown=1)


def located(identity,start,end,team='home',period=1,**extra):
    return dict(event(identity,team=team),period=period,location=dict(x=start[0],y=start[1]),end_location=dict(x=end[0],y=end[1]),**extra)


def test_locations_periods_and_progressive_passes_follow_the_recorded_direction_of_play(tmp_path):
    body=payload()
    body['periods']={'1':dict(home_attacks='right',reference_video_seconds=10,reference_clock_seconds=600),'2':dict(home_attacks='left')}
    body['events']=[
        located('own-half-long',(10,34),(45,34)),            # 35 m closer, both ends in the own half
        located('own-half-short',(10,34),(35,34)),           # 25 m closer: not enough in the own half
        located('across-halfway',(45,34),(62,34)),           # 17 m closer across halfway
        located('final-third',(80,34),(92,34),receiver='Player 9',body_part='foot',situation='free_kick'),
        located('backwards',(80,34),(60,34)),
        located('second-half',(95,34),(60,34),period=2),      # home attacks left after the break: 35 m closer
        located('away',(95,34),(60,34),team='away'),          # away attacks left in the first half
        dict(event('no-end'),location=dict(x=50,y=30)),
        dict(event('shot','shot','goal'),location=dict(x=96,y=30),body_part='head',situation='corner')]
    path=tmp_path/'review.json';saved=save_review(path,MANIFEST,body)
    home,away=saved['summary']['teams']['home'],saved['summary']['teams']['away']
    assert home['progression']==dict(shots_located=1,passes_located=6,passes_with_direction=6,progressive_passes=4)
    assert away['progression']==dict(shots_located=0,passes_located=1,passes_with_direction=1,progressive_passes=1)
    stored={e['id']:e for e in load_review(path,MANIFEST)['events']}
    assert stored['final-third']['receiver']=='Player 9' and stored['final-third']['situation']=='free_kick'
    assert stored['no-end']['end_location'] is None and stored['shot']['body_part']=='head' and stored['second-half']['period']==2
    # Without a recorded direction of play nothing is called progressive.
    body['revision']=1;body['periods']={}
    unknown=save_review(path,MANIFEST,body)['summary']['teams']['home']['progression']
    assert unknown==dict(shots_located=1,passes_located=6,passes_with_direction=0,progressive_passes=None)


def test_review_saved_before_locations_existed_still_loads(tmp_path):
    path=tmp_path/'review.json';saved=save_review(path,MANIFEST,dict(payload(),events=[event()]))
    old=json.loads(path.read_text());old['schema_version']='event-review-1';del old['periods'];del old['summary']
    for row in old['events']:
        for key in ('period','location','end_location','body_part','situation','receiver'):del row[key]
    path.write_text(json.dumps(old))
    loaded=load_review(path,MANIFEST)
    assert loaded['events'][0]['period']==1 and loaded['events'][0]['location'] is None and set(loaded['periods'])=={'1','2','3','4'}
    assert loaded['summary']['teams']['home']['progression']['progressive_passes'] is None
    assert save_review(path,MANIFEST,dict(payload(),revision=saved['revision'],events=[event()]))['revision']==2


def test_approved_export_is_versioned_and_leaves_out_drafts(tmp_path):
    from workers.video.event_review import approved_export
    body=payload()
    body['periods']={'1':dict(home_attacks='left',reference_video_seconds=10,reference_clock_seconds=600)}
    body['events']=[located('p1',(80,10),(60,20)),dict(event('draft','shot','goal',status='draft'),location=dict(x=5,y=34)),
                    dict(event('late','duel','won',at=20))]
    body['possession_intervals']=[dict(id='pos1',start_seconds=0,end_seconds=5,team='home',review_status='approved'),
                                  dict(id='pos2',start_seconds=5,end_seconds=8,team='away',review_status='draft')]
    out=approved_export(save_review(tmp_path/'review.json',MANIFEST,body),MANIFEST)
    assert (out['schema'],out['schema_version'])==('laas.approved-events','1.0.0') and out['review']['revision']==1
    assert [e['event_id'] for e in out['events']]==['p1','late'] and [r['interval_id'] for r in out['possession_intervals']]==['pos1']
    first=out['events'][0]
    assert first['match_clock_seconds']==595 and out['events'][1]['match_clock_seconds']==610
    assert first['location']==dict(x=80,y=10) and first['location_attacking']==dict(x=25,y=58) and first['end_location_attacking']==dict(x=45,y=48)
    assert first['review']==dict(status='approved',source='human_annotation') and out['coordinate_system']['pitch_length']==105
    assert out['periods']==[dict(period_id=1,home_attacks='left',reference_video_seconds=10,reference_clock_seconds=600)]
    assert out['summary']['approved_events']==2 and out['source']['sha256']=='source-hash'


@pytest.mark.parametrize('change',[lambda p:p.update(events=[event(at=31)]),
    lambda p:p.update(events=[dict(event(),location=dict(x=106,y=10))]),
    lambda p:p.update(events=[dict(event(),location=dict(x=10))]),
    lambda p:p.update(events=[dict(event(),period=5)]),
    lambda p:p.update(events=[dict(event(),body_part='knee')]),
    lambda p:p.update(events=[dict(event(kind='tackle',outcome='won'),end_location=dict(x=10,y=10))]),
    lambda p:p.update(events=[dict(event(kind='shot',outcome='goal'),receiver='Player 9')]),
    lambda p:p.update(events=[dict(event(kind='duel',outcome='won'),situation='corner')]),
    lambda p:p.update(periods={'1':dict(home_attacks='up')}),
    lambda p:p.update(periods={'1':dict(reference_video_seconds=5)}),
    lambda p:p.update(periods={'5':dict(home_attacks='left')}),
    lambda p:p.update(events=[event(at=float('nan'))]),
    lambda p:p.update(events=[event(),event()]),
    lambda p:p.update(events=[event(kind='goal',outcome='goal')]),
    lambda p:p.update(coverage=[[10,0]]),
    lambda p:p.update(possession_intervals=[dict(id='a',start_seconds=0,end_seconds=10,team='home',review_status='approved'),dict(id='b',start_seconds=8,end_seconds=15,team='away',review_status='approved')])])
def test_invalid_annotations_do_not_persist(tmp_path,change):
    body=payload();change(body);path=tmp_path/'review.json'
    with pytest.raises(ValueError):save_review(path,MANIFEST,body)
    assert not path.exists()


def test_private_review_api_persists_and_rejects_stale_revision_without_gpu(tmp_path):
    def forbidden(*args):raise AssertionError('Reviewing must not invoke GPU')
    volume=SimpleNamespace(reload=lambda:None,commit=lambda:None)
    client=TestClient(create_gateway(volume,SimpleNamespace(spawn=forbidden),'k'*32,tmp_path))
    job_id='a'*32;directory=tmp_path/'requests'/job_id/'outputs'/MANIFEST['run_id'];directory.mkdir(parents=True)
    job=dict(id=job_id,run_id=MANIFEST['run_id'],status='complete',result=MANIFEST,created_at='2026-10-24')
    (tmp_path/'web-jobs'/f'{job_id}.json').write_text(json.dumps(job))
    route=f"/runs/{MANIFEST['run_id']}/review";headers={'Authorization':'Bearer '+'k'*32}
    assert client.get(route).status_code==401
    assert client.post(route,json=payload()).status_code==401
    assert client.get(route,headers=headers).json()['revision']==0
    body=payload();body['events']=[event()]
    assert client.post(route,headers=headers,json=body).status_code==200
    assert client.post(route,headers=headers,json=body).status_code==409
    assert client.get(route,headers=headers).json()['summary']['approved_events']==1
    assert client.get(route+'/approved-events').status_code==401
    exported=client.get(route+'/approved-events',headers=headers).json()
    assert exported['schema_version']=='1.0.0' and [e['event_id'] for e in exported['events']]==['pass-1']
    body['revision']=1;body['events']=[event(at=31)]
    assert client.post(route,headers=headers,json=body).status_code==422
    assert client.get(route,headers=headers).json()['revision']==1
