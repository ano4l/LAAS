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


@pytest.mark.parametrize('change',[lambda p:p.update(events=[event(at=31)]),
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
    body['revision']=1;body['events']=[event(at=31)]
    assert client.post(route,headers=headers,json=body).status_code==422
    assert client.get(route,headers=headers).json()['revision']==1
