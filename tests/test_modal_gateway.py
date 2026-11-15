import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from apps.api.modal_gateway import create_gateway
from workers.video import real_analysis


def test_sequence_context_auth_cas_recalculation_and_event_separation(tmp_path):
    def forbidden(*args):raise AssertionError('Context editing must not start inference')
    client=TestClient(create_gateway(SimpleNamespace(commit=lambda:None,reload=lambda:None),
        SimpleNamespace(spawn=forbidden),'k'*32,tmp_path))
    headers={'Authorization':'Bearer '+'k'*32};run_id='9'*32;job_id='8'*32
    directory=tmp_path/'requests'/job_id/'outputs'/run_id;directory.mkdir(parents=True)
    result=dict(run_id=run_id,engine='yolo_bytetrack',metadata=dict(filename='source.mp4',sha256='a'*64,
        width=1000,height=600,fps=10,frame_count=10,duration_seconds=1),
        detection_summary=dict(sample_fps=10,decoded_frames=10),metrics={})
    record=tmp_path/'web-jobs'/f'{job_id}.json'
    record.write_text(json.dumps(dict(id=job_id,run_id=run_id,status='complete',created_at='2026-10-25',result=result)))
    (directory/'manifest.json').write_text(json.dumps(result));(directory/'tracking.json').write_text('[]')
    approved=b'{"preserved":"approved event review is separate"}';(directory/'review.json').write_bytes(approved)
    url=f'/runs/{run_id}/sequence-context'
    payload=dict(revision=0,confirmed=True,kit_mappings={},dead_ball_intervals=[[0,.2]])
    assert client.get(url).status_code==401
    assert client.post(url,json=payload).status_code==401
    assert client.get(url,headers=headers).json()['revision']==0
    assert client.post(url,headers=headers,json=dict(payload,kit_mappings={'alien':'Home'})).status_code==422
    assert client.post(url,headers=headers,json=dict(payload,dead_ball_intervals=[[0,2]])).status_code==422
    assert client.post(url,headers=headers,json=payload).json()['revision']==1
    assert client.post(url,headers=headers,json=payload).status_code==409
    sequences=client.get(f'/runs/{run_id}',headers=headers).json()['metrics']['video_insights']['sequences']
    assert sequences['context_revision']==1 and abs(sequences['dead_ball_seconds']-.2)<1e-8
    assert (directory/'review.json').read_bytes()==approved


def test_selected_insights_backfill_is_persistent_signed_and_never_starts_inference(tmp_path):
    def forbidden(*args):
        raise AssertionError('Recalculating saved video analytics must not start inference')
    client=TestClient(create_gateway(SimpleNamespace(commit=lambda:None,reload=lambda:None),
                                    SimpleNamespace(spawn=forbidden),'k'*32,tmp_path))
    headers={'Authorization':'Bearer '+'k'*32}
    job_id,run_id='5'*32,'6'*32
    directory=tmp_path/'requests'/job_id/'outputs'/run_id
    directory.mkdir(parents=True)
    result=dict(run_id=run_id,engine='yolo_bytetrack',metadata=dict(filename='source.mp4',width=1000,height=600,fps=10,frame_count=10,duration_seconds=1),
                detection_summary=dict(sample_fps=10,decoded_frames=10),metrics=dict(video_measurements={'existing':True}))
    job=dict(id=job_id,run_id=run_id,status='complete',created_at='2026-10-25',result=result)
    record=tmp_path/'web-jobs'/f'{job_id}.json'
    record.write_text(json.dumps(job))
    (directory/'tracking.json').write_text('[]')
    (directory/'manifest.json').write_text(json.dumps(result))
    assert client.get('/library',headers=headers).status_code==200
    assert 'video_insights' not in json.loads(record.read_text())['result']['metrics']
    assert client.get(f'/runs/{run_id}').status_code==401
    response=client.get(f'/runs/{run_id}',headers=headers)
    assert response.status_code==200
    out=response.json()
    assert out['metrics']['video_insights']['control']['unknown_seconds']==1
    assert json.loads(record.read_text())['result']['metrics']['video_insights']['schema_version']=='video-insights-4'
    metrics=client.get(out['files']['metrics_json']).json()
    assert metrics['video_measurements']=={'existing':True}
    assert metrics['video_insights']['analyzed_frames']==10
    (directory/'tracking.json').unlink()
    assert client.get(f'/runs/{run_id}',headers=headers).json()['metrics']['video_insights']==out['metrics']['video_insights']


def test_gateway_auth_upload_completion_and_signed_media(tmp_path, monkeypatch):
    token = "controlled-access-key-" * 3
    volume = SimpleNamespace(commit=lambda: None, reload=lambda: None)
    submitted = []
    def spawn(request_id):
        submitted.append(request_id)
        return SimpleNamespace(object_id="controlled-call")
    pending = True
    run_id = "a" * 32
    class Call:
        def get(self, timeout):
            if pending:
                raise TimeoutError()
            return dict(run_id=run_id)
    monkeypatch.setattr(real_analysis, "video_metadata", lambda _: dict(duration_seconds=5, width=1920, height=1080, fps=25))
    app = create_gateway(volume, SimpleNamespace(spawn=spawn), token, tmp_path, lambda _: Call())
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/analysis-jobs").status_code == 401
    headers = dict(Authorization=f"Bearer {token}")
    assert client.get("/capabilities", headers=headers).status_code == 200
    job = client.post("/videos/upload", files={"file": ("match.mp4", b"controlled video")}, headers=headers).json()
    assert submitted == [job["id"]]
    assert "call_id" not in job
    assert client.post("/videos/upload", files={"file": ("again.mp4", b"x")}, headers=headers).status_code == 409
    assert client.get(f"/analysis-jobs/{job['id']}", headers=headers).json()["status"] == "running"
    output = tmp_path / "requests" / job["id"] / "outputs" / run_id
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(dict(run_id=run_id)))
    (output / "annotated.mp4").write_bytes(b"controlled video artifact")
    pending = False
    done = client.get(f"/analysis-jobs/{job['id']}", headers=headers).json()
    assert done["status"] == "complete"
    url = done["result"]["files"]["annotated_video"]
    assert token not in url
    assert client.get(url).content == b"controlled video artifact"
    assert client.get(url + "corrupt").status_code == 403
    assert client.get(url.split("signature=")[0] + "signature=invalid%E2%9C%93").status_code == 403
    assert client.get(url.split("?")[0] + "?expires=1&signature=bad").status_code == 403
    assert client.get(url, headers={"Range": "bytes=0-3"}).status_code == 206
    source = done["result"]["files"]["source_video"]
    assert token not in source and client.get(source).content == b"controlled video"
    assert client.get(source.split("signature=")[0] + "signature=bad").status_code == 403
    assert client.get("/runs", headers=headers).json()[0]["run_id"] == run_id


def test_gateway_invalid_video_cannot_start_gpu(tmp_path, monkeypatch):
    def invalid(_):
        raise ValueError("Cannot decode video")
    def must_not_run(*args):
        raise AssertionError("Rejected uploads must not start a GPU")
    monkeypatch.setattr(real_analysis, "video_metadata", invalid)
    client = TestClient(create_gateway(SimpleNamespace(commit=lambda: None, reload=lambda: None),
                                      SimpleNamespace(spawn=must_not_run), "k" * 32, tmp_path))
    response = client.post("/videos/upload", files={"file": ("broken.mp4", b"bad")}, headers={"Authorization": "Bearer " + "k" * 32})
    assert response.status_code == 422
    assert not list((tmp_path / "web-jobs").glob("*.json"))


def test_sample_reuses_full_tracking_without_gpu_and_backfills_measurements(tmp_path):
    def must_not_run(*args):
        raise AssertionError("Opening a sample must not start GPU inference")
    volume = SimpleNamespace(commit=lambda: None, reload=lambda: None)
    client = TestClient(create_gateway(volume, SimpleNamespace(spawn=must_not_run), "k" * 32, tmp_path))
    headers = {"Authorization": "Bearer " + "k" * 32}
    assert client.get("/sample").status_code == 401
    assert client.get("/sample", headers=headers).status_code == 404
    job_id, run_id = "b" * 32, "c" * 32
    directory = tmp_path / "requests" / job_id / "outputs" / run_id
    directory.mkdir(parents=True)
    result = dict(run_id=run_id, engine="yolo_bytetrack", metrics={},
                  metadata=dict(width=100, height=100, fps=1, frame_count=30, duration_seconds=30),
                  detection_summary=dict(sample_fps=1, decoded_frames=30), tracking=[])
    row = dict(frame_idx=29, object_type="player", track_id="T1", confidence=.9, bbox=[10, 10, 20, 30])
    (directory / "tracking.json").write_text(json.dumps([row]))
    (directory / "manifest.json").write_text(json.dumps(result))
    job = dict(id=job_id, run_id=run_id, status="complete", sample=True, result=result, created_at="2026-10-21")
    (tmp_path / "web-jobs" / f"{job_id}.json").write_text(json.dumps(job))
    response = client.get("/sample", headers=headers).json()
    measured = response["result"]["metrics"]["video_measurements"]
    assert measured["analyzed_frames"] == 30
    assert measured["mean_visible_players"] == .03  # Empty frames count too.
    assert measured["tracks"][0]["first_seen_seconds"] == 29
    assert json.loads((directory / "metrics.json").read_text())["video_measurements"] == measured
    assert client.get("/sample", headers=headers).json()["id"] == job_id


def test_library_is_small_source_bound_and_never_backfills_unselected_runs(tmp_path):
    def forbidden(*args):raise AssertionError('Library browsing must not start GPU inference')
    volume=SimpleNamespace(commit=lambda:None,reload=lambda:None)
    client=TestClient(create_gateway(volume,SimpleNamespace(spawn=forbidden),'k'*32,tmp_path))
    headers={'Authorization':'Bearer '+'k'*32}
    assert client.get('/library').status_code==401
    assert client.get('/library',headers=headers).json()==dict(runs=[],jobs=[])
    for index in range(2):
        job_id=str(index+1)*32;run_id=str(index+3)*32
        result=dict(run_id=run_id,engine='yolo_bytetrack',metadata=dict(filename='source.mp4',duration_seconds=30),
                    tracking=[dict(frame_idx=frame,bbox=[1,2,3,4]) for frame in range(1000)],
                    metrics={} if index==0 else dict(video_measurements=dict(analyzed_frames=750,ball_coverage_pct=55,
                        tracks=[dict(track_id='T1')]*1000)),detection_summary=dict(analyzed_frames=750,unique_tracks=49))
        job=dict(id=job_id,run_id=run_id,status='complete',result=result,created_at=f'2026-10-2{index+3}',
                 filename=f'clip-{index}.mp4',stage='Complete',progress=1,call_id='private-call')
        (tmp_path/'web-jobs'/f'{job_id}.json').write_text(json.dumps(job))
    response=client.get('/library',headers=headers)
    assert response.status_code==200 and len(response.content)<3000
    listing=response.json();assert len(listing['runs'])==2
    assert listing['runs'][0]['metadata']['filename']=='clip-1.mp4'
    assert listing['runs'][0]['summary']['ball_candidate_coverage_pct']==55
    assert listing['runs'][1]['summary']['ball_candidate_coverage_pct'] is None
    assert 'tracking' not in listing['runs'][0] and 'files' not in listing['runs'][0]
    assert 'result' not in listing['jobs'][0] and 'call_id' not in listing['jobs'][0]
    # The first run has no tracking file; selecting the second must not backfill it.
    selected=client.get('/runs/'+'4'*32,headers=headers)
    assert selected.status_code==200 and len(selected.json()['tracking'])==1000
    assert 'signature=' in selected.json()['files']['annotated_video']
    assert selected.json()['created_at']=='2026-10-24'


def test_library_reads_saved_status_without_resolving_provider_calls(tmp_path):
    def forbidden(*args):raise AssertionError('Library must not query provider call results')
    client=TestClient(create_gateway(SimpleNamespace(commit=lambda:None,reload=lambda:None),
        SimpleNamespace(spawn=forbidden),'k'*32,tmp_path,forbidden))
    job=dict(id='a'*32,status='queued',stage='Queued',progress=.05,created_at='2026-10-25',call_id='pending',result=None)
    (tmp_path/'web-jobs'/('a'*32+'.json')).write_text(json.dumps(job))
    result=client.get('/library',headers={'Authorization':'Bearer '+'k'*32}).json()
    assert result['jobs'][0]['status']=='queued' and result['runs']==[]
