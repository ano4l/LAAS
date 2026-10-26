import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.repository import Repository
from workers.video import pipeline, real_analysis


def make_video(path, frames=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def client(tmp_path, monkeypatch):
    uploads, outputs = tmp_path / "uploads", tmp_path / "outputs"
    uploads.mkdir()
    outputs.mkdir()
    repository = Repository(tmp_path / "jobs.sqlite3", outputs)
    repository.init()
    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "settings", SimpleNamespace(upload_dir=uploads, output_dir=outputs,
                        max_upload_mb=1, max_upload_bytes=1024 * 1024))
    monkeypatch.setattr(main, "WORKER", SimpleNamespace(submit=lambda *args: None))
    monkeypatch.setattr(main, "yolo_capabilities", lambda _: {"available": True})
    main.ACTIVE_JOBS.clear()
    with TestClient(main.app) as session:
        yield session
    main.ACTIVE_JOBS.clear()


def test_upload_validates_content_and_cleans_rejected_files(client):
    response = client.post("/videos/upload", files={"file": ("broken.mp4", b"not a video", "video/mp4")})
    assert response.status_code == 422
    assert not list(main.settings.upload_dir.iterdir())
    assert not main.repository.list_jobs()


def test_local_library_does_not_load_full_manifests_or_expose_paths(client, monkeypatch):
    run_id='d'*32
    main.repository.upsert_run(dict(run_id=run_id,engine='yolo_bytetrack',metadata=dict(filename='match.mp4',duration_seconds=30),
        metrics=dict(data_quality=dict(mode='yolo_bytetrack')),files={}))
    main.repository.create_job(dict(id='e'*32,run_id=run_id,status='complete',stage='Complete',progress=1,
        created_at='2026-10-25',input_path='private-source',source_video='/media/uploads/clip.mp4'))
    def forbidden(*args):raise AssertionError('Library must not read full manifests')
    monkeypatch.setattr(main,'_read_manifest',forbidden)
    result=client.get('/library').json()
    assert result['runs'][0]['run_id']==run_id and result['runs'][0]['engine']=='yolo_bytetrack'
    assert result['jobs'][0]['run_id']==run_id and result['jobs'][0]['filename']=='match.mp4'
    assert result['runs'][0]['summary']['mean_visible_players'] is None
    assert 'input_path' not in result['jobs'][0] and 'tracking' not in result['runs'][0]


def test_rejects_unsupported_and_oversized_uploads(client):
    assert client.post("/videos/upload", files={"file": ("x.txt", b"x")}).status_code == 415
    assert client.post("/videos/upload", files={"file": ("x.mp4", b"x" * (1024 * 1024 + 1))}).status_code == 413
    assert not list(main.settings.upload_dir.iterdir())


def test_missing_runtime_never_returns_demo(client, monkeypatch):
    monkeypatch.setattr(main, "yolo_capabilities", lambda _: {"available": False, "blockers": ["Missing model weights"]})
    response = client.post("/videos/upload", files={"file": ("x.mp4", b"x")})
    assert response.status_code == 503
    assert "Missing model" in response.json()["detail"]


def test_valid_upload_persists_job_and_retry_keeps_source(client, tmp_path):
    video = make_video(tmp_path / "input.mp4")
    response = client.post("/videos/upload", files={"file": ("input.mp4", video.read_bytes())})
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "queued"
    assert main.repository.get_job(job["id"])["status"] == "queued"
    assert client.post(f"/analysis-jobs/{job['id']}/retry").status_code == 409
    main.repository.update_job(job["id"], status="failed", error="test failure")
    retried = client.post(f"/analysis-jobs/{job['id']}/retry").json()
    assert retried["id"] != job["id"]
    assert retried["source_video"] == job["source_video"]


def test_upload_engine_failure_is_not_synthetic(tmp_path, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("Detector failed")
    monkeypatch.setattr(real_analysis, "run_real_analysis", unavailable)
    with pytest.raises(RuntimeError, match="Detector failed"):
        pipeline.run_analysis(tmp_path / "input.mp4", tmp_path)


def test_modal_upload_limit_rejected_before_job_submission(client, monkeypatch):
    monkeypatch.setenv("ANALYSIS_COMPUTE", "modal")
    monkeypatch.setattr(main, "video_metadata", lambda _: dict(duration_seconds=121))
    response = client.post("/videos/upload", files={"file": ("long.mp4", b"controlled metadata", "video/mp4")})
    assert response.status_code == 422
    assert "120 seconds" in response.json()["detail"]
    assert not list(main.settings.upload_dir.iterdir())
    assert not main.repository.list_jobs()


def test_restart_recovers_interrupted_jobs_and_processing_error_is_saved(client, tmp_path, monkeypatch):
    path = make_video(tmp_path / "input.mp4")
    job = main._new_job("restart-test", path)
    main._register_job(job)
    main.repository.update_job(job["id"], status="running")
    submitted = []
    monkeypatch.setattr(main, "WORKER", SimpleNamespace(submit=lambda *args: submitted.append(args)))
    main.startup()
    assert len(submitted) == 1
    assert submitted[0][1] == job["id"]
    assert main.ACTIVE_JOBS[job["id"]]["status"] == "queued"

    def broken_model(*args, **kwargs):
        raise RuntimeError("Model weights are incompatible")

    monkeypatch.setattr(main, "run_analysis", broken_model)
    main._process_job(job["id"], path)
    response = client.get(f"/analysis-jobs/{job['id']}").json()
    assert response["status"] == "failed"
    assert "incompatible" in response["error"]
    assert main.repository.get_job(job["id"])["status"] == "failed"
    assert main.repository.list_runs() == []


def test_streaming_preserves_full_duration_and_unknown_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSIS_PROFILE", "preview")
    # A controlled detector checks pipeline plumbing, not model accuracy.
    class Detector:
        names = {0: "player"}

        def __init__(self, *args, **kwargs):
            pass

        def detect(self, frame):
            return [dict(object_type="player", track_id="T7", confidence=0.9,
                         bbox=[4, 4, 20, 40], source="model_detection", interpolated=False)]

    monkeypatch.setattr(real_analysis, "yolo_capabilities", lambda _: {"available": True})
    weights = tmp_path / "test.pt"
    weights.write_bytes(b"controlled test detector")
    monkeypatch.setattr(real_analysis, "model_path", lambda _: weights)
    monkeypatch.setattr(real_analysis, "version", lambda _: "test-runtime")
    monkeypatch.setattr(real_analysis, "FootballDetector", Detector)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), set_num_threads=lambda _: None))
    video = make_video(tmp_path / "long.mp4", frames=1005)
    result = real_analysis.run_real_analysis(video, tmp_path / "outputs")
    summary = result["detection_summary"]
    assert summary["decoded_frames"] == 1005  # Beyond the old 900-frame truncation.
    assert summary["analyzed_frames"] == 201
    assert result["metrics"]["possession"]["home_pct"] is None
    measured = result["metrics"]["video_measurements"]
    assert measured["analyzed_frames"] == 201
    insights = result["metrics"]["video_insights"]
    assert insights["analyzed_frames"] == 201
    assert insights["source_video"]["sha256"] == result["metadata"]["sha256"]
    assert insights["quality"]["accuracy_evaluated"] is False
    assert measured["tracks"][0]["observations"] == 201
    assert measured["ball_coverage_pct"] == 0
    run = tmp_path / "outputs" / result["run_id"]
    tracking = json.loads((run / "tracking.json").read_text())
    assert tracking[-1]["timestamp_ms"] == 40000
    assert all(row["team"] is None and row["player_id"] is None for row in tracking)
    assert all(row["coordinate_space"] == "image_pixels" for row in tracking)
    output = real_analysis.video_metadata(run / "annotated.mp4")
    assert output["frame_count"] == 1005
    assert output["duration_seconds"] == pytest.approx(40.2)


@pytest.mark.parametrize("visible_player", [True, False])
def test_ball_detections_survive_without_confirmed_tracker_id(tmp_path, monkeypatch, visible_player):
    import torch
    import ultralytics
    from ultralytics.engine.results import Boxes
    from workers.video.detector import FootballDetector

    class RawDetector:
        names = {0: "ball", 1: "player"}

        def __init__(self, *args):
            pass

        def predict(self, frame, **kwargs):
            # Ball score is below ByteTrack's new-track threshold but above detection threshold.
            rows = [[20, 20, 24, 24, 0.16, 0]]
            if visible_player:
                rows.append([5, 5, 15, 40, 0.9, 1])
            boxes = Boxes(torch.tensor(rows), (48, 64))
            return [SimpleNamespace(names=self.names, boxes=boxes)]

    monkeypatch.setattr(ultralytics, "YOLO", RawDetector)
    detector = FootballDetector(tmp_path / "weights.pt", device="cpu", image_size=640,
                                ball_confidence=0.1, object_confidence=0.15, sample_fps=5)
    for _ in range(3):
        rows = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
        balls = [row for row in rows if row["object_type"] == "ball"]
        assert len(balls) == 1
        assert balls[0]["track_id"] is None
        assert balls[0]["source"] == "model_detection"
        assert balls[0]["interpolated"] is False
        if visible_player:
            assert any(row["track_id"] for row in rows if row["object_type"] == "player")


def test_tile_ball_boxes_map_back_to_source_pixels(tmp_path, monkeypatch):
    import torch
    import ultralytics
    from ultralytics.engine.results import Boxes
    from workers.video.detector import FootballDetector

    class Model:
        names = {0: "ball", 1: "player"}

        def __init__(self, *args):
            pass

        def predict(self, frame, **kwargs):
            # Full image has no ball. Each crop has one at the same local location.
            data = [[10, 10, 20, 20, 0.8, 0]] if frame.shape[1] == 128 else []
            boxes = Boxes(torch.tensor(data).reshape(-1, 6), frame.shape[:2])
            return [SimpleNamespace(names=self.names, boxes=boxes)]

    monkeypatch.setattr(ultralytics, "YOLO", Model)
    detector = FootballDetector(tmp_path / "test.pt", device="cpu", image_size=640,
                                ball_confidence=0.1, object_confidence=0.15, sample_fps=25, tile_size=128)
    rows = detector.detect(np.zeros((128, 256, 3), dtype=np.uint8))
    assert [r["bbox"] for r in rows] == [[10, 10, 20, 20], [106, 10, 116, 20], [138, 10, 148, 20]]
    assert all(row["model_role"] == "ball_tile_detector" and row["track_id"] is None for row in rows)
