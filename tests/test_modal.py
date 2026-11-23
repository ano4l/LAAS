from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from workers.video import pipeline, modal_client, real_analysis
from workers.video.modal_policy import reserve_attempt, validate_request, validate_video, ARTIFACTS, OPTIONAL_ARTIFACTS, MAX_DAILY_ATTEMPTS


def test_usage_limit_counts_failures_and_resets_on_calendar_boundaries():
    ledger = {}
    now = datetime(2026, 10, 20, tzinfo=timezone.utc)
    for _ in range(MAX_DAILY_ATTEMPTS):
        reserve_attempt(ledger, now)
    with pytest.raises(ValueError, match="daily"):
        reserve_attempt(ledger, now)
    ledger["months"]["2026-10"] = 20
    with pytest.raises(ValueError, match="monthly"):
        reserve_attempt(ledger, datetime(2026, 10, 21, tzinfo=timezone.utc))
    reserve_attempt(ledger, datetime(2026, 11, 1, tzinfo=timezone.utc))


def test_modal_limits_reject_path_traversal_and_long_video(tmp_path):
    with pytest.raises(ValueError):
        validate_request("../usage.json")
    with pytest.raises(ValueError):
        validate_request("a" * 32, tile_size=1)
    source = tmp_path / "video.mp4"
    source.touch()
    with pytest.raises(ValueError, match="120 seconds"):
        validate_video(source, dict(duration_seconds=121))


def test_cloud_error_is_not_replaced_by_local_analysis(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYSIS_COMPUTE", "modal")
    def fail(*args):
        raise RuntimeError("Cloud quota exhausted")
    monkeypatch.setattr(modal_client, "run_modal_analysis", fail)
    with pytest.raises(RuntimeError, match="Cloud quota"):
        pipeline.run_analysis(tmp_path / "source.mp4", tmp_path)


@pytest.mark.parametrize("corrupt", [False, True])
def test_modal_import_verifies_hashes_and_cleans_only_after_success(tmp_path, monkeypatch, corrupt):
    for key in ["YOLO_BALL_MODEL_PATH", "ANALYSIS_FPS", "YOLO_IMAGE_SIZE", "BALL_TILE_SIZE"]:
        monkeypatch.delenv(key, raising=False)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"controlled video source")
    monkeypatch.setattr(real_analysis, "video_metadata", lambda _: dict(duration_seconds=5, width=1920, height=1080, fps=25))
    run_id = "b" * 32
    manifest = dict(run_id=run_id, metadata=dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    artifacts = {name: b"artifact" for name in (*ARTIFACTS, *OPTIONAL_ARTIFACTS)}
    artifacts["manifest.json"] = json.dumps(manifest).encode()
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in artifacts.items()}
    if corrupt:
        hashes["annotated.mp4"] = "wrong"
    removed = []
    class Volume:
        def batch_upload(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def put_file(self, local, remote):
            assert remote.endswith("/source.mp4")
        def read_file(self, remote):
            yield artifacts[Path(remote).name]
        def remove_file(self, path, recursive):
            assert path.startswith("/requests/") and recursive
            removed.append(path)
    polls = []
    class Call:
        def get(self, timeout):
            polls.append(timeout)
            if len(polls) == 1:
                raise TimeoutError
            return dict(run_id=run_id, sha256=hashes)
    monkeypatch.setattr(modal_client, "POLL_SECONDS", 0)
    remote = SimpleNamespace(spawn=lambda *args: Call())
    fake_modal = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda _: Volume()),
                                Function=SimpleNamespace(from_name=lambda *args: remote))
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    outputs = tmp_path / "outputs"
    if corrupt:
        with pytest.raises(RuntimeError, match="checksum"):
            modal_client.run_modal_analysis(source, outputs)
        assert not (outputs / run_id).exists()
        assert not removed
    else:
        assert modal_client.run_modal_analysis(source, outputs) == manifest
        assert (outputs / run_id / "annotated.mp4").is_file()
        assert len(removed) == 1
        assert len(polls) == 2  # spawned, then polled until the result arrived
        for name in OPTIONAL_ARTIFACTS:
            assert (outputs / run_id / name).is_file()


def test_older_deployment_without_optional_files_still_imports(tmp_path, monkeypatch):
    for key in ["YOLO_BALL_MODEL_PATH", "ANALYSIS_FPS", "YOLO_IMAGE_SIZE", "BALL_TILE_SIZE"]:
        monkeypatch.delenv(key, raising=False)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"controlled video source")
    monkeypatch.setattr(real_analysis, "video_metadata", lambda _: dict(duration_seconds=5, width=1920, height=1080, fps=25))
    run_id = "c" * 32
    manifest = dict(run_id=run_id, metadata=dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    artifacts = {name: b"artifact" for name in ARTIFACTS}
    artifacts["manifest.json"] = json.dumps(manifest).encode()
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in artifacts.items()}
    class Volume:
        def batch_upload(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def put_file(self, local, remote): pass
        def read_file(self, remote): yield artifacts[Path(remote).name]
        def remove_file(self, path, recursive): pass
    call = SimpleNamespace(get=lambda timeout: dict(run_id=run_id, sha256=hashes))
    fake_modal = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda _: Volume()),
                                Function=SimpleNamespace(from_name=lambda *args: SimpleNamespace(spawn=lambda *a: call)))
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    assert modal_client.run_modal_analysis(source, tmp_path / "outputs") == manifest
    assert not (tmp_path / "outputs" / run_id / "track-links.json").exists()


def test_wait_cancels_the_cloud_job_when_it_never_finishes():
    cancelled = []
    def get(timeout):
        raise TimeoutError
    call = SimpleNamespace(get=get, cancel=lambda: cancelled.append(True))
    with pytest.raises(TimeoutError, match="cancelled"):
        modal_client.wait_for_call(call, wait_seconds=0, poll_seconds=0)
    assert cancelled == [True]


def test_opt_in_fallback_runs_locally_and_records_the_reason(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYSIS_COMPUTE", "modal")
    monkeypatch.setenv("ANALYSIS_FALLBACK", "local")
    monkeypatch.setenv("ANALYSIS_PROFILE", "accurate")
    def fail(*args):
        raise ValueError("Modal prototype daily attempt limit reached (5). No analysis started.")
    monkeypatch.setattr(modal_client, "run_modal_analysis", fail)
    seen = {}
    def local(source, outputs, progress):
        import os
        seen["profile"] = os.environ["ANALYSIS_PROFILE"]
        (outputs / "run").mkdir(parents=True)
        return dict(run_id="run", detection_summary={}, warnings=[])
    monkeypatch.setattr(real_analysis, "run_real_analysis", local)
    stages = []
    result = pipeline.run_analysis(tmp_path / "source.mp4", tmp_path / "out", lambda stage, fraction: stages.append(stage))
    import os
    assert seen["profile"] == "balanced" and os.environ["ANALYSIS_PROFILE"] == "accurate"
    assert result["detection_summary"]["compute_fallback"]["requested"] == "modal"
    assert "daily attempt limit" in result["detection_summary"]["compute_fallback"]["reason"]
    assert any("this server" in stage for stage in stages)
    assert json.loads((tmp_path / "out" / "run" / "manifest.json").read_text(encoding="utf-8"))["detection_summary"]["compute_provider"] == "local"


def test_balanced_profile_sits_between_preview_and_accurate(monkeypatch):
    from workers.video.accuracy import analysis_settings
    for key in ["ANALYSIS_FPS", "YOLO_IMAGE_SIZE", "BALL_TILE_SIZE"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANALYSIS_PROFILE", "balanced")
    assert analysis_settings(25) == dict(profile="balanced", stride=2, image_size=960, tile_size=0)
    assert analysis_settings(30)["stride"] == 3 and analysis_settings(60)["stride"] == 5
