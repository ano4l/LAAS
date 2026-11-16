"""Private GPU worker. Deploy: python -m modal deploy deploy/modal_app.py"""
from pathlib import Path
import os
import sys

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/app")
sys.path.insert(0, str(ROOT))
from workers.video.modal_policy import APP_NAME, VOLUME_NAME

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Only source and the explicitly selected weights enter the image, never .env or user videos.
if modal.is_local():
    from workers.video.yolo_adapter import model_path, pitch_model_path
    weights = model_path(ROOT)
    pitch_weights = pitch_model_path(ROOT)
    if not weights.is_file():
        raise FileNotFoundError("Set YOLO_MODEL_PATH to trusted football weights before deploying Modal")
else:
    weights = Path("/models/football.pt")
    pitch_weights = None  # already baked into the image when it was built
rescue_weights = ROOT / 'references/Soccer-Analytics/yolov8m.pt' if modal.is_local() else Path('/models/ball-rescue.pt')

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.13.0", "torchvision==0.28.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ultralytics==8.4.178", "lap>=0.5.12", "imageio-ffmpeg>=0.6.0")
    .add_local_dir(str(ROOT / "workers"), "/app/workers", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(str(weights), "/models/football.pt", copy=True)
    .add_local_file(str(rescue_weights), "/models/ball-rescue.pt", copy=True)
    .env({"PYTHONPATH": "/app", "YOLO_MODEL_PATH": "/models/football.pt", "YOLO_DEVICE": "0",
          "BALL_CONFIDENCE": "0.1", "OBJECT_CONFIDENCE": "0.45", "TRACKER_BACKEND":"botsort", "YOLO_BALL_RESCUE_PATH": "/models/ball-rescue.pt"})
)
# Optional pitch landmark model. Without it the worker runs exactly as before and reports no pitch positions.
if pitch_weights is not None and pitch_weights.is_file():
    image = image.add_local_file(str(pitch_weights), "/models/pitch.pt", copy=True).env({"YOLO_PITCH_MODEL_PATH": "/models/pitch.pt"})


@app.function(image=image, gpu="L4", cpu=2, memory=8192, volumes={"/jobs": volume},
              min_containers=0, max_containers=1, scaledown_window=2, timeout=600,
              startup_timeout=180, retries=0)
def analyze(request_id: str, profile: str = "accurate", tile_size: int = 0):
    import hashlib
    import json
    import torch
    from workers.video.modal_policy import ARTIFACTS, OPTIONAL_ARTIFACTS, reserve_attempt, validate_request, validate_video
    from workers.video.real_analysis import run_real_analysis, video_metadata

    validate_request(request_id, profile, tile_size)
    volume.reload()
    source = Path("/jobs/requests") / request_id / "source.mp4"
    validate_video(source, video_metadata(source))
    ledger_path = Path("/jobs/usage.json")
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    reserve_attempt(ledger)
    ledger_path.write_text(json.dumps(ledger))
    volume.commit()  # Failed attempts also count. Serial execution prevents concurrent ledger writes.
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not provide CUDA; refusing CPU fallback")
    os.environ["ANALYSIS_PROFILE"] = profile
    os.environ["BALL_TILE_SIZE"] = str(tile_size)
    result = run_real_analysis(source, source.parent / "outputs",
                               lambda stage, fraction: print(f"{fraction:.0%} {stage}", flush=True))
    result["detection_summary"].update(compute_provider="modal", gpu=torch.cuda.get_device_name(0),
                                       torch_version=torch.__version__)
    run_dir = source.parent / "outputs" / result["run_id"]
    (run_dir / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    hashes = {}
    for name in (*ARTIFACTS, *(name for name in OPTIONAL_ARTIFACTS if (run_dir / name).is_file())):
        with (run_dir / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    volume.commit()
    return dict(run_id=result["run_id"], sha256=hashes)


# Falls back to a T4 (cheaper, slower) when no L4 is free: this job is not urgent and only runs one small model.
@app.function(image=image, gpu=["L4", "T4"], cpu=2, memory=8192, volumes={"/jobs": volume},
              min_containers=0, max_containers=1, scaledown_window=2, timeout=1800,
              startup_timeout=180, retries=0)
def backfill_pitch(request_ids: str = "", force: bool = False):
    """Add pitch calibration to finished runs without re-tracking them.

    Operator-only: it has no web route and is called from a trusted machine,
    `modal run deploy/modal_app.py::backfill_pitch`. `request_ids` is a
    comma-separated list; empty means every job. It does not count against
    the analysis attempt ledger, and runs that already have calibration are
    skipped unless `force` is set. Returns one outcome per job.
    """
    import json
    from workers.video.modal_policy import validate_request
    from workers.video.pitch_keypoints import calibrate_run
    from workers.video.yolo_adapter import pitch_model_path

    weights = pitch_model_path(Path("/app"))
    if weights is None or not weights.is_file():
        raise RuntimeError("This image was deployed without the pitch landmark model")
    volume.reload()
    jobs = Path("/jobs/web-jobs")
    outcomes = {}
    for request_id in [value.strip() for value in request_ids.split(",") if value.strip()] or sorted(path.stem for path in jobs.glob("*.json")):
        validate_request(request_id)
        job_path = jobs / f"{request_id}.json"
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if job.get("status") != "complete" or not job.get("run_id") or (job.get("result") or {}).get("engine") != "yolo_bytetrack":
                outcomes[request_id] = "skipped: no finished real run"
                continue
            source = Path("/jobs/requests") / request_id / "source.mp4"
            run_dir = source.parent / "outputs" / job["run_id"]
            if not source.is_file() or not (run_dir / "tracking.json").is_file():
                outcomes[request_id] = "skipped: source video or tracking is not stored"
                continue
            if (run_dir / "pitch-calibration.json").is_file() and not force:
                outcomes[request_id] = "skipped: already calibrated"
                continue
            manifest = calibrate_run(source, run_dir, weights, "0")
            volume.commit()
            # Re-read the job: the web gateway may have saved it while this ran.
            volume.reload()
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["result"] = manifest
            job_path.write_text(json.dumps(job), encoding="utf-8")
            volume.commit()
            outcomes[request_id] = manifest["detection_summary"]["pitch_calibration"]["summary"]
        except Exception as error:  # one bad run must not stop the rest
            outcomes[request_id] = f"failed: {type(error).__name__}: {error}"
        print(request_id[:8], outcomes[request_id], flush=True)
    return outcomes


# No GPU: the check draws the stored mappings into the stored video; the landmark model is not used.
@app.function(image=image, cpu=2, memory=4096, volumes={"/jobs": volume}, min_containers=0, max_containers=1,
              scaledown_window=2, timeout=1800, retries=0)
def backfill_lines(request_ids: str = ""):
    """Check finished runs' pitch calibration against their video's lines and store the result.

    Operator-only, like backfill_pitch: `modal run deploy/modal_app.py::backfill_lines`.
    Nothing is rejected or refitted; a clip whose drawn lines mostly miss the video's lines
    gets a warning on its pitch figures. Runs without a stored calibration are skipped.
    """
    import json
    from workers.video.modal_policy import validate_request
    from workers.video.pitch_keypoints import check_run_lines

    volume.reload()
    jobs = Path("/jobs/web-jobs")
    outcomes = {}
    for request_id in [value.strip() for value in request_ids.split(",") if value.strip()] or sorted(path.stem for path in jobs.glob("*.json")):
        validate_request(request_id)
        job_path = jobs / f"{request_id}.json"
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            source = Path("/jobs/requests") / request_id / "source.mp4"
            run_dir = source.parent / "outputs" / str(job.get("run_id"))
            if job.get("status") != "complete" or not source.is_file() or not (run_dir / "pitch-calibration.json").is_file():
                outcomes[request_id] = "skipped: no finished run with stored video and calibration"
                continue
            outcomes[request_id] = check_run_lines(source, run_dir)
            volume.commit()
            volume.reload()  # the web gateway may have saved the job while this ran
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["result"] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            job_path.write_text(json.dumps(job), encoding="utf-8")
            volume.commit()
        except Exception as error:  # one bad run must not stop the rest
            outcomes[request_id] = f"failed: {type(error).__name__}: {error}"
        print(request_id[:8], outcomes[request_id], flush=True)
    return outcomes
