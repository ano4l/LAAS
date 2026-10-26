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
    from workers.video.yolo_adapter import model_path
    weights = model_path(ROOT)
    if not weights.is_file():
        raise FileNotFoundError("Set YOLO_MODEL_PATH to trusted football weights before deploying Modal")
else:
    weights = Path("/models/football.pt")
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


@app.function(image=image, gpu="L4", cpu=2, memory=8192, volumes={"/jobs": volume},
              min_containers=0, max_containers=1, scaledown_window=2, timeout=600,
              startup_timeout=180, retries=0)
def analyze(request_id: str, profile: str = "accurate", tile_size: int = 0):
    import hashlib
    import json
    import torch
    from workers.video.modal_policy import ARTIFACTS, reserve_attempt, validate_request, validate_video
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
    for name in ARTIFACTS:
        with (run_dir / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    volume.commit()
    return dict(run_id=result["run_id"], sha256=hashes)
