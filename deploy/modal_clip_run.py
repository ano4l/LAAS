"""Operator-only: analyse one of your own clips on a GPU with the code in this checkout, without touching the deployed worker.

    python -m modal volume put laas-eval path/to/clip.mp4 clips/<name>/<name>.mp4
    python -m modal run deploy/modal_clip_run.py --name <name> [--person yolov8m]
    python -m modal volume get laas-eval clips/<name>/outputs/<run id> data/storage/outputs/

It runs the same analysis as the worker (`run_real_analysis`) and then the experimental shirt-number
reader with the product's settings, and stores the run beside the clip on the `laas-eval` volume.
Copying the run folder into the local output folder makes it show up in the local library. The
deployed analysis app and the volume holding users' clips are not used.
"""
from pathlib import Path
import os
import sys

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/app")
sys.path.insert(0, str(ROOT))

PARSEQ_COMMIT = "1902db043c029a7e03a3818c616c06600af574be"
PERSON_MODELS = {"off": "off", "yolov8m": "/models/ball-rescue.pt", "yolo11x": "/models/yolo11x.pt"}
app = modal.App("laas-clip-run")
volume = modal.Volume.from_name("laas-eval", create_if_missing=True)
CLIPS = Path("/eval/clips")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install("torch==2.13.0", "torchvision==0.28.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ultralytics==8.4.178", "lap>=0.5.12", "imageio-ffmpeg>=0.6.0", "pytorch-lightning>=2.2", "timm>=0.9.16", "nltk", "pyyaml")
    .run_commands(f"git clone https://github.com/baudm/parseq /opt/parseq && cd /opt/parseq && git checkout {PARSEQ_COMMIT}",
                  "python -c \"import torch; torch.hub.load('/opt/parseq', 'parseq', pretrained=True, source='local')\"")
    .run_commands("mkdir -p /models && cd /models && python -c \"from ultralytics import YOLO; YOLO('yolo11x.pt')\"")
    .add_local_dir(str(ROOT / "workers"), "/app/workers", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
)
if modal.is_local():
    from workers.video.yolo_adapter import model_path, pitch_model_path
    weights, pitch_weights, rescue_weights = model_path(ROOT), pitch_model_path(ROOT), ROOT / "references/Soccer-Analytics/yolov8m.pt"
    if not weights.is_file() or pitch_weights is None or not pitch_weights.is_file() or not rescue_weights.is_file():
        raise FileNotFoundError("The detector, pitch landmark and yolov8m weights must be present, as for deploy/modal_gsr_eval.py")
    image = (image.add_local_file(str(weights), "/models/football.pt", copy=True)
             .add_local_file(str(rescue_weights), "/models/ball-rescue.pt", copy=True)
             .add_local_file(str(pitch_weights), "/models/pitch.pt", copy=True))
image = image.env({"PYTHONPATH": "/app", "YOLO_MODEL_PATH": "/models/football.pt", "YOLO_DEVICE": "0", "BALL_CONFIDENCE": "0.1",
                   "OBJECT_CONFIDENCE": "0.45", "TRACKER_BACKEND": "botsort", "YOLO_BALL_RESCUE_PATH": "/models/ball-rescue.pt",
                   "YOLO_PITCH_MODEL_PATH": "/models/pitch.pt", "ANALYSIS_PROFILE": "accurate", "BALL_TILE_SIZE": "0"})


@app.function(image=image, gpu=["L4", "A10G", "T4"], cpu=2, memory=12288, volumes={"/eval": volume}, max_containers=2, timeout=2400, retries=0)
def analyse_clip(name: str, person: str = "yolov8m"):
    import json
    import shutil
    import torch
    from workers.video import shirt_numbers as sn
    from workers.video.real_analysis import run_real_analysis

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU")
    volume.reload()
    target = CLIPS / name
    source = target / f"{name}.mp4"
    if not source.is_file():
        raise FileNotFoundError(f"Put the clip at clips/{name}/{name}.mp4 on the laas-eval volume first")
    os.environ["YOLO_PERSON_MODEL_PATH"] = PERSON_MODELS[person]
    shutil.rmtree(target / "outputs", ignore_errors=True)
    result = run_real_analysis(source, target / "outputs", lambda stage, fraction: print(f"{fraction:.0%} {stage}", flush=True))
    result["detection_summary"].update(compute_provider="modal", gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    run_dir = target / "outputs" / result["run_id"]
    (run_dir / "manifest.json").write_text(json.dumps(result), encoding="utf-8")
    # The reader with the product's own settings: one-digit numbers are counted but never proposed.
    numbers = sn.read_run(source, json.loads((run_dir / "tracking.json").read_text(encoding="utf-8")), sn.ParseqReader("/opt/parseq", device="cuda"))
    (run_dir / "shirt-numbers.json").write_text(json.dumps(numbers), encoding="utf-8")
    volume.commit()
    summary = result["detection_summary"]
    return dict(name=name, run_id=result["run_id"], gpu=summary.get("gpu"), seconds=summary.get("processing_seconds"), person_model=summary.get("person_model"),
                identities=len(summary.get("linked_tracks") or []), tracks=summary.get("unique_tracks"), roles=summary.get("role_settling"), off_pitch=summary.get("off_pitch"),
                calibration=(summary.get("pitch_calibration") or {}).get("summary"), shirt_number_proposals=numbers.get("identities_proposed"),
                files=sorted(path.name for path in run_dir.iterdir()))


@app.local_entrypoint()
def main(name: str, person: str = "yolov8m"):
    import json
    if person not in PERSON_MODELS:
        raise SystemExit(f"--person must be one of {', '.join(PERSON_MODELS)}")
    print(json.dumps(analyse_clip.remote(name, person), indent=1))
