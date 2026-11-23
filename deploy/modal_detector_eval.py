"""Operator-only detector comparison on the stored SoccerNet GSR clips. Never deployed.

    python -m modal run deploy/modal_detector_eval.py --out-dir <folder> [--sequences SNGS-116,SNGS-121] [--every 5]

Uses the clips `deploy/modal_gsr_eval.py` already stored on the `laas-eval` volume; it fetches
nothing and writes nothing there. For every Nth frame of each clip it runs each detector once at a
low confidence and returns the raw boxes, with the labelled boxes for those frames, the field
outline the pipeline's filter would use, and the stored pipeline's own boxes. Everything else
(thresholds, filters, scoring) is then done locally on a CPU from the saved file, so trying an
idea costs no GPU time. Scoring is `workers/video/detection_diagnosis.py`.
"""
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/app")
sys.path.insert(0, str(ROOT))

app = modal.App("laas-detector-eval")
volume = modal.Volume.from_name("laas-eval", create_if_missing=False)
DATA = Path("/eval/sn-gsr-2025/test")
MIN_CONFIDENCE = .05  # boxes are kept from here up, so any higher threshold can be applied afterwards
# name -> (weights file in /models, classes to keep or None for all, image sizes to run)
DETECTORS = {
    "football": ("football.pt", None, (1280, 1920)),       # the detector the app uses today
    "coco-yolov8m": ("yolov8m.pt", ("person",), (1280,)),  # general person detector, already shipped for the ball
    "coco-yolo11x": ("yolo11x.pt", ("person",), (1280, 1920)),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.13.0", "torchvision==0.28.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ultralytics==8.4.178", "lap>=0.5.12", "imageio-ffmpeg>=0.6.0")
    # The general model comes from Ultralytics' own release assets, fetched once while the image is built.
    .run_commands("mkdir -p /models && cd /models && python -c \"from ultralytics import YOLO; YOLO('yolo11x.pt')\"")
    .add_local_dir(str(ROOT / "workers"), "/app/workers", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
)
if modal.is_local():
    from workers.video.yolo_adapter import model_path
    weights, general = model_path(ROOT), ROOT / "references/Soccer-Analytics/yolov8m.pt"
    if not weights.is_file() or not general.is_file():
        raise FileNotFoundError("The football detector and yolov8m weights must be present, as for deploy/modal_gsr_eval.py")
    image = image.add_local_file(str(weights), "/models/football.pt", copy=True).add_local_file(str(general), "/models/yolov8m.pt", copy=True)
image = image.env({"PYTHONPATH": "/app"})


@app.function(image=image, gpu=["L4", "T4", "A10G"], cpu=2, memory=12288, volumes={"/eval": volume}, max_containers=3, timeout=1800, retries=0)
def detect(sequence: str, every: int = 5):
    import json
    import time
    import cv2
    import torch
    from ultralytics import YOLO
    from workers.video import gsr_benchmark as gb
    from workers.video.vision_quality import field_region

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU")
    target = DATA / sequence
    labels = gb.load_labels(json.loads((target / "Labels-GameState.json").read_text(encoding="utf-8")))
    stored = {}
    run_dirs = sorted((target / "outputs").iterdir()) if (target / "outputs").is_dir() else []
    if run_dirs and (run_dirs[0] / "tracking.json").is_file():
        for row in json.loads((run_dirs[0] / "tracking.json").read_text(encoding="utf-8")):
            if row.get("object_type") in {"player", "goalkeeper", "referee", "person"} and row["frame_idx"] % every == 0:
                stored.setdefault(row["frame_idx"], []).append([*[round(v, 1) for v in row["bbox"]], row["confidence"], row["object_type"], row.get("tracking_association")])

    models = {name: YOLO(f"/models/{spec[0]}") for name, spec in DETECTORS.items()}
    runs = {f"{name}@{size}": {} for name, spec in DETECTORS.items() for size in spec[2]}
    seconds = dict.fromkeys(runs, 0.)
    frames, fields, kept_labels = [], {}, {}
    capture = cv2.VideoCapture(str(target / "source.mp4"))
    frame_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            frames.append(index)
            kept_labels[index] = [[*[round(v, 1) for v in p["box"]], p["role"], p["team"], p["track"]] for p in labels["frames"].get(index, [])]
            region = field_region(frame)
            fields[index] = None if region is None else [[round(float(x), 1), round(float(y), 1)] for x, y in region.reshape(-1, 2)]
            for name, (_, classes, sizes) in DETECTORS.items():
                model = models[name]
                ids = None if classes is None else [k for k, v in model.names.items() if str(v).lower() in classes]
                for size in sizes:
                    started = time.perf_counter()
                    result = model.predict(frame, classes=ids, imgsz=size, conf=MIN_CONFIDENCE, iou=.45, device=0, verbose=False)[0]
                    seconds[f"{name}@{size}"] += time.perf_counter() - started
                    runs[f"{name}@{size}"][index] = [[*[round(float(v), 1) for v in box.xyxy[0].tolist()], round(float(box.conf.item()), 3), str(result.names[int(box.cls.item())]).lower()]
                                                     for box in result.boxes]
        index += 1
    capture.release()
    # The first frame of each run includes model warm-up, so the timing is a little high for the first clip in a container.
    return dict(sequence=sequence, every=every, frames=frames, size=frame_size, labels=kept_labels, field=fields,
                runs=runs, stored_pipeline=stored, ms_per_frame={name: round(1000 * total / max(1, len(frames)), 1) for name, total in seconds.items()},
                gpu=torch.cuda.get_device_name(0), min_confidence=MIN_CONFIDENCE, action=labels["action"])


@app.local_entrypoint()
def main(out_dir: str, sequences: str = "", every: int = 5):
    import json
    import subprocess

    chosen = [s.strip() for s in sequences.split(",") if s.strip()]
    if not chosen:
        listing = subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "laas-eval", "sn-gsr-2025/test"], capture_output=True, text=True, check=True).stdout
        chosen = sorted(line.strip().rsplit("/", 1)[-1] for line in listing.splitlines() if "SNGS-" in line)
    print("clips", chosen, flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sequence, result in zip(chosen, detect.map(chosen, kwargs=dict(every=every), return_exceptions=True)):
        if isinstance(result, Exception):
            print(sequence, "failed", type(result).__name__, result, flush=True)
            continue
        (out / f"{sequence}.json").write_text(json.dumps(result), encoding="utf-8")
        print(sequence, result["gpu"], "frames", len(result["frames"]), "ms/frame", result["ms_per_frame"], flush=True)
    print("written", out)
