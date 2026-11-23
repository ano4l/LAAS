"""Operator-only comparison of ways to detect the ball on the stored SoccerNet GSR clips. Never deployed.

    python -m modal run deploy/modal_ball_eval.py --out-dir <folder> [--sequences SNGS-116,SNGS-121] [--every 5]
    python scripts/score_ball_detectors.py <folder>

Uses the clips `deploy/modal_gsr_eval.py` already stored on the `laas-eval` volume; it fetches nothing
and writes nothing there. For every Nth frame of each clip it runs each detector once at a low
confidence, on the whole frame and on overlapping tiles at the frame's own resolution, and returns
the raw ball boxes with the labelled ball and people for those frames. Thresholds and scoring are
then done locally on a CPU from the saved file (`workers/video/ball_benchmark.py` has the matching rule).
"""
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/app")
sys.path.insert(0, str(ROOT))

app = modal.App("laas-ball-eval")
volume = modal.Volume.from_name("laas-eval", create_if_missing=False)
DATA = Path("/eval/sn-gsr-2025/test")
MIN_CONFIDENCE = .03  # boxes are kept from here up, so any higher threshold can be applied afterwards
BALLS = {"ball", "sports ball"}
TILE = 960
# name -> (weights file in /models, whole-frame image sizes, tile image sizes)
DETECTORS = {
    "football": ("football.pt", (1280, 1920), (960,)),      # the ball detector the app uses today, at 1280
    "coco-yolov8m": ("yolov8m.pt", (1280,), (960,)),         # shipped already as the secondary ball model
    "coco-yolo11x": ("yolo11x.pt", (1280, 1920), (960, 1280)),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.13.0", "torchvision==0.28.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ultralytics==8.4.178", "lap>=0.5.12", "imageio-ffmpeg>=0.6.0")
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


@app.function(image=image, gpu=["L4", "T4", "A10G"], cpu=2, memory=12288, volumes={"/eval": volume}, max_containers=4, timeout=2400, retries=0)
def detect(sequence: str, every: int = 5, label: str = "yolov8m-kits"):
    import json
    import time
    import cv2
    import torch
    from ultralytics import YOLO
    from workers.video import ball_benchmark as bb
    from workers.video.accuracy import tile_windows

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU")
    target = DATA / sequence
    balls, people = bb.load_balls(json.loads((target / "Labels-GameState.json").read_text(encoding="utf-8")))
    stored = {}
    run_dirs = sorted((target / f"outputs-{label}").iterdir()) if (target / f"outputs-{label}").is_dir() else []
    if run_dirs and (run_dirs[0] / "tracking.json").is_file():
        for row in json.loads((run_dirs[0] / "tracking.json").read_text(encoding="utf-8")):
            if row.get("object_type") in BALLS and row["frame_idx"] % every == 0:
                stored.setdefault(row["frame_idx"], []).append([*[round(v, 1) for v in row["bbox"]], row["confidence"], bool(row.get("ball_selected")), row.get("model_role")])

    models = {name: YOLO(f"/models/{spec[0]}") for name, spec in DETECTORS.items()}
    ids = {name: [k for k, v in model.names.items() if str(v).lower() in BALLS] for name, model in models.items()}
    runs = {**{f"{name}@{size}": {} for name, spec in DETECTORS.items() for size in spec[1]},
            **{f"{name}-tiles@{size}": {} for name, spec in DETECTORS.items() for size in spec[2]}}
    seconds = dict.fromkeys(runs, 0.)
    frames = []
    capture = cv2.VideoCapture(str(target / "source.mp4"))
    frame_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    windows = list(tile_windows(frame_size[0], frame_size[1], TILE))

    def boxes(result, dx=0, dy=0):
        return [[round(float(a) + dx, 1), round(float(b) + dy, 1), round(float(c) + dx, 1), round(float(d) + dy, 1), round(float(box.conf.item()), 3)]
                for box in result.boxes for a, b, c, d in [box.xyxy[0].tolist()]]

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            frames.append(index)
            for name, (_, sizes, tile_sizes) in DETECTORS.items():
                model = models[name]
                for size in sizes:
                    started = time.perf_counter()
                    result = model.predict(frame, classes=ids[name], imgsz=size, conf=MIN_CONFIDENCE, iou=.45, device=0, verbose=False)[0]
                    seconds[f"{name}@{size}"] += time.perf_counter() - started
                    runs[f"{name}@{size}"][index] = boxes(result)
                for size in tile_sizes:
                    started = time.perf_counter()
                    results = model.predict([frame[y1:y2, x1:x2] for x1, y1, x2, y2 in windows], classes=ids[name], imgsz=size, conf=MIN_CONFIDENCE, iou=.45, device=0, verbose=False)
                    seconds[f"{name}-tiles@{size}"] += time.perf_counter() - started
                    runs[f"{name}-tiles@{size}"][index] = [box for (x1, y1, _, _), result in zip(windows, results) for box in boxes(result, x1, y1)]
        index += 1
    capture.release()
    return dict(sequence=sequence, every=every, frames=frames, size=frame_size, tiles=windows,
                balls={f: [round(v, 1) for v in balls[f]] for f in frames if f in balls},
                people={f: [[round(v, 1) for v in box] for box in people.get(f, [])] for f in frames},
                runs=runs, stored_pipeline=stored, ms_per_frame={name: round(1000 * total / max(1, len(frames)), 1) for name, total in seconds.items()},
                gpu=torch.cuda.get_device_name(0), min_confidence=MIN_CONFIDENCE)


@app.local_entrypoint()
def main(out_dir: str, sequences: str = "", every: int = 5, label: str = "yolov8m-kits"):
    import json
    import subprocess

    chosen = [s.strip() for s in sequences.split(",") if s.strip()]
    if not chosen:
        listing = subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "laas-eval", "sn-gsr-2025/test"], capture_output=True, text=True, check=True).stdout
        chosen = sorted(line.strip().rsplit("/", 1)[-1] for line in listing.splitlines() if "SNGS-" in line)
    print("clips", chosen, flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sequence, result in zip(chosen, detect.map(chosen, kwargs=dict(every=every, label=label), return_exceptions=True)):
        if isinstance(result, Exception):
            print(sequence, "failed", type(result).__name__, result, flush=True)
            continue
        (out / f"{sequence}.json").write_text(json.dumps(result), encoding="utf-8")
        print(sequence, result["gpu"], "frames", len(result["frames"]), "ms/frame", result["ms_per_frame"], flush=True)
    print("written", out)
