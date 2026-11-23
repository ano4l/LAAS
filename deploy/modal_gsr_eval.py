"""Operator-only evaluation against SoccerNet Game State Reconstruction (SN-GSR-2025). Never deployed.

    python -m modal run deploy/modal_gsr_eval.py --out-dir <folder> [--count 10] [--sequences SNGS-116,SNGS-117] [--rescore-only]
                                                   [--label yolo11x --person yolo11x]

`--person` chooses the detector for people: `off` (the football weights, as first evaluated), `yolov8m` or
`yolo11x` (a general person detector; the football weights then supply only the ball and the roles).
`--label` keeps a run's outputs and report apart on the volume, so an earlier run is not overwritten.

Runs as its own short-lived app with its own volume (`laas-eval`); the analysis worker and the
volume holding users' clips are not touched. For each chosen clip of the dataset's labelled
`test` split it:

1. reads only that clip's bytes out of the 8.8 GB `test.zip` on Hugging Face (one ranged
   download per clip), stores the labels, and re-encodes the frames as a 25 fps H.264 video;
2. runs the same analysis as the worker (`run_real_analysis`: detection, tracking, linking,
   pitch calibration), then the experimental shirt-number reader with one-digit proposals
   allowed so they can be scored too;
3. scores the run with `workers/video/gsr_benchmark.py` and returns the report.

`--rescore-only` skips steps 1 and 2 and scores stored runs again, on a CPU.

Dataset: https://huggingface.co/datasets/SoccerNet/SN-GSR-2025 (public, not gated, tagged GPL-3.0).
"""
from pathlib import Path
import os
import sys

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/app")
sys.path.insert(0, str(ROOT))

DATASET_URL = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/test.zip"
PARSEQ_COMMIT = "1902db043c029a7e03a3818c616c06600af574be"
app = modal.App("laas-gsr-eval")
volume = modal.Volume.from_name("laas-eval", create_if_missing=True)
DATA = Path("/eval/sn-gsr-2025/test")

if modal.is_local():
    from workers.video.yolo_adapter import model_path, pitch_model_path
    weights, pitch_weights = model_path(ROOT), pitch_model_path(ROOT)
    if not weights.is_file() or pitch_weights is None or not pitch_weights.is_file():
        raise FileNotFoundError("The detector and pitch landmark weights must be present, as for deploy/modal_app.py")
    rescue_weights = ROOT / "references/Soccer-Analytics/yolov8m.pt"
else:
    weights = pitch_weights = rescue_weights = None

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("numpy", "opencv-python-headless>=4.9", "imageio-ffmpeg>=0.6.0")
    .add_local_dir(str(ROOT / "workers"), "/app/workers", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .env({"PYTHONPATH": "/app"})
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install("torch==2.13.0", "torchvision==0.28.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ultralytics==8.4.178", "lap>=0.5.12", "imageio-ffmpeg>=0.6.0", "pytorch-lightning>=2.2", "timm>=0.9.16", "nltk", "pyyaml")
    .run_commands(f"git clone https://github.com/baudm/parseq /opt/parseq && cd /opt/parseq && git checkout {PARSEQ_COMMIT}",
                  "python -c \"import torch; torch.hub.load('/opt/parseq', 'parseq', pretrained=True, source='local')\"")
    # A general person detector from Ultralytics' own release assets, fetched once while the image is built.
    .run_commands("mkdir -p /models && cd /models && python -c \"from ultralytics import YOLO; YOLO('yolo11x.pt')\"")
    .add_local_dir(str(ROOT / "workers"), "/app/workers", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
)
if modal.is_local():
    gpu_image = (gpu_image.add_local_file(str(weights), "/models/football.pt", copy=True)
                 .add_local_file(str(rescue_weights), "/models/ball-rescue.pt", copy=True)
                 .add_local_file(str(pitch_weights), "/models/pitch.pt", copy=True))
gpu_image = gpu_image.env({"PYTHONPATH": "/app", "YOLO_MODEL_PATH": "/models/football.pt", "YOLO_DEVICE": "0", "BALL_CONFIDENCE": "0.1",
                           "OBJECT_CONFIDENCE": "0.45", "TRACKER_BACKEND": "botsort", "YOLO_BALL_RESCUE_PATH": "/models/ball-rescue.pt",
                           "YOLO_PITCH_MODEL_PATH": "/models/pitch.pt", "ANALYSIS_PROFILE": "accurate", "BALL_TILE_SIZE": "0"})


def _remote_zip():
    """A seekable view of the remote zip that reads byte ranges over HTTP; `cache` serves one downloaded span locally."""
    import io
    import urllib.request

    class RangeFile(io.RawIOBase):
        def __init__(self, url):
            request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "laas-eval"})
            with urllib.request.urlopen(request, timeout=60) as response:
                self.url, self.size = response.url, int(response.headers["Content-Length"])
            self.pos, self.span = 0, None

        def seekable(self):
            return True

        def readable(self):
            return True

        def tell(self):
            return self.pos

        def seek(self, offset, whence=0):
            self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
            return self.pos

        def fetch(self, start, end, path):
            """Download bytes [start, end) to `path` in one request and serve reads inside it from there."""
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": "laas-eval"})
            with urllib.request.urlopen(request, timeout=600) as response, open(path, "wb") as out:
                while chunk := response.read(1 << 22):
                    out.write(chunk)
            self.span = (start, end, open(path, "rb"))

        def readinto(self, buffer):
            if self.pos >= self.size:
                return 0
            if self.span and self.span[0] <= self.pos < self.span[1]:
                start, end, stream = self.span
                stream.seek(self.pos - start)
                data = stream.read(min(len(buffer), end - self.pos))
            else:
                last = min(self.size, self.pos + len(buffer)) - 1
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{last}", "User-Agent": "laas-eval"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
            buffer[:len(data)] = data
            self.pos += len(data)
            return len(data)

    return RangeFile


@app.function(image=cpu_image, cpu=2, memory=4096, volumes={"/eval": volume}, max_containers=4, timeout=1800, retries=0)
def fetch(sequence: str):
    """Store one clip's labels and a 25 fps video of its frames. Returns what was stored."""
    import io
    import shutil
    import subprocess
    import tempfile
    import zipfile
    import imageio_ffmpeg

    target = DATA / sequence
    if (target / "source.mp4").is_file() and (target / "Labels-GameState.json").is_file():
        return dict(sequence=sequence, stored="already")
    remote = _remote_zip()(DATASET_URL)
    archive = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))
    members = [info for info in archive.infolist() if info.filename.startswith(sequence + "/") and not info.is_dir()]
    if not members:
        raise ValueError(f"{sequence} is not in the test split")
    start = min(info.header_offset for info in members)
    end = max(info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + info.compress_size for info in members) + 1024
    work = Path(tempfile.mkdtemp())
    remote.fetch(start, min(end, remote.size), work / "span.bin")
    frames = work / "img1"
    frames.mkdir()
    target.mkdir(parents=True, exist_ok=True)
    for info in members:
        name = info.filename[len(sequence) + 1:]
        if name == "Labels-GameState.json":
            (target / name).write_bytes(archive.read(info))
        elif name.startswith("img1/") and name.endswith(".jpg"):
            (frames / Path(name).name).write_bytes(archive.read(info))
    count = len(list(frames.glob("*.jpg")))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-framerate", "25", "-i", str(frames / "%06d.jpg"),
                    "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", str(target / "source.mp4")], check=True)
    shutil.rmtree(work)
    volume.commit()
    return dict(sequence=sequence, frames=count, span_mb=round((end - start) / 1e6, 1), video_mb=round((target / "source.mp4").stat().st_size / 1e6, 1))


PERSON_MODELS = {"off": "off", "yolov8m": "/models/ball-rescue.pt", "yolo11x": "/models/yolo11x.pt"}


def _named(name, label):
    """`outputs`, `report.json` and `shirt-numbers.json` for the first run; the same with the label added for a later one."""
    stem, dot, suffix = name.partition(".")
    return f"{stem}-{label}{dot}{suffix}" if label else name


def _score(sequence, label=""):
    import json
    from workers.video import gsr_benchmark as gb
    from workers.video import merge_suggestions as ms

    target = DATA / sequence
    run_dir = next((target / _named("outputs", label)).iterdir())
    rows = json.loads((run_dir / "tracking.json").read_text(encoding="utf-8"))
    document = json.loads((target / "Labels-GameState.json").read_text(encoding="utf-8"))
    labels = gb.load_labels(document)
    pitch = None
    if (run_dir / "pitch-calibration.json").is_file():
        pitch = {int(k): v for k, v in json.loads((run_dir / "pitch-calibration.json").read_text(encoding="utf-8")).get("frames", {}).items()}
    ms.MAX_SUGGESTIONS = 10 ** 6  # every suggestion is judged, not only the ones the profile would list
    suggestions = ms.suggest(rows, labels["fps"], pitch)
    numbers_path = target / _named("shirt-numbers.json", label)
    numbers = json.loads(numbers_path.read_text(encoding="utf-8")) if numbers_path.is_file() else None
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) if (run_dir / "manifest.json").is_file() else {}
    report = gb.score(rows, labels, pitch, suggestions, numbers)
    from workers.video import ball_benchmark as bb
    balls, people_boxes = bb.load_balls(document)
    report["ball"] = bb.score(rows, balls, people_boxes, labels["frame_count"])
    # The same detections with the ball path chosen again by the code in this checkout, so a change to the
    # path rules can be measured without running the detector again.
    import copy
    from workers.video.ball_path import select_ball_path
    from workers.video.camera_motion import CameraMotion
    again = copy.deepcopy(rows)
    motion = json.loads((run_dir / "camera-motion.json").read_text(encoding="utf-8")) if (run_dir / "camera-motion.json").is_file() else {}
    warps = {int(k): v for k, v in (motion.get("warps") or {}).items()}
    camera = CameraMotion.from_warps(warps, {int(k): v for k, v in (motion.get("scenes") or {}).items()}) if warps else None
    path = select_ball_path(again, manifest.get("metadata") or dict(fps=labels["fps"], width=1920, height=1080), camera=camera, pitch=pitch)
    report["ball_path_again"] = dict(bb.score(again, balls, people_boxes, labels["frame_count"]),
                                     path={k: path.get(k) for k in ("version", "selected", "segments", "static_candidates_rejected", "painted_spot_candidates_rejected")})
    report["run"] = dict(metadata={k: manifest.get("metadata", {}).get(k) for k in ("width", "height", "fps", "frame_count", "duration_seconds")},
                         calibration_summary=((manifest.get("detection_summary") or {}).get("pitch_calibration") or {}).get("summary"),
                         linking=(manifest.get("detection_summary") or {}).get("track_linking"),
                         person_model=(manifest.get("detection_summary") or {}).get("person_model"),
                         processing_seconds=(manifest.get("detection_summary") or {}).get("processing_seconds"))
    (target / _named("report.json", label)).write_text(json.dumps(report), encoding="utf-8")
    return report


@app.function(image=gpu_image, gpu=["L4", "T4"], cpu=2, memory=12288, volumes={"/eval": volume}, max_containers=2, timeout=2400, retries=0)
def analyse(sequence: str, label: str = "", person: str = "off"):
    import json
    import shutil
    import torch
    from workers.video import shirt_numbers as sn
    from workers.video.real_analysis import run_real_analysis

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU")
    volume.reload()
    target = DATA / sequence
    os.environ["YOLO_PERSON_MODEL_PATH"] = PERSON_MODELS[person]
    outputs = target / _named("outputs", label)
    shutil.rmtree(outputs, ignore_errors=True)
    result = run_real_analysis(target / "source.mp4", outputs, lambda stage, fraction: None)
    run_dir = outputs / result["run_id"]
    (run_dir / "manifest.json").write_text(json.dumps(result), encoding="utf-8")
    sn.PROPOSE_ONE_DIGIT = True  # scored separately, so the product setting can be checked
    reader = sn.ParseqReader("/opt/parseq", device="cuda")
    numbers = sn.read_run(target / "source.mp4", json.loads((run_dir / "tracking.json").read_text(encoding="utf-8")), reader)
    (target / _named("shirt-numbers.json", label)).write_text(json.dumps(numbers), encoding="utf-8")
    volume.commit()
    report = _score(sequence, label)
    volume.commit()
    return report


@app.function(image=cpu_image, cpu=2, memory=4096, volumes={"/eval": volume}, max_containers=4, timeout=900, retries=0)
def rescore(sequence: str, label: str = ""):
    volume.reload()
    report = _score(sequence, label)
    volume.commit()
    return report


@app.function(image=cpu_image, cpu=2, memory=4096, volumes={"/eval": volume}, max_containers=5, timeout=1800, retries=0)
def line_study(sequence: str):
    """Per calibrated frame: line-support scores of its mapping, before and after snapping to the lines, and the labelled error."""
    import json
    import cv2
    import numpy as np
    from workers.video import gsr_benchmark as gb
    from workers.video import pitch_lines as pl

    volume.reload()
    target = DATA / sequence
    run_dir = next((target / "outputs").iterdir())
    pitch = {int(k): v for k, v in json.loads((run_dir / "pitch-calibration.json").read_text(encoding="utf-8"))["frames"].items()}
    labels = gb.load_labels(json.loads((target / "Labels-GameState.json").read_text(encoding="utf-8")))
    sx, sy = json.loads((target / "report.json").read_text(encoding="utf-8"))["calibration"]["orientation"]

    def error(matrix, frame):
        out = []
        for p in labels["frames"].get(frame, []):
            if p["role"] in gb.PEOPLE and p["pitch"]:
                xy = gb._ours(dict(status="calibrated", matrix=list(np.asarray(matrix).ravel())), p["box"])
                if xy:
                    out.append(float(np.hypot(sx * xy[0] - p["pitch"][0], sy * xy[1] - p["pitch"][1])))
        return (round(float(np.median(out)), 3), len(out)) if out else (None, 0)

    rows = []
    capture = cv2.VideoCapture(str(target / "source.mp4"))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        record = pitch.get(index)
        if record and record.get("status") == "calibrated":
            mask = pl.line_mask(frame)
            distance = cv2.distanceTransform(np.where(mask, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
            row = dict(frame=index, landmarks=record.get("landmarks_used"), fit_error_m=record.get("mean_error_m"), stabilised=record.get("stabilised"),
                       mask_share=round(float(mask.mean()), 4))
            row["error_m"], row["people"] = error(record["matrix"], index)
            for name, tolerance in (("tight", .005), ("mid", .015), ("loose", .03)):
                row[name] = pl.support(mask, record["matrix"], distance, tolerance)
            refined = pl.refine(mask, record["matrix"])
            if refined is not None:
                row["refined"] = pl.support(mask, refined, distance)
                row["refined_error_m"], _ = error(refined, index)
            rows.append(row)
        index += 1
    capture.release()
    (target / "line-study.json").write_text(json.dumps(rows), encoding="utf-8")
    return dict(sequence=sequence, rows=rows)


@app.local_entrypoint()
def main(out_dir: str, count: int = 10, sequences: str = "", rescore_only: bool = False, lines_only: bool = False, label: str = "", person: str = "off"):
    import io
    import json
    import zipfile

    chosen = [s.strip() for s in sequences.split(",") if s.strip()]
    if not chosen:
        remote = _remote_zip()(DATASET_URL)
        names = sorted({info.filename.split("/")[0] for info in zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)).infolist()
                        if info.filename.endswith("/Labels-GameState.json")})
        chosen = [names[round(i * len(names) / count)] for i in range(count)]
    print("clips", chosen, flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if lines_only:
        for sequence, result in zip(chosen, line_study.map(chosen, return_exceptions=True)):
            if isinstance(result, Exception):
                print(sequence, "failed", type(result).__name__, result, flush=True)
            else:
                (out / f"{sequence}-lines.json").write_text(json.dumps(result["rows"]), encoding="utf-8")
                print(sequence, "frames", len(result["rows"]), flush=True)
        return
    if person not in PERSON_MODELS:
        raise SystemExit(f"--person must be one of {', '.join(PERSON_MODELS)}")
    if rescore_only:
        reports = list(rescore.map(chosen, kwargs=dict(label=label), return_exceptions=True))
    else:
        for stored in fetch.map(chosen, return_exceptions=True):
            print("fetched", stored, flush=True)
        reports = list(analyse.map(chosen, kwargs=dict(label=label, person=person), return_exceptions=True))
    for sequence, report in zip(chosen, reports):
        if isinstance(report, Exception):
            print(sequence, "failed", type(report).__name__, report, flush=True)
            continue
        (out / f"{sequence}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(sequence, json.dumps(dict(detection=(report["detection"]["recall"], report["detection"]["precision"]),
                                        identities_per_player=report["identities"]["identities_per_followed_player"],
                                        players_followed=(report["identities"]["labelled_players_followed"], report["identities"]["labelled_players"]),
                                        position=report["position_error"], calibration=report["calibration"]["share"],
                                        speed=(report["motion"]["ours"]["mean_speed_m_s"], report["motion"]["labelled"]["mean_speed_m_s"]),
                                        suggestions={k: report["suggestions"][k] for k in ("total", "same", "different", "unknown")})), flush=True)
    print("written", out)
