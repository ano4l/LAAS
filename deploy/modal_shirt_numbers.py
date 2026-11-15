"""Operator-only experiment: read shirt numbers on finished runs. Never deployed.

    python -m modal run deploy/modal_shirt_numbers.py --out-dir <folder> [--request-ids a,b] [--save]

Without --save nothing is written to the volume. With --save each run gets a
`shirt-numbers.json` beside its other outputs, which the Players tab shows as
unreviewed proposals.

It runs as its own short-lived app, separate from the analysis worker, reads each run's
stored source video and detections, and returns proposals plus a few sample crops so a
person can check them. PARSeq (Apache-2.0) is cloned at a pinned commit; its pretrained
weights come from that repository's v1.0.0 release and are hash-checked by torch.hub.
"""
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/app")
sys.path.insert(0, str(ROOT))
from workers.video.modal_policy import VOLUME_NAME

PARSEQ_COMMIT = "1902db043c029a7e03a3818c616c06600af574be"
app = modal.App("laas-shirt-numbers-test")
volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install("torch==2.13.0", "torchvision==0.28.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("opencv-python-headless>=4.9", "numpy", "pytorch-lightning>=2.2", "timm>=0.9.16", "nltk", "pyyaml")
    .run_commands(f"git clone https://github.com/baudm/parseq /opt/parseq && cd /opt/parseq && git checkout {PARSEQ_COMMIT}",
                  "python -c \"import torch; torch.hub.load('/opt/parseq', 'parseq', pretrained=True, source='local')\"")
    .add_local_dir(str(ROOT / "workers"), "/app/workers", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .env({"PYTHONPATH": "/app"})
)


@app.function(image=image, gpu=["L4", "T4"], cpu=2, memory=8192, volumes={"/jobs": volume}, max_containers=1, timeout=1800, retries=0)
def read_numbers(request_ids: str = "", samples_per_identity: int = 6, save: bool = False):
    import base64
    import json
    import cv2
    from workers.video.modal_policy import validate_request
    from workers.video.shirt_numbers import ParseqReader, read_run

    reader = ParseqReader("/opt/parseq", device="cuda")
    jobs = Path("/jobs/web-jobs")
    outcomes = {}
    for request_id in [value.strip() for value in request_ids.split(",") if value.strip()] or sorted(path.stem for path in jobs.glob("*.json")):
        validate_request(request_id)
        try:
            job = json.loads((jobs / f"{request_id}.json").read_text(encoding="utf-8"))
            source = Path("/jobs/requests") / request_id / "source.mp4"
            run_dir = source.parent / "outputs" / str(job.get("run_id"))
            if job.get("status") != "complete" or not source.is_file() or not (run_dir / "tracking.json").is_file():
                outcomes[request_id] = dict(skipped="no finished run with stored video and detections")
                continue
            kept = {}

            def sink(identity, frame, crop, text, confidence):
                kept.setdefault(identity, []).append((confidence if text.isdigit() else 0., frame, text, crop))

            result = read_run(source, json.loads((run_dir / "tracking.json").read_text(encoding="utf-8")), reader, sink)
            if save:
                (run_dir / "shirt-numbers.json").write_text(json.dumps(result), encoding="utf-8")
                volume.commit()
            crops = {}
            for proposal in result["proposals"]:
                if proposal["confident_reads"]:
                    best = sorted(kept.get(proposal["track_id"], []), key=lambda item: -item[0])[:samples_per_identity]
                    crops[proposal["track_id"]] = [dict(frame=frame, text=text, confidence=round(confidence, 3),
                                                        jpeg=base64.b64encode(cv2.imencode(".jpg", crop)[1].tobytes()).decode())
                                                   for confidence, frame, text, crop in best]
            meta = (job.get("result") or {}).get("metadata", {})
            outcomes[request_id] = dict(result, video=dict(width=meta.get("width"), height=meta.get("height"), duration_seconds=meta.get("duration_seconds")), sample_crops=crops)
        except Exception as error:  # one bad run must not stop the rest
            outcomes[request_id] = dict(failed=f"{type(error).__name__}: {error}")
        summary = outcomes[request_id]
        print(request_id[:8], {key: summary.get(key) for key in ("identities_read", "identities_proposed", "crops_read", "skipped", "failed") if key in summary}, flush=True)
    return outcomes


@app.local_entrypoint()
def main(out_dir: str, request_ids: str = "", save: bool = False):
    import base64
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outcomes = read_numbers.remote(request_ids, 6, save)
    for request_id, outcome in outcomes.items():
        for identity, crops in outcome.pop("sample_crops", {}).items():
            for index, crop in enumerate(crops):
                (out / f"{request_id[:8]}-{identity}-{index}-read-{crop['text'] or 'none'}.jpg").write_bytes(base64.b64decode(crop.pop("jpeg")))
    (out / "shirt-numbers.json").write_text(json.dumps(outcomes, indent=1), encoding="utf-8")
    print("written", out)
