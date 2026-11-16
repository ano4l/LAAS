r"""Smoke-test an installed or freshly built bundle with its own Python.

Usage (from the bundle folder):  python\python.exe -B smoke.py [--compute local|modal] [--clip PATH]

Starts the server on a spare port with throwaway storage, uploads a 1.5 s clip and checks the
run's files. --compute local never contacts Modal. --compute modal uses one Modal run if the
token works and otherwise reports the CPU fallback. Prints one JSON line; exits non-zero on failure.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--compute", choices=["local", "modal"], default="local")
parser.add_argument("--clip", type=Path, default=ROOT / "smoke-clip.mp4")
parser.add_argument("--timeout", type=int, default=1800, help="Seconds to wait for the analysis")
parser.add_argument("--keep", action="store_true", help="Keep the temporary storage folder")
args = parser.parse_args()

storage = Path(tempfile.mkdtemp(prefix="laas-smoke-"))
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
process = subprocess.Popen([sys.executable, "-B", str(ROOT / "Start-LAAS.py"), "--storage", str(storage),
                            "--port", str(port), "--compute", args.compute], cwd=ROOT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                           env={**os.environ, "PYTHONNOUSERSITE": "1"})
base = f"http://127.0.0.1:{port}"


def get(path):
    with urlopen(base + path, timeout=15) as response:
        return json.load(response)


ok = False
try:
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early. Check {storage / 'logs' / 'server.log'}")
        try:
            assert get("/ready")["ok"]
            break
        except URLError:
            time.sleep(1)
    else:
        raise TimeoutError("Server never became ready")
    capabilities = get("/capabilities")
    yolo = capabilities["engines"]["yolo"]
    assert yolo["available"], yolo
    with urlopen(base, timeout=15) as response:
        assert b"<div id=\"root\">" in response.read()
    boundary = uuid4().hex
    header = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.mp4"\r\nContent-Type: video/mp4\r\n\r\n'
    data = header.encode() + args.clip.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    started = time.monotonic()
    with urlopen(Request(base + "/videos/upload", data=data, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}), timeout=60) as response:
        job_id = json.load(response)["id"]
    while True:
        job = get(f"/analysis-jobs/{job_id}")
        if job["status"] in ("failed", "complete"):
            break
        if time.monotonic() - started > args.timeout:
            raise TimeoutError(f"Analysis still {job['status']} ({job.get('stage')}) after {args.timeout} s")
        time.sleep(2)
    assert job["status"] == "complete", job
    summary = job["result"]["detection_summary"]
    run_dir = next((storage / "data" / "outputs").rglob("manifest.json")).parent
    present = sorted(path.name for path in run_dir.iterdir())
    for name in ("annotated.mp4", "tracking.json", "manifest.json", "track-links.json"):
        assert name in present, f"{name} missing from the run"
    print(json.dumps(dict(status="complete", requested_compute=args.compute, ran_on=summary.get("compute_provider", "local"),
                          fallback=summary.get("compute_fallback"), gpu=summary.get("gpu"), device=summary.get("device"),
                          profile=summary.get("analysis_profile"), cpu_threads=summary.get("cpu_threads"),
                          wall_seconds=round(time.monotonic() - started, 1), processing_seconds=summary.get("processing_seconds"),
                          pitch_calibration="pitch-calibration.json" in present, files=present)))
    ok = True
finally:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
    if ok and not args.keep:
        shutil.rmtree(storage, ignore_errors=True)
    else:
        print(f"Storage kept at {storage}", file=sys.stderr)
