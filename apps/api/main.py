from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apps.api.ai import enrich_report, provider_status
from apps.api.config import ROOT, settings
from apps.api.repository import Repository, utc_now
from workers.video.pipeline import run_analysis
from workers.video.yolo_adapter import yolo_capabilities


JobStatus = Literal["queued", "running", "complete", "failed"]
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

repository = Repository(settings.database_path, settings.output_dir)
app = FastAPI(title="Football AI Analytics API", version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
app.mount("/media/outputs", StaticFiles(directory=settings.output_dir), name="outputs")

ACTIVE_JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    stage: str
    progress: float
    created_at: str
    source_video: str | None = None
    result: dict | None = None
    error: str | None = None


@app.on_event("startup")
def startup() -> None:
    repository.init()
    repository.index_existing_manifests()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "football-ai-analytics", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
def ready() -> dict:
    repository.init()
    return {
        "ok": True,
        "database": str(settings.database_path),
        "storage": str(settings.storage_root),
        "engine": settings.football_analytics_engine,
    }


@app.get("/version")
def version() -> dict:
    return {"version": settings.app_version, "environment": settings.app_env}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "engines": {
            "fallback": {"available": True, "description": "Deterministic tactical demo engine"},
            "yolo": yolo_capabilities(ROOT),
        },
        "limits": {"max_upload_mb": settings.max_upload_mb, "allowed_video_suffixes": sorted(ALLOWED_VIDEO_SUFFIXES)},
        "ai": provider_status(),
    }


@app.get("/ai/status")
def ai_status() -> dict:
    return provider_status()


@app.post("/videos/upload", response_model=JobResponse)
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    original_name = file.filename or "clip.mp4"
    suffix = Path(original_name).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported video type. Use one of: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}")

    job_id = uuid4().hex
    target = settings.upload_dir / f"{job_id}{suffix}"
    _save_upload(file, target)

    job = _new_job(job_id, target)
    _register_job(job)
    background_tasks.add_task(_process_job, job_id, target)
    return _public_job(job)


@app.post("/demo-job", response_model=JobResponse)
def create_demo_job(background_tasks: BackgroundTasks) -> dict:
    demo = _ensure_demo_video()
    job_id = uuid4().hex
    job = _new_job(job_id, demo)
    _register_job(job)
    background_tasks.add_task(_process_job, job_id, demo)
    return _public_job(job)


@app.get("/analysis-jobs", response_model=list[JobResponse])
def list_jobs() -> list[dict]:
    active_ids = set()
    with LOCK:
        active = [_public_job(job) for job in ACTIVE_JOBS.values()]
        active_ids = {job["id"] for job in active}
    persisted = [_public_job(_job_from_record(row)) for row in repository.list_jobs() if row["id"] not in active_ids]
    return sorted(active + persisted, key=lambda row: row["created_at"], reverse=True)


@app.get("/analysis-jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> dict:
    with LOCK:
        job = ACTIVE_JOBS.get(job_id)
    if job:
        return _public_job(job)

    record = repository.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return _public_job(_job_from_record(record, include_result=True))


@app.get("/runs")
def list_runs() -> list[dict]:
    repository.index_existing_manifests()
    runs = []
    for row in repository.list_runs():
        manifest = _read_manifest(row["run_id"])
        if manifest:
            row["phases"] = manifest.get("phases", [])[:8]
        runs.append(row)
    return runs


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    manifest = _read_manifest(run_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Run not found")
    return manifest


@app.post("/runs/{run_id}/ai-summary")
def ai_summary(run_id: str) -> dict:
    manifest = _read_manifest(run_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Run not found")
    return enrich_report(manifest.get("report", {}))


def _process_job(job_id: str, path: Path) -> None:
    def progress(stage: str, pct: float) -> None:
        patch = {"stage": stage, "progress": round(pct, 3), "status": "running"}
        with LOCK:
            ACTIVE_JOBS[job_id].update(patch)
        repository.update_job(job_id, **patch)

    try:
        progress("Starting", 0.02)
        previous_engine = os.environ.get("FOOTBALL_ANALYTICS_ENGINE")
        os.environ["FOOTBALL_ANALYTICS_ENGINE"] = settings.football_analytics_engine
        try:
            result = run_analysis(path, settings.output_dir, progress)
        finally:
            if previous_engine is None:
                os.environ.pop("FOOTBALL_ANALYTICS_ENGINE", None)
            else:
                os.environ["FOOTBALL_ANALYTICS_ENGINE"] = previous_engine

        repository.upsert_run(result, job_id=job_id)
        patch = {"status": "complete", "stage": "Complete", "progress": 1.0, "run_id": result.get("run_id"), "error": None}
        with LOCK:
            ACTIVE_JOBS[job_id].update(patch)
            ACTIVE_JOBS[job_id]["result"] = result
        repository.update_job(job_id, **patch)
    except Exception as exc:  # pragma: no cover - keeps the API loop alive for user-facing failures
        patch = {"status": "failed", "stage": "Failed", "error": str(exc)}
        with LOCK:
            ACTIVE_JOBS[job_id].update(patch)
        repository.update_job(job_id, **patch)


def _new_job(job_id: str, path: Path) -> dict:
    return {
        "id": job_id,
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "created_at": utc_now(),
        "source_video": f"/media/uploads/{path.name}",
        "result": None,
        "error": None,
        "input_path": str(path),
        "run_id": None,
    }


def _register_job(job: dict) -> None:
    with LOCK:
        ACTIVE_JOBS[job["id"]] = job
    repository.create_job(job)


def _public_job(job: dict) -> dict:
    return {key: value for key, value in job.items() if key not in {"input_path", "run_id"}}


def _job_from_record(record: dict, include_result: bool = False) -> dict:
    result = _read_manifest(record["run_id"]) if include_result and record.get("run_id") else None
    return {
        "id": record["id"],
        "status": record["status"],
        "stage": record["stage"],
        "progress": record["progress"],
        "created_at": record["created_at"],
        "source_video": record.get("source_video"),
        "result": result,
        "error": record.get("error"),
        "input_path": record.get("input_path"),
        "run_id": record.get("run_id"),
    }


def _read_manifest(run_id: str | None) -> dict | None:
    if not run_id:
        return None
    record = repository.get_run_record(run_id)
    manifest_path = Path(record["manifest_path"]) if record else settings.output_dir / run_id / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _save_upload(file: UploadFile, target: Path) -> None:
    total = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail=f"Video exceeds {settings.max_upload_mb}MB upload limit")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _ensure_demo_video() -> Path:
    path = settings.upload_dir / "demo-match.mp4"
    if path.exists():
        return path

    import cv2
    import numpy as np

    width, height, fps = 1280, 720, 25
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_idx in range(350):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (23, 54, 42)
        cv2.rectangle(frame, (70, 70), (1210, 650), (120, 170, 135), 2)
        cv2.line(frame, (640, 70), (640, 650), (120, 170, 135), 2)
        cv2.circle(frame, (640, 360), 86, (120, 170, 135), 2)
        t = frame_idx / fps
        for idx in range(11):
            x = int(240 + idx % 4 * 90 + 65 * np.sin(t + idx))
            y = int(150 + idx % 3 * 155 + 22 * np.cos(t * 1.3 + idx))
            cv2.circle(frame, (x, y), 11, (72, 128, 255), -1)
        for idx in range(11):
            x = int(1020 - idx % 4 * 90 + 65 * np.sin(t + idx + 2))
            y = int(150 + idx % 3 * 155 + 22 * np.cos(t * 1.4 + idx))
            cv2.circle(frame, (x, y), 11, (240, 80, 120), -1)
        ball_x = int(640 + 360 * np.sin(t / 2.8))
        ball_y = int(360 + 160 * np.sin(t / 1.7))
        cv2.circle(frame, (ball_x, ball_y), 7, (30, 240, 255), -1)
        cv2.putText(frame, "Demo match footage", (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
        writer.write(frame)
    writer.release()
    return path


web_dist = ROOT / "apps" / "web" / "dist"
if web_dist.exists():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
