from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
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
from workers.video.compute import analysis_capabilities as yolo_capabilities
from workers.video.real_analysis import video_metadata


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
WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-analysis")


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
    # A single local worker retries interrupted jobs from their saved input.
    for record in repository.unfinished_jobs():
        job = _job_from_record(record)
        job.update(status="queued", stage="Resuming after restart", progress=0.0)
        with LOCK:
            ACTIVE_JOBS[job["id"]] = job
        path = Path(job["input_path"])
        WORKER.submit(_process_job, job["id"], path, path.name == "demo-match.mp4")


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
        "storage_free_gib": round(shutil.disk_usage(settings.storage_root).free / 1024**3, 2),
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
        "analysis": {"synthetic_upload_fallback": False, "worker_concurrency": 1,
                     "event_detection": False,
                     # Automatic per-frame estimates for the shape view only; not measured distances or speeds.
                     "pitch_calibration": bool(yolo_capabilities(ROOT).get("pitch_calibration")),
                     # A video can be stored for tagging by hand with no detection, so its length is not bound by the analysis.
                     "review_only_upload": True,
                     "automatic_video_insights": True, "event_candidate_method": "unreviewed_temporal_geometry"},
    }


@app.get("/ai/status")
def ai_status() -> dict:
    return provider_status()


@app.post("/videos/upload", response_model=JobResponse)
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), mode: Literal["analysis", "review"] = "analysis") -> dict:
    # mode=review stores the video for tagging by hand: no detector is needed and none is run.
    caps = yolo_capabilities(ROOT)
    if mode == "analysis" and not caps["available"]:
        raise HTTPException(status_code=503, detail=" ".join(caps["blockers"]))
    original_name = file.filename or "clip.mp4"
    suffix = Path(original_name).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported video type. Use one of: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}")

    job_id = uuid4().hex
    target = settings.upload_dir / f"{job_id}{suffix}"
    _save_upload(file, target)
    try:
        metadata = video_metadata(target)
        if mode == "analysis" and os.getenv("ANALYSIS_COMPUTE", "local") == "modal":
            from workers.video.modal_policy import validate_video
            validate_video(target, metadata)
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job = _new_job(job_id, target)
    _register_job(job)
    if mode == "review":
        # Shown under the name it was uploaded with; the stored copy keeps its own name.
        return _public_job(_complete_review_only(job, target, dict(metadata, filename=Path(original_name).name, stored_filename=target.name)))
    WORKER.submit(_process_job, job_id, target)
    return _public_job(job)


def _complete_review_only(job: dict, target: Path, metadata: dict) -> dict:
    from workers.video.review_only import create_run
    try:
        result = create_run(target, settings.output_dir, metadata, job["source_video"])
    except Exception as exc:
        target.unlink(missing_ok=True)
        patch = {"status": "failed", "stage": "Failed", "error": str(exc)}
        with LOCK:
            ACTIVE_JOBS[job["id"]].update(patch)
        repository.update_job(job["id"], **patch)
        raise HTTPException(status_code=500, detail="The video was uploaded but could not be prepared for tagging.") from exc
    repository.upsert_run(result, job_id=job["id"])
    patch = {"status": "complete", "stage": "Ready to tag", "progress": 1.0, "run_id": result["run_id"], "error": None}
    with LOCK:
        ACTIVE_JOBS[job["id"]].update(patch)
        ACTIVE_JOBS[job["id"]]["result"] = result
    repository.update_job(job["id"], **patch)
    return ACTIVE_JOBS[job["id"]]


@app.post("/demo-job", response_model=JobResponse)
def create_demo_job(background_tasks: BackgroundTasks) -> dict:
    demo = _ensure_demo_video()
    job_id = uuid4().hex
    job = _new_job(job_id, demo)
    _register_job(job)
    WORKER.submit(_process_job, job_id, demo, True)
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


@app.post("/analysis-jobs/{job_id}/retry", response_model=JobResponse)
def retry_job(job_id: str) -> dict:
    record = repository.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    if record["status"] != "failed":
        raise HTTPException(status_code=409, detail="Only failed jobs can be retried")
    path = Path(record["input_path"])
    if not path.is_file():
        raise HTTPException(status_code=410, detail="Source video is no longer available")
    job = _new_job(uuid4().hex, path)
    _register_job(job)
    WORKER.submit(_process_job, job["id"], path, path.name == "demo-match.mp4")
    return _public_job(job)


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


@app.get('/library')
def library() -> dict:
    from apps.api.library import compact_job, compact_run
    # Startup and successful processing index manifests. Browsing does not reopen
    # every full tracking manifest or backfill measurements for unrelated clips.
    rows = repository.list_runs()
    by_id = {row['run_id']: row for row in rows}
    with LOCK:
        active = {identity: dict(job) for identity, job in ACTIVE_JOBS.items()}
    entries = []
    for record in repository.list_jobs():
        job = active.get(record['id']) or record
        entries.append(compact_job(job, by_id.get(job.get('run_id'))))
    return dict(runs=[compact_run(row, row['created_at']) for row in rows], jobs=entries)


@app.get("/sample")
def sample_run() -> dict:
    registry = settings.output_dir / "sample.json"
    if not registry.is_file():
        raise HTTPException(404, "Sample has not been prepared yet")
    sample = json.loads(registry.read_text(encoding="utf-8"))
    manifest = _read_manifest(sample["run_id"])
    if not manifest:
        raise HTTPException(404, "Sample run not found")
    return dict(id=sample["id"], status="complete", stage="Complete", progress=1,
                created_at=sample["created_at"], result=manifest, error=None)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    manifest = _read_manifest(run_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Run not found")
    return manifest


REVIEW_LOCK=threading.RLock()


@app.get('/runs/{run_id}/sequence-context')
def get_sequence_context(run_id:str):
    from workers.video.sequence_context import load_context
    with REVIEW_LOCK:
        path,manifest=_review_location(run_id)
        return load_context(path.with_name('sequence-context.json'),manifest)


@app.post('/runs/{run_id}/sequence-context')
def put_sequence_context(run_id:str,payload:dict):
    from workers.video.sequence_context import save_context
    from workers.video.event_review import ReviewConflict
    with REVIEW_LOCK:
        path,manifest=_review_location(run_id)
        try:return save_context(path.with_name('sequence-context.json'),manifest,payload)
        except ReviewConflict as exc:raise HTTPException(409,str(exc)) from exc
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@app.get('/runs/{run_id}/players')
def get_player_labels(run_id:str):
    from workers.video.player_labels import load_labels
    with REVIEW_LOCK:
        path,manifest=_review_location(run_id)
        try:
            from workers.video.event_review import load_review
            from workers.video.video_roster import refresh_labels
            return refresh_labels(load_labels(path.with_name('player-labels.json'),manifest),load_review(path,manifest)['lineups'])
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@app.get('/runs/{run_id}/roster')
def get_video_roster(run_id:str):
    from workers.video.event_review import load_review
    from workers.video.player_labels import load_labels
    from workers.video.video_roster import video_roster
    with REVIEW_LOCK:
        path,manifest=_review_location(run_id)
        try:return video_roster(load_review(path,manifest),load_labels(path.with_name('player-labels.json'),manifest))
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@app.get('/roster')
def get_roster():
    from workers.video.player_labels import roster
    return dict(players=roster(settings.output_dir.glob('*/player-labels.json')))


@app.get('/runs/{run_id}/players/suggestions')
def get_merge_suggestions(run_id:str):
    from workers.video.merge_suggestions import suggest_for_run
    path,manifest=_review_location(run_id)
    out=suggest_for_run(path.parent,manifest)
    if out.get('shirt_numbers'):
        # Reads are checked against the line-up saved under Review each time, so a corrected line-up shows at once.
        from workers.video.event_review import load_review
        from workers.video.player_labels import load_labels
        from workers.video.video_roster import lineup_number_matches
        with REVIEW_LOCK:
            out['shirt_numbers']=lineup_number_matches(out['shirt_numbers'],load_review(path,manifest),load_labels(path.with_name('player-labels.json'),manifest),manifest)
    return out


@app.post('/runs/{run_id}/players')
def put_player_labels(run_id:str,payload:dict):
    from workers.video.player_labels import save_labels
    from workers.video.event_review import ReviewConflict
    with REVIEW_LOCK:
        path,manifest=_review_location(run_id)
        try:
            from workers.video.event_review import load_review
            return save_labels(path.with_name('player-labels.json'),manifest,payload,path.with_name('tracking.json'),load_review(path,manifest)['lineups'])
        except ReviewConflict as exc:raise HTTPException(409,str(exc)) from exc
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc


def _review_location(run_id):
    manifest=_read_manifest(run_id)
    if not manifest:raise HTTPException(404,'Run not found')
    if manifest.get('engine') not in {'yolo_bytetrack','review_only'}:raise HTTPException(422,'Only real video runs can be reviewed')
    return settings.output_dir/run_id/'review.json',manifest


@app.get('/runs/{run_id}/review')
def get_review(run_id:str):
    from workers.video.event_review import load_review
    with REVIEW_LOCK:
        path,manifest=_review_location(run_id)
        return load_review(path,manifest)


@app.get('/runs/{run_id}/review/approved-events')
def get_approved_events(run_id:str):
    from workers.video.event_review import load_review,approved_export
    with REVIEW_LOCK:
        path,manifest=_review_location(run_id)
        return approved_export(load_review(path,manifest),manifest)


@app.post('/runs/{run_id}/review')
def put_review(run_id:str,payload:dict):
    from workers.video.event_review import save_review,ReviewConflict
    with REVIEW_LOCK:
        path,manifest=_review_location(run_id)
        try:return save_review(path,manifest,payload)
        except ReviewConflict as exc:raise HTTPException(409,str(exc)) from exc
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@app.post("/runs/{run_id}/ai-summary")
def ai_summary(run_id: str) -> dict:
    manifest = _read_manifest(run_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Run not found")
    return enrich_report(manifest.get("report", {}))


def _process_job(job_id: str, path: Path, demo: bool = False) -> None:
    def progress(stage: str, pct: float) -> None:
        patch = {"stage": stage, "progress": round(pct, 3), "status": "running"}
        with LOCK:
            ACTIVE_JOBS[job_id].update(patch)
        repository.update_job(job_id, **patch)

    try:
        progress("Starting", 0.02)
        result = run_analysis(path, settings.output_dir, progress, demo=demo)

        repository.upsert_run(result, job_id=job_id)
        patch = {"status": "complete", "stage": "Complete", "progress": 1.0, "run_id": result.get("run_id"), "error": None}
        with LOCK:
            ACTIVE_JOBS[job_id].update(patch)
            ACTIVE_JOBS[job_id]["result"] = result
        repository.update_job(job_id, **patch)
    except Exception as exc:  # pragma: no cover - keeps the API loop alive for user-facing failures
        logging.getLogger(__name__).exception("Analysis job %s failed", job_id)
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
    if not run_id or not run_id.isalnum():
        return None
    record = repository.get_run_record(run_id)
    manifest_path = Path(record["manifest_path"]) if record else settings.output_dir / run_id / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("engine") == "yolo_bytetrack" and not manifest.get("metrics", {}).get("video_measurements"):
            from workers.video.measurements import from_tracking
            manifest["metrics"]["video_measurements"] = from_tracking(
                json.loads((manifest_path.parent / "tracking.json").read_text(encoding="utf-8")),
                manifest["metadata"], manifest["detection_summary"])
            (manifest_path.parent / "metrics.json").write_text(json.dumps(manifest["metrics"]), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        from workers.video.insights import ensure_insights
        ensure_insights(manifest, manifest_path.parent)
        # The untouched upload, for clean frames behind diagrams. Offered only when that exact file is still stored.
        name = Path(str(manifest.get("metadata", {}).get("filename") or "")).name
        if manifest.get("engine") != "review_only" and name and (settings.upload_dir / name).is_file():
            manifest.setdefault("files", {})["source_video"] = f"/media/uploads/{name}"
        return manifest
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
