"""Private prototype API for a separately hosted browser frontend."""
import copy
import hashlib
import hmac
import json
from pathlib import Path
import re
import time
import threading
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from workers.video.modal_policy import ARTIFACTS, MAX_BYTES, validate_video


def create_gateway(volume, analyze, access_key, root=Path("/jobs"), call_lookup=None):
    if len(access_key) < 32:
        raise ValueError("A strong workspace access key is required")
    app = FastAPI(title="LAAS private video API")
    # No cookie credentials; every data request requires the workspace bearer key.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                       allow_headers=["Authorization", "Content-Type"], expose_headers=["Content-Range"])
    jobs = root / "web-jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    review_lock=threading.RLock()

    def authorized(request: Request):
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {access_key}".encode()):
            raise HTTPException(401, "Enter a valid workspace access key")

    def valid_id(value):
        if not re.fullmatch(r"[0-9a-f]{32}", value):
            raise HTTPException(404, "Not found")

    def save(job):
        (jobs / f"{job['id']}.json").write_text(json.dumps(job), encoding="utf-8")
        volume.commit()

    def load(job_id):
        valid_id(job_id)
        volume.reload()
        path = jobs / f"{job_id}.json"
        if not path.is_file():
            raise HTTPException(404, "Job not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def update(job, backfill=True):
        if job["status"] in {"queued", "running"} and job.get("call_id"):
            try:
                if call_lookup is None:
                    import modal
                    call = modal.FunctionCall.from_id(job["call_id"])
                else:
                    call = call_lookup(job["call_id"])
                result = call.get(timeout=0)
                volume.reload()
                run_id = result["run_id"]
                valid_id(run_id)
                manifest = root / "requests" / job["id"] / "outputs" / run_id / "manifest.json"
                job.update(status="complete", stage="Complete", progress=1, run_id=run_id,
                           result=json.loads(manifest.read_text(encoding="utf-8")))
                save(job)
            except TimeoutError:
                job.update(status="running", stage="Waiting for or processing on Modal GPU", progress=0.2)
            except Exception:
                # Tracebacks belong in the provider logs, not the public API response.
                import logging
                logging.getLogger(__name__).exception("Cloud analysis failed for %s", job["id"])
                job.update(status="failed", stage="Failed", error="Cloud analysis failed. Check the Modal logs before retrying.")
                save(job)
        if job.get("result") and job["result"].get("engine") == "yolo_bytetrack":
            result = job["result"]
            if backfill and not result.get("metrics", {}).get("video_measurements"):
                from workers.video.measurements import from_tracking
                directory = root / "requests" / job["id"] / "outputs" / job["run_id"]
                result.setdefault("metrics", {})["video_measurements"] = from_tracking(
                    json.loads((directory / "tracking.json").read_text(encoding="utf-8")),
                    result["metadata"], result["detection_summary"])
                (directory / "metrics.json").write_text(json.dumps(result["metrics"]), encoding="utf-8")
                (directory / "manifest.json").write_text(json.dumps(result), encoding="utf-8")
                save(job)
            if backfill:
                from workers.video.insights import ensure_insights
                directory = root / "requests" / job["id"] / "outputs" / job["run_id"]
                if ensure_insights(result, directory):
                    save(job)
            if job.get("filename"):
                result["metadata"]["filename"] = job["filename"]
        return job

    def signature(path, expires):
        return hmac.new(access_key.encode(), f"{path}:{expires}".encode(), hashlib.sha256).hexdigest()

    def public(job):
        out = {key: copy.deepcopy(value) for key, value in job.items() if key not in {"call_id", "run_id"}}
        if out.get("result"):
            result = out["result"]
            expires = int(time.time()) + 3600
            result["files"] = {}
            for key, name in dict(annotated_video="annotated.mp4", tracking_json="tracking.json",
                                 metrics_json="metrics.json", report_json="report.json", manifest_json="manifest.json").items():
                path = f"/media/{job['id']}/{job['run_id']}/{name}"
                result["files"][key] = f"{path}?expires={expires}&signature={signature(path, expires)}"
            # The untouched upload, for clean frames behind diagrams. Offered only while the file is still stored.
            if (root / "requests" / job["id"] / "source.mp4").is_file():
                path = f"/media/{job['id']}/{job['run_id']}/source.mp4"
                result["files"]["source_video"] = f"{path}?expires={expires}&signature={signature(path, expires)}"
        return out

    @app.get("/health")
    def health():
        return dict(ok=True, compute="modal")

    @app.get("/capabilities", dependencies=[Depends(authorized)])
    def capabilities():
        return dict(engines=dict(fallback=dict(available=False, description="Hosted prototype uses real uploads"),
                                yolo=dict(available=True, blockers=[], compute_provider="modal")),
                    limits=dict(max_upload_mb=250), analysis=dict(event_detection=False, pitch_calibration=False,
                        automatic_video_insights=True, event_candidate_method="unreviewed_temporal_geometry"))

    @app.post("/videos/upload", dependencies=[Depends(authorized)])
    def upload(file: UploadFile = File(...)):
        from workers.video.real_analysis import video_metadata
        volume.reload()
        # A shared private workspace admits only one outstanding web job.
        for path in jobs.glob("*.json"):
            existing = update(json.loads(path.read_text(encoding="utf-8")))
            if existing["status"] in {"queued", "running"}:
                raise HTTPException(409, "A video is already queued or processing. Wait for it to finish.")
        if Path(file.filename or "").suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
            raise HTTPException(415, "Unsupported video format")
        job_id = uuid4().hex
        directory = root / "requests" / job_id
        directory.mkdir(parents=True)
        source = directory / "source.mp4"
        try:
            size = 0
            with source.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "Video exceeds 250 MiB")
                    target.write(chunk)
            validate_video(source, video_metadata(source))
        except (ValueError, HTTPException) as exc:
            source.unlink(missing_ok=True)
            directory.rmdir()
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(422, str(exc)) from exc
        job = dict(id=job_id, status="queued", stage="Queued for Modal GPU", progress=0.05,
                   created_at=datetime.now(timezone.utc).isoformat(), result=None, error=None,
                   filename=Path(file.filename).name)
        save(job)  # Persist the input before a separate GPU container reads it.
        try:
            call = analyze.spawn(job_id)
            job["call_id"] = call.object_id
        except Exception:
            job.update(status="failed", stage="Failed", error="Could not submit GPU job")
            save(job)
            raise HTTPException(503, "Could not submit GPU job")
        save(job)
        return public(job)

    @app.get("/analysis-jobs", dependencies=[Depends(authorized)])
    def list_jobs():
        volume.reload()
        return sorted([public(update(json.loads(p.read_text(encoding="utf-8")))) for p in jobs.glob("*.json")],
                      key=lambda row: row["created_at"], reverse=True)

    @app.get("/analysis-jobs/{job_id}", dependencies=[Depends(authorized)])
    def get_job(job_id: str):
        return public(update(load(job_id)))

    @app.get("/sample", dependencies=[Depends(authorized)])
    def sample():
        volume.reload()
        for path in jobs.glob("*.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("sample") and job["status"] == "complete":
                return public(update(job))
        raise HTTPException(404, "Sample has not been prepared yet")

    @app.get("/runs", dependencies=[Depends(authorized)])
    def runs():
        return [dict(**job["result"], created_at=job["created_at"]) for job in list_jobs() if job.get("result")]

    @app.get('/library', dependencies=[Depends(authorized)])
    def library():
        from apps.api.library import compact_job, compact_run
        volume.reload()
        # Status resolution stays on the selected job's polling endpoint. A
        # library visit must not wait on provider lookups for unrelated jobs.
        entries = [json.loads(path.read_text(encoding='utf-8')) for path in jobs.glob('*.json')]
        entries.sort(key=lambda row: row['created_at'], reverse=True)
        return dict(jobs=[compact_job(job) for job in entries], runs=[
            compact_run(job['result'], job['created_at'], job.get('filename'))
            for job in entries if job.get('result') and job['status']=='complete'])

    @app.get("/runs/{run_id}", dependencies=[Depends(authorized)])
    def run(run_id: str):
        valid_id(run_id)
        volume.reload()
        for path in jobs.glob('*.json'):
            job=json.loads(path.read_text(encoding='utf-8'))
            if job.get('run_id')==run_id and job.get('result'):
                return dict(**public(update(job))['result'], created_at=job['created_at'])
        raise HTTPException(404, "Run not found")

    def review_location(run_id):
        valid_id(run_id)
        volume.reload()
        for path in jobs.glob('*.json'):
            job=json.loads(path.read_text(encoding='utf-8'))
            if job.get('run_id')==run_id and job.get('result') and job['status']=='complete':
                directory=root/'requests'/job['id']/'outputs'/run_id
                return directory/'review.json',job['result']
        raise HTTPException(404,'Completed run not found')

    @app.get('/runs/{run_id}/review',dependencies=[Depends(authorized)])
    def get_review(run_id:str):
        from workers.video.event_review import load_review
        with review_lock:
            path,manifest=review_location(run_id)
            return load_review(path,manifest)

    @app.get('/runs/{run_id}/review/approved-events',dependencies=[Depends(authorized)])
    def get_approved_events(run_id:str):
        from workers.video.event_review import load_review,approved_export
        with review_lock:
            path,manifest=review_location(run_id)
            return approved_export(load_review(path,manifest),manifest)

    @app.get('/runs/{run_id}/sequence-context',dependencies=[Depends(authorized)])
    def get_sequence_context(run_id:str):
        from workers.video.sequence_context import load_context
        with review_lock:
            path,manifest=review_location(run_id)
            return load_context(path.with_name('sequence-context.json'),manifest)

    @app.post('/runs/{run_id}/sequence-context',dependencies=[Depends(authorized)])
    def put_sequence_context(run_id:str,payload:dict):
        from workers.video.sequence_context import save_context
        from workers.video.event_review import ReviewConflict
        with review_lock:
            path,manifest=review_location(run_id)
            try:state=save_context(path.with_name('sequence-context.json'),manifest,payload)
            except ReviewConflict as exc:raise HTTPException(409,str(exc)) from exc
            except ValueError as exc:raise HTTPException(422,str(exc)) from exc
            volume.commit()
            return state

    @app.get('/runs/{run_id}/players',dependencies=[Depends(authorized)])
    def get_player_labels(run_id:str):
        from workers.video.player_labels import load_labels
        with review_lock:
            path,manifest=review_location(run_id)
            try:return load_labels(path.with_name('player-labels.json'),manifest)
            except ValueError as exc:raise HTTPException(422,str(exc)) from exc

    @app.get('/roster',dependencies=[Depends(authorized)])
    def get_roster():
        from workers.video.player_labels import roster
        volume.reload()
        return dict(players=roster(root.glob('requests/*/outputs/*/player-labels.json')))

    @app.get('/runs/{run_id}/players/suggestions',dependencies=[Depends(authorized)])
    def get_merge_suggestions(run_id:str):
        from workers.video.merge_suggestions import suggest_for_run
        path,manifest=review_location(run_id)
        return suggest_for_run(path.parent,manifest)

    @app.post('/runs/{run_id}/players',dependencies=[Depends(authorized)])
    def put_player_labels(run_id:str,payload:dict):
        from workers.video.player_labels import save_labels
        from workers.video.event_review import ReviewConflict
        with review_lock:
            path,manifest=review_location(run_id)
            try:state=save_labels(path.with_name('player-labels.json'),manifest,payload,path.with_name('tracking.json'))
            except ReviewConflict as exc:raise HTTPException(409,str(exc)) from exc
            except ValueError as exc:raise HTTPException(422,str(exc)) from exc
            volume.commit()
            return state

    @app.post('/runs/{run_id}/review',dependencies=[Depends(authorized)])
    def put_review(run_id:str,payload:dict):
        from workers.video.event_review import save_review,ReviewConflict
        with review_lock:
            path,manifest=review_location(run_id)
            try:state=save_review(path,manifest,payload)
            except ReviewConflict as exc:raise HTTPException(409,str(exc)) from exc
            except ValueError as exc:raise HTTPException(422,str(exc)) from exc
            volume.commit()
            return state

    @app.get("/media/{job_id}/{run_id}/{filename}")
    def media(job_id: str, run_id: str, filename: str, expires: int, signature: str):
        valid_id(job_id)
        valid_id(run_id)
        path = f"/media/{job_id}/{run_id}/{filename}"
        expected = hmac.new(access_key.encode(), f"{path}:{expires}".encode(), hashlib.sha256).hexdigest()
        if expires < time.time() or expires > time.time() + 3660 or not hmac.compare_digest(signature.encode(), expected.encode()):
            raise HTTPException(403, "Media link expired; refresh the analysis")
        if filename not in ARTIFACTS and filename != "source.mp4":
            raise HTTPException(404, "File not found")
        volume.reload()
        file_path = root / "requests" / job_id / "outputs" / run_id / filename
        if filename == "source.mp4":
            if not (file_path.parent / "manifest.json").is_file():
                raise HTTPException(404, "File not found")
            file_path = root / "requests" / job_id / "source.mp4"
        if not file_path.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(file_path)

    return app
