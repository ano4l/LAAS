from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, database_path: Path, output_dir: Path) -> None:
        self.database_path = database_path
        self.output_dir = output_dir

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_video TEXT,
                    input_path TEXT,
                    run_id TEXT,
                    error TEXT
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    created_at TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    annotated_video TEXT,
                    tracking_json TEXT,
                    metrics_json TEXT,
                    report_json TEXT,
                    metadata_json TEXT NOT NULL,
                    metrics_summary_json TEXT NOT NULL,
                    report_summary_json TEXT NOT NULL
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON analysis_jobs(created_at DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON analysis_runs(created_at DESC)")

    def create_job(self, job: dict[str, Any]) -> None:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO analysis_jobs (
                    id, status, stage, progress, created_at, updated_at, source_video, input_path, run_id, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["id"],
                    job["status"],
                    job["stage"],
                    job["progress"],
                    job["created_at"],
                    now,
                    job.get("source_video"),
                    job.get("input_path"),
                    job.get("run_id"),
                    job.get("error"),
                ),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [job_id]
        with self.connect() as db:
            db.execute(f"UPDATE analysis_jobs SET {assignments} WHERE id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def unfinished_jobs(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM analysis_jobs WHERE status IN ('queued', 'running') ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM analysis_jobs ORDER BY created_at DESC LIMIT 100").fetchall()
        return [dict(row) for row in rows]

    def upsert_run(self, manifest: dict[str, Any], job_id: str | None = None, manifest_path: Path | None = None) -> None:
        run_id = manifest["run_id"]
        files = manifest.get("files", {})
        report = manifest.get("report", {})
        metrics = manifest.get("metrics", {})
        manifest_path = manifest_path or self.output_dir / run_id / "manifest.json"
        created_at = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc).isoformat() if manifest_path.exists() else utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO analysis_runs (
                    run_id, job_id, created_at, manifest_path, annotated_video, tracking_json, metrics_json,
                    report_json, metadata_json, metrics_summary_json, report_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    job_id = COALESCE(excluded.job_id, analysis_runs.job_id),
                    created_at = excluded.created_at,
                    manifest_path = excluded.manifest_path,
                    annotated_video = excluded.annotated_video,
                    tracking_json = excluded.tracking_json,
                    metrics_json = excluded.metrics_json,
                    report_json = excluded.report_json,
                    metadata_json = excluded.metadata_json,
                    metrics_summary_json = excluded.metrics_summary_json,
                    report_summary_json = excluded.report_summary_json
                """,
                (
                    run_id,
                    job_id,
                    created_at,
                    str(manifest_path),
                    files.get("annotated_video"),
                    files.get("tracking_json"),
                    files.get("metrics_json"),
                    files.get("report_json"),
                    json.dumps(manifest.get("metadata", {})),
                    json.dumps(_summarize_metrics(metrics)),
                    json.dumps(_summarize_report(report)),
                ),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM analysis_runs ORDER BY created_at DESC LIMIT 100").fetchall()
        return [_run_row_to_public(dict(row)) for row in rows]

    def get_run_record(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM analysis_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def index_existing_manifests(self) -> None:
        for manifest_path in self.output_dir.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(manifest, dict) and manifest.get("run_id"):
                self.upsert_run(manifest, manifest_path=manifest_path)


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "possession": metrics.get("possession", {}),
        "teams": metrics.get("teams", {}),
        "players": metrics.get("players", [])[:12],
        "events": metrics.get("events", [])[:10],
        "phase_count": metrics.get("phase_count", 0),
        "data_quality": metrics.get("data_quality", {}),
    }


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": report.get("title"),
        "summary": report.get("summary"),
        "recommendations": report.get("recommendations", [])[:5],
        "evidence": report.get("evidence", {}),
    }


def _run_row_to_public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "metadata": json.loads(row["metadata_json"]),
        "metrics": json.loads(row["metrics_summary_json"]),
        "phases": [],
        "report": json.loads(row["report_summary_json"]),
        "files": {
            "annotated_video": row["annotated_video"],
            "tracking_json": row["tracking_json"],
            "metrics_json": row["metrics_json"],
            "report_json": row["report_json"],
        },
        "created_at": row["created_at"],
    }
