from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - import depends on local native packages
    cv2 = None


Progress = Callable[[str, float], None]


@dataclass
class TrackPoint:
    frame_idx: int
    timestamp_ms: int
    track_id: str
    object_type: str
    team: str | None
    x: float
    y: float
    bbox: list[float]
    confidence: float


def run_analysis(input_video: Path, output_root: Path, progress: Progress | None = None, *, demo: bool = False) -> dict:
    """Real uploads use the detector; synthetic analytics require explicit demo=True."""
    if not demo:
        import os
        provider = os.getenv("ANALYSIS_COMPUTE", "local")
        if provider == "modal":
            from workers.video.modal_client import run_modal_analysis
            from workers.video.compute import fallback_enabled, run_local_fallback
            try:
                return run_modal_analysis(input_video, output_root, progress)
            except Exception as exc:
                # A cloud failure stays a failure unless the operator opted in with ANALYSIS_FALLBACK=local.
                if not fallback_enabled():
                    raise
                return run_local_fallback(input_video, output_root, progress, exc)
        if provider != "local":
            raise ValueError("ANALYSIS_COMPUTE must be local or modal")
        from workers.video.real_analysis import run_real_analysis
        return run_real_analysis(input_video, output_root, progress)
    run_id = uuid4().hex[:10]
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _tick(progress, "Reading video", 0.08)
    metadata = _video_metadata(input_video)

    analysis = None  # Synthetic data is available only through the explicit demo endpoint.
    if analysis:
        metadata = analysis["metadata"]
        tracking = analysis["tracking"]
        engine_quality = analysis["quality"]
        engine = analysis["engine"]
    else:
        frame_count = metadata["frame_count"] or 900
        fps = metadata["fps"] or 25

        _tick(progress, "Tracking players and ball", 0.24)
        tracking = _synthetic_tracking(frame_count, fps)
        engine_quality = {
            "tracking_confidence": 0.82,
            "homography_quality": 0.7,
            "ball_missing_ratio": 0.04,
            "mode": "deterministic_fallback",
        }
        engine = "deterministic_fallback"

    _tick(progress, "Projecting pitch coordinates", 0.42)
    phases = _classify_phases(tracking)

    _tick(progress, "Calculating team metrics", 0.62)
    metrics = _build_metrics(tracking, phases, metadata["fps"] or 25, engine_quality)

    _tick(progress, "Rendering annotated output", 0.78)
    annotated_path = out_dir / "annotated.mp4"
    _render_annotated_video(input_video, annotated_path, tracking, metadata)

    _tick(progress, "Writing tactical report", 0.92)
    report = _build_report(metrics, phases)

    preview_tracking = _preview_tracking(tracking)
    artifacts = {
        "run_id": run_id,
        "metadata": metadata,
        "engine": engine,
        "tracking": preview_tracking,
        "metrics": metrics,
        "phases": phases,
        "report": report,
        "files": {
            "annotated_video": f"/media/outputs/{run_id}/annotated.mp4",
            "tracking_json": f"/media/outputs/{run_id}/tracking.json",
            "metrics_json": f"/media/outputs/{run_id}/metrics.json",
            "report_json": f"/media/outputs/{run_id}/report.json",
        },
    }

    _write_json(out_dir / "tracking.json", tracking)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "phases.json", phases)
    _write_json(out_dir / "report.json", report)
    _write_json(out_dir / "manifest.json", artifacts)
    _tick(progress, "Complete", 1.0)
    return artifacts


def _video_metadata(path: Path) -> dict:
    if cv2 is None:
        return {"filename": path.name, "fps": 25, "frame_count": 900, "duration_seconds": 36, "width": 1280, "height": 720}

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"filename": path.name, "fps": 25, "frame_count": 900, "duration_seconds": 36, "width": 1280, "height": 720}

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 900)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    capture.release()
    return {
        "filename": path.name,
        "fps": round(float(fps), 2),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 2) if fps else 0,
        "width": width,
        "height": height,
    }


def _synthetic_tracking(frame_count: int, fps: float) -> list[dict]:
    usable_frames = min(max(frame_count, 300), 3000)
    tracking: list[TrackPoint] = []
    team_specs = [("Home", 1, 0.0), ("Away", 12, math.pi)]

    for frame_idx in range(0, usable_frames, 2):
        t = frame_idx / fps
        timestamp_ms = int(t * 1000)
        ball_x = 52.5 + 34 * math.sin(t / 7)
        ball_y = 34 + 20 * math.sin(t / 4.3)
        tracking.append(
            TrackPoint(frame_idx, timestamp_ms, "ball", "ball", None, _clamp(ball_x, 0, 105), _clamp(ball_y, 0, 68), [0, 0, 0, 0], 0.84)
        )

        for team, base_id, phase_shift in team_specs:
            direction = 1 if team == "Home" else -1
            for idx in range(11):
                line = idx // 3
                slot = idx % 3
                base_x = 20 + line * 17 if team == "Home" else 85 - line * 17
                base_y = 14 + slot * 18 + (idx % 2) * 5
                wave = math.sin(t / 5 + idx * 0.7 + phase_shift)
                x = base_x + direction * (6 * wave + 10 * math.sin(t / 12))
                y = base_y + 5 * math.cos(t / 6 + idx)
                tracking.append(
                    TrackPoint(
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        track_id=f"{team[0]}{base_id + idx}",
                        object_type="player",
                        team=team,
                        x=_clamp(x, 0, 105),
                        y=_clamp(y, 0, 68),
                        bbox=[0, 0, 0, 0],
                        confidence=0.78 + (idx % 4) * 0.04,
                    )
                )

    return [asdict(point) for point in tracking]


def _classify_phases(tracking: list[dict]) -> list[dict]:
    ball = [row for row in tracking if row["object_type"] == "ball"]
    phases = []
    current = None
    for row in ball:
        if row["x"] < 35:
            phase = "Build-Up"
        elif row["x"] < 70:
            phase = "Progression"
        elif row["y"] < 18 or row["y"] > 50:
            phase = "Wide Final Attack"
        else:
            phase = "Final Attack"
        if current is None or current["phase"] != phase:
            if current:
                current["end_ms"] = row["timestamp_ms"]
                phases.append(current)
            current = {"phase": phase, "start_ms": row["timestamp_ms"], "end_ms": row["timestamp_ms"], "confidence": 0.67}
    if current:
        current["end_ms"] = ball[-1]["timestamp_ms"] if ball else 0
        phases.append(current)
    return phases


def _build_metrics(tracking: list[dict], phases: list[dict], fps: float, data_quality: dict | None = None) -> dict:
    players = [row for row in tracking if row["object_type"] == "player"]
    by_team = {"Home": [], "Away": []}
    for row in players:
        by_team[row["team"]].append(row)

    team_metrics = {}
    for team, rows in by_team.items():
        xs = [row["x"] for row in rows]
        ys = [row["y"] for row in rows]
        team_metrics[team] = {
            "avg_line_height": round(float(np.mean(xs)), 2) if xs else 0,
            "avg_width": round(float(np.percentile(ys, 90) - np.percentile(ys, 10)), 2) if ys else 0,
            "compactness": round(float(np.std(xs) + np.std(ys)), 2) if xs and ys else 0,
            "distance_estimate_km": round(len(rows) * 0.0027, 2),
            "sprint_count_estimate": int(len(rows) / 95),
        }

    possession = _estimate_possession(tracking)
    player_summaries = _summarize_players(tracking, fps)
    events = _build_key_events(possession, phases)
    return {
        "possession": possession,
        "teams": team_metrics,
        "players": player_summaries,
        "events": events,
        "phase_count": len(phases),
        "data_quality": data_quality or {},
    }


def _estimate_possession(tracking: list[dict]) -> dict:
    by_frame: dict[int, list[dict]] = {}
    for row in tracking:
        by_frame.setdefault(row["frame_idx"], []).append(row)

    counts = {"Home": 0, "Away": 0}
    timeline = []
    for frame_idx, rows in by_frame.items():
        ball = next((row for row in rows if row["object_type"] == "ball"), None)
        players = [row for row in rows if row["object_type"] == "player"]
        if not ball or not players:
            continue
        nearest = min(players, key=lambda p: (p["x"] - ball["x"]) ** 2 + (p["y"] - ball["y"]) ** 2)
        counts[nearest["team"]] += 1
        timeline.append({"frame_idx": frame_idx, "timestamp_ms": ball["timestamp_ms"], "team": nearest["team"]})

    total = max(sum(counts.values()), 1)
    return {
        "home_pct": round(counts["Home"] * 100 / total, 1),
        "away_pct": round(counts["Away"] * 100 / total, 1),
        "timeline": timeline[::10],
    }


def _summarize_players(tracking: list[dict], fps: float) -> list[dict]:
    players = [row for row in tracking if row["object_type"] == "player"]
    by_player: dict[str, list[dict]] = {}
    for row in players:
        by_player.setdefault(row["track_id"], []).append(row)

    summaries = []
    for track_id, rows in by_player.items():
        rows = sorted(rows, key=lambda row: row["frame_idx"])
        distance = 0.0
        max_speed = 0.0
        for first, second in zip(rows, rows[1:]):
            dt = max((second["frame_idx"] - first["frame_idx"]) / fps, 0.001)
            step = math.dist((first["x"], first["y"]), (second["x"], second["y"]))
            distance += step
            max_speed = max(max_speed, (step / dt) * 3.6)
        xs = [row["x"] for row in rows]
        ys = [row["y"] for row in rows]
        influence = (distance * 0.8) + (max(xs) - min(xs) if xs else 0) + ((max(ys) - min(ys)) * 0.55 if ys else 0)
        summaries.append(
            {
                "track_id": track_id,
                "team": rows[0].get("team"),
                "distance_m": round(distance, 1),
                "max_speed_kmh": round(max_speed, 1),
                "avg_x": round(float(np.mean(xs)), 1) if xs else 0,
                "avg_y": round(float(np.mean(ys)), 1) if ys else 0,
                "influence": round(influence, 1),
            }
        )
    return sorted(summaries, key=lambda row: row["influence"], reverse=True)[:12]


def _build_key_events(possession: dict, phases: list[dict]) -> list[dict]:
    events = []
    last_team = None
    for item in possession.get("timeline", []):
        team = item["team"]
        if last_team and team != last_team:
            events.append(
                {
                    "timestamp_ms": item["timestamp_ms"],
                    "type": "Possession Swing",
                    "team": team,
                    "description": f"{team} regained control after a possession change.",
                    "confidence": 0.7,
                }
            )
        last_team = team

    for phase in phases[:8]:
        duration = phase["end_ms"] - phase["start_ms"]
        if duration >= 1800:
            events.append(
                {
                    "timestamp_ms": phase["start_ms"],
                    "type": "Phase Entry",
                    "team": None,
                    "description": f"{phase['phase']} phase sustained for {round(duration / 1000)} seconds.",
                    "confidence": phase.get("confidence", 0.65),
                }
            )

    return sorted(events, key=lambda row: row["timestamp_ms"])[:10]


def _render_annotated_video(input_video: Path, output_path: Path, tracking: list[dict], metadata: dict) -> None:
    if cv2 is None:
        shutil.copyfile(input_video, output_path)
        return

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        _render_pitch_animation(output_path, tracking, metadata)
        return

    fps = metadata["fps"] or 25
    width = metadata["width"] or 1280
    height = metadata["height"] or 720
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    by_frame = _tracking_by_frame(tracking)
    frame_idx = 0

    while frame_idx < min(metadata["frame_count"], 900):
        ok, frame = capture.read()
        if not ok:
            break
        _draw_overlay(frame, by_frame.get(frame_idx if frame_idx % 2 == 0 else frame_idx - 1, []))
        writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()


def _render_pitch_animation(output_path: Path, tracking: list[dict], metadata: dict) -> None:
    width, height, fps = 1280, 720, metadata.get("fps", 25) or 25
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    by_frame = _tracking_by_frame(tracking)
    for frame_idx in sorted(by_frame.keys())[:450]:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (25, 45, 35)
        cv2.rectangle(frame, (60, 60), (1220, 660), (120, 160, 130), 2)
        cv2.line(frame, (640, 60), (640, 660), (120, 160, 130), 2)
        _draw_overlay(frame, by_frame[frame_idx])
        writer.write(frame)
    writer.release()


def _draw_overlay(frame, rows: list[dict]) -> None:
    height, width = frame.shape[:2]
    for row in rows:
        px = int((row["x"] / 105) * width)
        py = int((row["y"] / 68) * height)
        if row["object_type"] == "ball":
            cv2.circle(frame, (px, py), 7, (30, 240, 255), -1)
        else:
            color = (72, 128, 255) if row["team"] == "Home" else (240, 80, 120)
            cv2.circle(frame, (px, py), 11, color, -1)
            cv2.putText(frame, row["track_id"], (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1)
    cv2.putText(frame, "AI tactical tracking preview", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2)


def _build_report(metrics: dict, phases: list[dict]) -> dict:
    home = metrics["teams"]["Home"]
    away = metrics["teams"]["Away"]
    phase_names = ", ".join(dict.fromkeys(phase["phase"] for phase in phases[:8]))
    return {
        "title": "Initial Tactical Intelligence Report",
        "summary": (
            f"Home controlled {metrics['possession']['home_pct']}% of estimated possession. "
            f"The dominant detected phases were {phase_names}. "
            f"Home average line height was {home['avg_line_height']}m versus Away at {away['avg_line_height']}m."
        ),
        "recommendations": [
            {
                "label": "Review progression spacing",
                "detail": "Compare team width and compactness during Progression phases to identify whether support lines are too stretched.",
                "confidence": 0.66,
            },
            {
                "label": "Validate possession clips",
                "detail": "Use the possession timeline to manually approve ball-control switches before downstream tactical conclusions.",
                "confidence": 0.72,
            },
            {
                "label": "Upgrade calibration",
                "detail": "Enable Eagle-style keypoint homography for production-grade phase maps and distance metrics.",
                "confidence": 0.81,
            },
        ],
        "evidence": {
            "possession": metrics["possession"],
            "data_quality": metrics["data_quality"],
        },
    }


def _tracking_by_frame(tracking: list[dict]) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    for row in tracking:
        by_frame.setdefault(row["frame_idx"], []).append(row)
    return by_frame


def _preview_tracking(tracking: list[dict]) -> list[dict]:
    frames = sorted({row["frame_idx"] for row in tracking})
    if len(frames) <= 70:
        allowed = set(frames)
    else:
        stride = max(1, len(frames) // 70)
        allowed = set(frames[::stride])
    return [row for row in tracking if row["frame_idx"] in allowed]


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tick(progress: Progress | None, stage: str, pct: float) -> None:
    if progress:
        progress(stage, pct)
    time.sleep(0.08)


def _clamp(value: float, low: float, high: float) -> float:
    return round(max(low, min(high, value)), 3)
