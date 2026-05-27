from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from workers.video.pipeline import TrackPoint


Progress = Callable[[str, float], None]


def yolo_capabilities(root: Path) -> dict:
    model_path = _default_model_path(root)
    return {
        "engine": "yolo",
        "available": bool(cv2 and model_path.exists() and importlib.util.find_spec("ultralytics")),
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "ultralytics_installed": bool(importlib.util.find_spec("ultralytics")),
        "opencv_available": cv2 is not None,
    }


def try_run_yolo_analysis(input_video: Path, root: Path, progress: Progress | None = None) -> dict | None:
    caps = yolo_capabilities(root)
    if not caps["available"]:
        return None

    from ultralytics import YOLO

    if cv2 is None:
        return None

    model = YOLO(caps["model_path"])
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        return None

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = min(frame_count or 450, 450)

    frames = []
    frame_indices = []
    idx = 0
    stride = max(1, math.floor((frame_count or max_frames) / max_frames))
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
            frame_indices.append(idx)
        idx += 1
    capture.release()

    if not frames:
        return None

    if progress:
        progress("YOLO detecting objects", 0.28)

    results = model.predict(frames, conf=0.16, verbose=False)
    tracking = _detections_to_tracks(results, frame_indices, fps, width, height)
    if len([row for row in tracking if row["object_type"] == "player"]) < 8:
        return None

    return {
        "tracking": tracking,
        "metadata": {
            "filename": input_video.name,
            "fps": round(float(fps), 2),
            "frame_count": frame_count or len(frames),
            "duration_seconds": round((frame_count or len(frames)) / fps, 2) if fps else 0,
            "width": width,
            "height": height,
        },
        "engine": "yolo_image_projection",
        "quality": {
            "tracking_confidence": _average_confidence(tracking),
            "homography_quality": 0.25,
            "ball_missing_ratio": _ball_missing_ratio(tracking, len(frames)),
            "mode": "yolo_image_projection",
        },
    }


def _detections_to_tracks(results, frame_indices: list[int], fps: float, width: int, height: int) -> list[dict]:
    tracks: list[TrackPoint] = []
    active: dict[str, list[tuple[str, float, float]]] = {"player": [], "ball": []}
    next_id = 1

    for result, frame_idx in zip(results, frame_indices):
        names = result.names
        timestamp_ms = int((frame_idx / fps) * 1000)
        detections = []
        for box in result.boxes:
            cls_name = str(names[int(box.cls[0])]).lower()
            object_type = _object_type(cls_name)
            if object_type is None:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0])
            cx = (x1 + x2) / 2
            cy = y2 if object_type == "player" else (y1 + y2) / 2
            detections.append((object_type, cx, cy, [x1, y1, x2, y2], confidence))

        current: dict[str, list[tuple[str, float, float]]] = {"player": [], "ball": []}
        for object_type, cx, cy, bbox, confidence in detections:
            track_id, next_id = _assign_track(active[object_type], object_type, cx, cy, next_id)
            current[object_type].append((track_id, cx, cy))
            pitch_x = max(0, min(105, cx / width * 105))
            pitch_y = max(0, min(68, cy / height * 68))
            team = _team_from_x(pitch_x) if object_type == "player" else None
            tracks.append(
                asdict(
                    TrackPoint(
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        track_id=track_id,
                        object_type=object_type,
                        team=team,
                        x=round(pitch_x, 3),
                        y=round(pitch_y, 3),
                        bbox=[round(value, 2) for value in bbox],
                        confidence=round(confidence, 3),
                    )
                )
            )
        active = current

    return tracks


def _object_type(cls_name: str) -> str | None:
    if cls_name in {"player", "goalkeeper"}:
        return "player"
    if cls_name == "ball":
        return "ball"
    return None


def _assign_track(active: list[tuple[str, float, float]], object_type: str, cx: float, cy: float, next_id: int) -> tuple[str, int]:
    threshold = 90 if object_type == "player" else 130
    if active:
        track_id, px, py = min(active, key=lambda item: (item[1] - cx) ** 2 + (item[2] - cy) ** 2)
        if math.dist((px, py), (cx, cy)) <= threshold:
            return track_id, next_id
    if object_type == "ball":
        return "ball", next_id
    track_id = f"Y{next_id}"
    return track_id, next_id + 1


def _team_from_x(pitch_x: float) -> str:
    return "Home" if pitch_x <= 52.5 else "Away"


def _average_confidence(tracking: list[dict]) -> float:
    if not tracking:
        return 0
    return round(float(np.mean([row["confidence"] for row in tracking])), 3)


def _ball_missing_ratio(tracking: list[dict], frame_count: int) -> float:
    seen = {row["frame_idx"] for row in tracking if row["object_type"] == "ball"}
    return round(1 - (len(seen) / max(frame_count, 1)), 3)


def _default_model_path(root: Path) -> Path:
    return root / "references" / "FootBall-Analytics-MVP" / "yolo11l.pt"

