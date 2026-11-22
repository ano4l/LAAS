"""A run with no model output: a stored video and an empty review, for a person to tag by hand.

Detection is what limits clip length. Tagging needs only playback, so a full half can be
uploaded this way and its events recorded under Review. Nothing is detected, tracked or
estimated for such a run, and its manifest says so.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

ENGINE = "review_only"
# Codecs browsers commonly play. Anything else is stored and can be tagged only if this browser happens to play it.
BROWSER_CODECS = {"avc1", "h264", "x264", "vp80", "vp08", "vp90", "vp09", "av01"}
WARNING = "No detection or tracking was run on this video. Every figure for it comes from events a person records and approves."


def video_codec(path: Path) -> str | None:
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        code = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    text = "".join(chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24)).strip("\x00 ").lower()
    return text if text.isalnum() else None


def create_run(video: Path, output_root: Path, metadata: dict, media_path: str) -> dict:
    """Write the manifest for a tagging-only run of `video`, which is served to the browser at `media_path`."""
    run_id = uuid4().hex
    directory = output_root / run_id
    directory.mkdir(parents=True)
    with video.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    codec = video_codec(video)
    warnings = [WARNING]
    if codec not in BROWSER_CODECS:
        warnings.append(f"This video uses the {codec or 'an unrecognised'} codec, which many browsers cannot play. "
                        "If it does not play, export it as an MP4 with H.264 video and upload it again.")
    manifest = dict(
        schema_version="1.0", run_id=run_id, engine=ENGINE, metadata=dict(metadata, sha256=digest, codec=codec),
        tracking=[], warnings=warnings,
        metrics=dict(possession=dict(home_pct=None, away_pct=None, timeline=[]), teams={}, players=[], events=[],
                     data_quality=dict(mode=ENGINE), phase_count=0),
        phases=[], report=dict(title="Video for tagging by hand", summary=WARNING, recommendations=[]),
        files=dict(annotated_video=media_path, source_video=media_path, manifest_json=f"/media/outputs/{run_id}/manifest.json"))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
