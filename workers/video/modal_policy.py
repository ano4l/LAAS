"""Shared limits for the private Modal prototype; not an account billing cap."""
from datetime import datetime, timezone
from pathlib import Path
import re

APP_NAME = "laas-football"
VOLUME_NAME = "laas-football-jobs"
MAX_BYTES = 250 * 1024 * 1024
MAX_SECONDS = 120
MAX_MONTHLY_ATTEMPTS = 20
MAX_DAILY_ATTEMPTS = 5
ARTIFACTS = ("manifest.json", "tracking.json", "metrics.json", "report.json", "annotated.mp4", "encoder.log")


def validate_request(request_id, profile="accurate", tile_size=0):
    if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ValueError("Invalid Modal request ID")
    if profile not in {"accurate", "preview"} or tile_size not in {0, 640}:
        raise ValueError("Unsupported Modal analysis options")


def validate_video(path: Path, metadata: dict):
    import math
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Modal prototype accepts videos up to 250 MiB")
    if not math.isfinite(metadata["duration_seconds"]) or not 0 < metadata["duration_seconds"] <= MAX_SECONDS:
        raise ValueError("Modal prototype accepts clips up to 120 seconds; trim the video first")
    if metadata["width"] * metadata["height"] > 3840 * 2160 or not 0 < metadata["fps"] <= 60:
        raise ValueError("Modal prototype accepts up to 4K / 60fps footage")


def reserve_attempt(ledger: dict, now=None):
    """Call only from the single, non-concurrent GPU function; commit before inference."""
    now = now or datetime.now(timezone.utc)
    month, day = now.strftime("%Y-%m"), now.strftime("%Y-%m-%d")
    months, days = ledger.setdefault("months", {}), ledger.setdefault("days", {})
    if months.get(month, 0) >= MAX_MONTHLY_ATTEMPTS:
        raise ValueError("Modal prototype monthly attempt limit reached (20). No analysis started.")
    if days.get(day, 0) >= MAX_DAILY_ATTEMPTS:
        raise ValueError(f"Modal prototype daily attempt limit reached ({MAX_DAILY_ATTEMPTS}). No analysis started.")
    months[month], days[day] = months.get(month, 0) + 1, days.get(day, 0) + 1
    return ledger
