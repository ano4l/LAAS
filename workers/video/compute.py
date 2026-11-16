"""Compute routing. A failed cloud job falls back to this machine only when asked to, and says so."""
from contextlib import contextmanager
import importlib.util
import json
import logging
import os


def fallback_enabled():
    value = os.getenv("ANALYSIS_FALLBACK", "none")
    if value not in {"none", "local"}:
        raise ValueError("ANALYSIS_FALLBACK must be none or local")
    return value == "local"


@contextmanager
def _environment(values):
    """Jobs run one at a time on a single worker thread, so a temporary overlay is safe."""
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_local_fallback(input_video, output_root, progress, cause):
    """Run on this machine after a cloud failure and record why in the run itself."""
    from workers.video.real_analysis import run_real_analysis
    reason = f"{type(cause).__name__}: {cause}"[:300]
    logging.getLogger(__name__).warning("Modal analysis failed (%s); running on this machine instead", reason)
    if progress:
        progress("Cloud GPU unavailable · analysing on this server (slower)", 0.02)
    overlay = dict(ANALYSIS_PROFILE=os.getenv("FALLBACK_PROFILE", "balanced"))
    if os.getenv("FALLBACK_TRACKER"):
        overlay["TRACKER_BACKEND"] = os.environ["FALLBACK_TRACKER"]
    with _environment(overlay):
        result = run_real_analysis(input_video, output_root, progress)
    result["detection_summary"].update(compute_provider="local", compute_fallback=dict(requested="modal", reason=reason))
    result.setdefault("warnings", []).append("The cloud GPU was unavailable, so this clip was analysed on the server CPU with the reduced fallback settings.")
    (output_root / result["run_id"] / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def analysis_capabilities(root):
    from workers.video.yolo_adapter import yolo_capabilities
    provider = os.getenv("ANALYSIS_COMPUTE", "local")
    if provider == "local":
        return dict(yolo_capabilities(root), compute_provider="local")
    blockers = []
    if provider != "modal":
        blockers.append("ANALYSIS_COMPUTE must be local or modal")
    if not importlib.util.find_spec("modal"):
        blockers.append("Install Modal: pip install -e .[modal]")
    else:
        from modal.config import config
        if not (config.get("token_id") and config.get("token_secret")):
            blockers.append("Authenticate Modal: python -m modal token new")
    if blockers and provider == "modal" and fallback_enabled():
        # Uploads stay possible: the job will run on this machine and report the fallback.
        return dict(yolo_capabilities(root), compute_provider="local", fallback_from="modal", cloud_blockers=blockers)
    return dict(engine="yolo_bytetrack", compute_provider=provider, available=not blockers, blockers=blockers,
                deployment_verified=False, note="Deployment availability is checked when submitting a job")
