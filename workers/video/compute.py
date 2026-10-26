"""Compute routing without silently falling back from a failed cloud job."""
import importlib.util
import os


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
    return dict(engine="yolo_bytetrack", compute_provider=provider, available=not blockers, blockers=blockers,
                deployment_verified=False, note="Deployment availability is checked when submitting a job")
