"""CPU upload/status/media gateway; GPU inference stays in the private worker app."""
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workers.video.modal_policy import APP_NAME, VOLUME_NAME

app = modal.App("laas-football-web")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi>=0.115", "python-multipart>=0.0.9", "opencv-python-headless>=4.10")
if modal.is_local():
    image = image.add_local_dir(str(ROOT / "workers"), "/app/workers", copy=True,
                                ignore=["**/__pycache__/**", "**/*.pyc"]).add_local_dir(
        str(ROOT / "apps/api"), "/app/apps/api", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.env({"PYTHONPATH": "/app"})


@app.function(image=image, cpu=0.5, memory=2048, volumes={"/jobs": volume},
              secrets=[modal.Secret.from_name("laas-web-access")], min_containers=0,
              max_containers=1, scaledown_window=15, timeout=300)
@modal.asgi_app()
def web():
    import os
    from apps.api.modal_gateway import create_gateway
    return create_gateway(volume, modal.Function.from_name(APP_NAME, "analyze"), os.environ["APP_ACCESS_KEY"])
