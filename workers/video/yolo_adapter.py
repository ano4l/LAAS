"""Model/runtime discovery independent of application code."""
import importlib.util
import os
from pathlib import Path


def model_path(root: Path) -> Path:
    if os.getenv('YOLO_MODEL_PATH'):
        return Path(os.environ['YOLO_MODEL_PATH']).expanduser().resolve()
    packaged = root / 'models' / 'football.pt'
    football = root / 'references' / 'Soccer-Analytics' / 'models' / 'best.pt'
    reference = root / 'references' / 'FootBall-Analytics-MVP' / 'yolo11l.pt'
    return next((path for path in (packaged, football, reference) if path.is_file()), packaged)


def yolo_capabilities(root: Path) -> dict:
    path = model_path(root)
    installed = bool(importlib.util.find_spec('ultralytics'))
    opencv = bool(importlib.util.find_spec('cv2'))
    encoder = bool(importlib.util.find_spec('imageio_ffmpeg'))
    blockers = []
    if not path.is_file():
        blockers.append('Set YOLO_MODEL_PATH to a trusted football detector .pt file.')
    if os.getenv('YOLO_BALL_MODEL_PATH') and not Path(os.environ['YOLO_BALL_MODEL_PATH']).expanduser().is_file():
        blockers.append('YOLO_BALL_MODEL_PATH does not point to an existing ball detector file.')
    if os.getenv('YOLO_BALL_RESCUE_PATH') and not Path(os.environ['YOLO_BALL_RESCUE_PATH']).expanduser().is_file():
        blockers.append('YOLO_BALL_RESCUE_PATH does not point to an existing secondary ball model.')
    if not installed:
        blockers.append('Install the Python CV dependencies: pip install -e .[cv]')
    if not opencv:
        blockers.append('OpenCV is missing.')
    if not encoder:
        blockers.append('Install imageio-ffmpeg for browser-compatible H.264 output.')
    return dict(engine='yolo_bytetrack', available=not blockers, model_path=str(path),
                model_exists=path.is_file(), ultralytics_installed=installed,
                opencv_available=opencv, encoder_available=encoder, blockers=blockers)
