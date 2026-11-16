"""Start the LAAS dashboard on a Windows server: Modal GPU first, this machine's CPU as the fallback."""
import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Readable by the Local Service account the scheduled task runs as; a user profile is not.
MODAL_CONFIG = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "LAAS Football Analytics" / "modal.toml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=Path, default=Path(os.environ["LOCALAPPDATA"]) / "LAAS Football Analytics")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--compute", choices=["modal", "local"], default=os.getenv("LAAS_COMPUTE", "modal"),
                        help="modal: cloud GPU with CPU fallback (default). local: this machine only.")
    args = parser.parse_args()
    storage = args.storage.resolve()
    logs = storage / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(ROOT))
    from workers.video.real_analysis import physical_cores
    threads = os.getenv("YOLO_CPU_THREADS") or str(physical_cores())
    # ONNX Runtime ran the detector seven times faster than PyTorch on the server's CPU, which has no AVX.
    # LAAS_RUNTIME=pytorch goes back to the original weights.
    import importlib.util
    exported = os.getenv("LAAS_RUNTIME", "onnx") == "onnx" and importlib.util.find_spec("onnxruntime") is not None
    def weights(path):
        return path.with_suffix(".onnx") if exported and path.with_suffix(".onnx").is_file() else path
    detector = weights(ROOT / "models" / "football.pt")
    pitch = weights(ROOT / "models" / "pitch-keypoints" / "football-pitch-detection.pt")
    os.environ.update(
        APP_ENV="production",
        YOLO_DEVICE="cpu",
        YOLO_MODEL_PATH=str(detector),
        YOLO_PITCH_MODEL_PATH=str(pitch) if pitch.is_file() else "off",
        YOLO_CONFIG_DIR=str(storage / "ultralytics"),
        STORAGE_ROOT=str(storage / "data"),
        MAX_UPLOAD_MB="250",
        CORS_ORIGINS=f"http://127.0.0.1:{args.port},http://localhost:{args.port}",
        YOLO_CPU_THREADS=threads,
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        # CPU settings: analyse about 12 frames a second at 960 px and write only those frames to the playback video.
        ANNOTATED_FRAMES="analysed",
    )
    if args.compute == "modal":
        # The cloud run uses full settings; the fallback profile applies only when a job runs here instead.
        os.environ.update(ANALYSIS_COMPUTE="modal", ANALYSIS_FALLBACK="local", ANALYSIS_PROFILE="accurate",
                          FALLBACK_PROFILE="balanced")
        if MODAL_CONFIG.is_file():
            os.environ.setdefault("MODAL_CONFIG_PATH", str(MODAL_CONFIG))
    else:
        os.environ.update(ANALYSIS_COMPUTE="local", ANALYSIS_PROFILE="balanced")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[RotatingFileHandler(logs / "server.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")],
    )

    import uvicorn

    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=args.port, workers=1, log_config=None)


if __name__ == "__main__":
    main()
