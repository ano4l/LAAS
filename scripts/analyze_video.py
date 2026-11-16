"""Run the same streaming worker locally or in a GPU container."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ball-model", type=Path)
    parser.add_argument("--device", help="cpu or CUDA device index such as 0")
    parser.add_argument("--fps", type=float, help="Target analysis frames per second")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--profile", choices=["accurate", "balanced", "preview"])
    parser.add_argument("--ball-tile-size", type=int, help="Experimental overlapping ball crops; 0 disables")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error("Input video does not exist")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    for field, env in [("model", "YOLO_MODEL_PATH"), ("ball_model", "YOLO_BALL_MODEL_PATH"),
                       ("device", "YOLO_DEVICE"), ("fps", "ANALYSIS_FPS"), ("image_size", "YOLO_IMAGE_SIZE"),
                       ("profile", "ANALYSIS_PROFILE"), ("ball_tile_size", "BALL_TILE_SIZE")]:
        if getattr(args, field) is not None:
            os.environ[env] = str(getattr(args, field))
    requested_device = os.getenv("YOLO_DEVICE")
    if requested_device and requested_device != "cpu":
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable. Check the GPU runtime, driver and PyTorch CUDA build.")
    from workers.video.real_analysis import run_real_analysis
    output = args.output or Path(os.getenv("STORAGE_ROOT", ROOT / "data/storage")) / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    result = run_real_analysis(args.video.resolve(), output.resolve(),
                              lambda stage, fraction: print(f"{fraction:.0%} {stage}", file=sys.stderr, flush=True))
    print(json.dumps(dict(run_id=result["run_id"], manifest=str(output.resolve() / result["run_id"] / "manifest.json"),
                          summary=result["detection_summary"]), indent=2))


if __name__ == "__main__":
    main()
