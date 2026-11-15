"""Add automatic pitch calibration to an existing run, without re-tracking.

Reads the run's source video, calibrates the frames the run analysed, writes
pitch-calibration.json and recomputes the insights that use it. Detections and
tracks are untouched. See docs/research/AUTOMATIC_PITCH_CALIBRATION.md.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="The exact source video of the run")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--model", type=Path, help="Pitch landmark weights; defaults to YOLO_PITCH_MODEL_PATH or the packaged file")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device index such as 0")
    parser.add_argument("--stride", type=int, default=1, help="Calibrate every Nth analysed frame")
    args = parser.parse_args()
    if not args.video.is_file() or not (args.run_directory / "manifest.json").is_file() or not (args.run_directory / "tracking.json").is_file():
        parser.error("Need an existing video and a run directory with manifest.json and tracking.json")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    from workers.video.pitch_keypoints import calibrate_run
    from workers.video.yolo_adapter import pitch_model_path

    weights = args.model or pitch_model_path(ROOT)
    if not weights or not weights.is_file():
        parser.error("No pitch landmark model found. Pass --model or set YOLO_PITCH_MODEL_PATH.")
    try:
        manifest = calibrate_run(args.video, args.run_directory, weights, args.device, args.stride,
                                 lambda done, total: print(f"{done} / {total} frames", flush=True))
    except ValueError as error:
        raise SystemExit(str(error))
    print(json.dumps(manifest["detection_summary"]["pitch_calibration"]["summary"], indent=2))


if __name__ == "__main__":
    main()
