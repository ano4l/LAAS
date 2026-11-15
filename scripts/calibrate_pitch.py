"""Add automatic pitch calibration to an existing run, without re-tracking.

Reads the run's source video, calibrates the frames the run analysed, writes
pitch-calibration.json and recomputes the insights that use it. Detections and
tracks are untouched. See docs/research/AUTOMATIC_PITCH_CALIBRATION.md.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream)
        temporary = stream.name
    try:
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
    import cv2
    from workers.video.insights import from_tracking
    from workers.video.pitch_keypoints import PitchKeypointModel, document
    from workers.video.sequence_context import load_context
    from workers.video.yolo_adapter import pitch_model_path

    weights = args.model or pitch_model_path(ROOT)
    if not weights or not weights.is_file():
        parser.error("No pitch landmark model found. Pass --model or set YOLO_PITCH_MODEL_PATH.")
    manifest = json.loads((args.run_directory / "manifest.json").read_text(encoding="utf-8"))
    metadata, summary = manifest["metadata"], manifest["detection_summary"]
    with args.video.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != metadata.get("sha256"):
            parser.error("This video is not the run's source: SHA-256 differs")
    frame_stride = max(1, round(metadata["fps"] / summary["sample_fps"]))
    wanted = set(range(0, summary["decoded_frames"], frame_stride * args.stride))
    model, frames, index = PitchKeypointModel(weights, args.device), {}, 0
    capture = cv2.VideoCapture(str(args.video))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = model.calibrate(frame)
                if len(frames) % 25 == 0:
                    print(f"{len(frames)} / {len(wanted)} frames", flush=True)
            index += 1
    finally:
        capture.release()
    if index < summary["decoded_frames"] - max(2, frame_stride):
        raise SystemExit(f"Video decoding stopped at frame {index} of {summary['decoded_frames']}; nothing was written.")
    with weights.open("rb") as stream:
        calibration = document(frames, weights.name, hashlib.file_digest(stream, "sha256").hexdigest(), frame_stride * args.stride)
    write_json(args.run_directory / "pitch-calibration.json", calibration)
    summary["pitch_calibration"] = {key: calibration[key] for key in ("schema_version", "method", "model", "model_sha256", "frame_stride", "assumed_pitch", "summary")}
    note = ("Pitch positions in the shape view are estimates from automatic per-frame calibration that assumes a 105 x 68 m pitch. "
            "They are unreviewed, not checked against measured positions, and absent for frames that fail the calibration checks.")
    if note not in manifest.setdefault("warnings", []):
        manifest["warnings"].append(note)
    context = load_context(args.run_directory / "sequence-context.json", manifest)
    manifest.setdefault("metrics", {})["video_insights"] = from_tracking(
        json.loads((args.run_directory / "tracking.json").read_text(encoding="utf-8")), metadata, summary, context, frames)
    write_json(args.run_directory / "metrics.json", manifest["metrics"])
    write_json(args.run_directory / "manifest.json", manifest)
    print(json.dumps(calibration["summary"], indent=2))


if __name__ == "__main__":
    main()
