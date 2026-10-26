"""Submit an authenticated short-video job to the deployed Modal worker."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/storage/outputs")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error("Video file does not exist")
    from workers.video.modal_client import run_modal_analysis
    result = run_modal_analysis(args.video.resolve(), args.output.resolve(),
                                lambda stage, fraction: print(f"{fraction:.0%} {stage}", file=sys.stderr, flush=True))
    summary = result["detection_summary"]
    print(json.dumps(dict(run_id=result["run_id"], output=str(args.output / result["run_id"]),
                          analyzed_frames=summary["analyzed_frames"], device=summary["device"],
                          gpu=summary.get("gpu"), processing_seconds=summary["processing_seconds"],
                          ball_detected_frames=summary["ball_detected_frames"]), indent=2))


if __name__ == "__main__":
    main()
