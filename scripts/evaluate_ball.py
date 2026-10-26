"""Prepare a review set or score detections against manually reviewed ball boxes.

prepare VIDEO DIRECTORY --frames 50
score LABELS_JSON RUN_DIRECTORY
Labels use original-image pixel xyxy boxes. An empty reviewed boxes list means no
visible ball. Leave ambiguous/occluded frames unreviewed. Never use predictions
as ground truth. Use separate matches for tuning and final evaluation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.video.accuracy import box_iou


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(labels, predictions, confidence=0.1, iou=0.5):
    if not 0 <= confidence <= 1 or not 0 < iou <= 1:
        raise ValueError("Invalid confidence or IoU threshold")
    reviewed = [frame for frame in labels["frames"] if frame.get("reviewed") is True]
    if not reviewed:
        raise ValueError("No manually reviewed frames. Mark reviewed only after checking all visible balls.")
    ids = [frame["frame_idx"] for frame in reviewed]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate reviewed frame indexes")
    grouped = {}
    for row in predictions:
        if row["object_type"] in {"ball", "sports ball"} and row["confidence"] >= confidence:
            grouped.setdefault(row["frame_idx"], []).append(row)
    tp = fp = fn = 0
    details = []
    for frame in reviewed:
        truth = frame["boxes"]
        for box in truth:
            if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box) or not 0 <= box[0] < box[2] <= labels["width"] or not 0 <= box[1] < box[3] <= labels["height"]:
                raise ValueError("Labels must be valid original-image xyxy boxes")
        remaining = set(range(len(truth)))
        hits = false = 0
        for row in sorted(grouped.get(frame["frame_idx"], []), key=lambda r: r["confidence"], reverse=True):
            best = max(remaining, key=lambda index: box_iou(row["bbox"], truth[index]), default=None)
            if best is not None and box_iou(row["bbox"], truth[best]) >= iou:
                remaining.remove(best)
                hits += 1
            else:
                false += 1
        tp += hits
        fp += false
        fn += len(remaining)
        details.append(dict(frame_idx=frame["frame_idx"], tp=hits, fp=false, fn=len(remaining)))
    return dict(reviewed_frames=len(reviewed), excluded_unreviewed_frames=len(labels["frames"]) - len(reviewed),
                true_positives=tp, false_positives=fp, false_negatives=fn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                confidence_threshold=confidence, iou_threshold=iou, frames=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("video", type=Path)
    prepare.add_argument("directory", type=Path)
    prepare.add_argument("--frames", type=int, default=50)
    score = commands.add_parser("score")
    score.add_argument("labels", type=Path)
    score.add_argument("run", type=Path)
    score.add_argument("--confidence", type=float, default=0.1)
    score.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "prepare":
        import cv2
        from workers.video.real_analysis import video_metadata
        if args.frames < 1:
            parser.error("--frames must be positive")
        metadata = video_metadata(args.video)
        args.directory.mkdir(parents=True, exist_ok=False)
        total = metadata["frame_count"]
        count = min(total, args.frames)
        indexes = sorted(set(round(i * (total - 1) / max(1, count - 1)) for i in range(count)))
        cap = cv2.VideoCapture(str(args.video))
        frames = []
        try:
            for index in indexes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode review frame {index}")
                name = f"frame-{index:08d}.png"
                if not cv2.imwrite(str(args.directory / name), frame):
                    raise ValueError("Could not save review image")
                frames.append(dict(frame_idx=index, image=name, reviewed=False, boxes=[]))
        finally:
            cap.release()
        labels = dict(schema_version=1, video_sha256=digest(args.video), width=metadata["width"],
                      height=metadata["height"], frames=frames)
        (args.directory / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
        template = Path(__file__).with_name("ball_review.html").read_text(encoding="utf-8")
        (args.directory / "review.html").write_text(template.replace("__LABELS_JSON__", json.dumps(labels).replace("<", "\\u003c")), encoding="utf-8")
        print(f"Prepared {count} UNREVIEWED frames in {args.directory}. Annotate boxes before scoring.")
    else:
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
        if not labels.get("video_sha256") or labels["video_sha256"] != manifest["metadata"].get("sha256"):
            parser.error("Labels and run must match the same source-video SHA256. Rerun older analyses.")
        if (labels["width"], labels["height"]) != (manifest["metadata"]["width"], manifest["metadata"]["height"]):
            parser.error("Label dimensions do not match the source video")
        if args.confidence < manifest["detection_summary"]["ball_confidence"]:
            parser.error("Scoring threshold cannot be below the run's detection threshold; rerun inference at the lower threshold")
        if any(not isinstance(f["frame_idx"], int) or not 0 <= f["frame_idx"] < manifest["metadata"]["frame_count"] for f in labels["frames"]):
            parser.error("Label frame indexes must be within the source video")
        predictions = json.loads((args.run / "tracking.json").read_text(encoding="utf-8"))
        result = evaluate(labels, predictions, args.confidence, args.iou)
        result.update(run_id=manifest["run_id"], video_sha256=labels["video_sha256"], labels_sha256=digest(args.labels),
                      model_sha256=manifest["detection_summary"]["model_sha256"])
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
