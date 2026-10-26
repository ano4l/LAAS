"""Publish a verified existing real run as the reusable hosted sample (no GPU call)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.video.measurements import from_tracking
from workers.video.insights import ensure_insights
from workers.video.modal_policy import ARTIFACTS, VOLUME_NAME
from workers.video.real_analysis import video_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--library-only", action="store_true", help="Publish a completed result without changing the reusable sample")
    parser.add_argument("--label", help="Display filename in the hosted Library; source hash remains unchanged")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("engine") != "yolo_bytetrack":
        raise ValueError("Sample must be a real detection run")
    with args.source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["metadata"]["sha256"]:
            raise ValueError("Source does not match analysis")
    video = video_metadata(args.run / "annotated.mp4")
    if (video["frame_count"] != manifest["metadata"]["frame_count"] or
            abs(video["duration_seconds"] - manifest["metadata"]["duration_seconds"]) > .1):
        raise ValueError("Sample playback does not cover full source duration")
    manifest["metadata"]["source_filename"] = args.source.name
    manifest["metadata"]["filename"] = args.label or args.source.name
    manifest["metrics"]["video_measurements"] = from_tracking(
        json.loads((args.run / "tracking.json").read_text(encoding="utf-8")),
        manifest["metadata"], manifest["detection_summary"])
    ensure_insights(manifest, args.run)
    job_id = uuid4().hex
    job = dict(id=job_id, status="complete", stage="Complete", progress=1,
               created_at=datetime.now(timezone.utc).isoformat(), result=manifest,
               error=None, run_id=manifest["run_id"], sample=not args.library_only, filename=args.label or args.source.name)
    import modal
    volume = modal.Volume.from_name(VOLUME_NAME)
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (staging / "metrics.json").write_text(json.dumps(manifest["metrics"]), encoding="utf-8")
        (staging / "job.json").write_text(json.dumps(job), encoding="utf-8")
        with volume.batch_upload() as batch:
            batch.put_file(str(args.source), f"/requests/{job_id}/source.mp4")
            for name in ARTIFACTS:
                source = staging / name if name in {"manifest.json", "metrics.json"} else args.run / name
                batch.put_file(str(source), f"/requests/{job_id}/outputs/{manifest['run_id']}/{name}")
            batch.put_file(str(staging / "job.json"), f"/web-jobs/{job_id}.json")
    if not args.library_only:
        (args.run.parent / "sample.json").write_text(json.dumps(dict(
            id=job_id, run_id=manifest["run_id"], created_at=job["created_at"])), encoding="utf-8")
    print(json.dumps(dict(job_id=job_id, duration_seconds=video["duration_seconds"], frames=video["frame_count"],
                          measurements=manifest["metrics"]["video_measurements"]["analyzed_frames"])))


if __name__ == "__main__":
    main()
