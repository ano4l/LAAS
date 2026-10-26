"""Authenticated SDK bridge: upload a clip, run GPU inference, import real artifacts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from workers.video.modal_policy import APP_NAME, VOLUME_NAME, ARTIFACTS, validate_request, validate_video


def run_modal_analysis(input_video: Path, output_root: Path, progress=None):
    import modal
    from workers.video.real_analysis import video_metadata

    validate_video(input_video, video_metadata(input_video))
    request_id = uuid4().hex
    profile = os.getenv("ANALYSIS_PROFILE", "accurate")
    tile_size = int(os.getenv("BALL_TILE_SIZE", "0"))
    validate_request(request_id, profile, tile_size)
    if os.getenv("YOLO_BALL_MODEL_PATH") or os.getenv("ANALYSIS_FPS") or os.getenv("YOLO_IMAGE_SIZE"):
        raise ValueError("Modal uses its deployed weights and accurate/preview profile. Remove local FPS, image-size and specialist-weight overrides before using Modal.")
    if progress:
        progress("Uploading video to private Modal storage", 0.08)
    volume = modal.Volume.from_name(VOLUME_NAME)
    prefix = f"/requests/{request_id}"
    with volume.batch_upload() as batch:
        batch.put_file(str(input_video), f"{prefix}/source.mp4")
    imported = False
    try:
        if progress:
            progress("Queued or processing on Modal GPU · first run may build the image", 0.2)
        remote = modal.Function.from_name(APP_NAME, "analyze")
        response = remote.remote(request_id, profile, tile_size)
        run_id = response["run_id"]
        validate_request(run_id)
        output_root.mkdir(parents=True, exist_ok=True)
        destination = output_root / run_id
        if destination.exists():
            raise ValueError("Run directory already exists; refusing to overwrite it")
        if progress:
            progress("Downloading verified Modal results", 0.92)
        # A run becomes visible only once every artifact and its hash are verified.
        with tempfile.TemporaryDirectory(prefix="modal-import-", dir=output_root) as directory:
            staging = Path(directory) / run_id
            staging.mkdir()
            for name in ARTIFACTS:
                digest = hashlib.sha256()
                with (staging / name).open("wb") as stream:
                    for chunk in volume.read_file(f"{prefix}/outputs/{run_id}/{name}"):
                        digest.update(chunk)
                        stream.write(chunk)
                if digest.hexdigest() != response["sha256"][name]:
                    raise RuntimeError(f"Modal download checksum mismatch: {name}")
            manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
            with input_video.open("rb") as source:
                source_hash = hashlib.file_digest(source, "sha256").hexdigest()
            if manifest["run_id"] != run_id or manifest["metadata"]["sha256"] != source_hash:
                raise RuntimeError("Modal output does not match the submitted video")
            shutil.move(str(staging), str(destination))
        imported = True
        return manifest
    finally:
        # This request's staged cloud files only; imported outputs stay on the local server.
        # A process kill/timeout can leave files behind; see the cleanup instructions.
        if imported:
            try:
                volume.remove_file(prefix, recursive=True)
            except Exception:
                import logging
                logging.getLogger(__name__).warning("Modal staged files remain at %s; clean them from the volume", prefix)
        else:
            import logging
            logging.getLogger(__name__).warning("Modal request %s failed; files retained for recovery. Check the dashboard before cleanup.", request_id)
