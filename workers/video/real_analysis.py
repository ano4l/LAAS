"""Streaming detection and tracking without synthetic football statistics."""
from __future__ import annotations

import json
import hashlib
from importlib.metadata import version
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

import cv2

from workers.video.yolo_adapter import model_path, person_model_path, pitch_model_path, yolo_capabilities
from workers.video.detector import FootballDetector
from workers.video.accuracy import analysis_settings
from workers.video import insights as video_insights, measurements as video_measurements
from workers.video.ball_path import select_ball_path
from workers.video.camera_motion import CameraMotion
from workers.video.track_linking import AppearanceBank, PEOPLE, apply_links, link_tracks, write_links, write_rows

ROOT = Path(__file__).resolve().parents[2]


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or frame is None or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Video cannot be decoded or has invalid timing. Export an MP4 with H.264 video and try again.")
        height, width = frame.shape[:2]
        return dict(filename=path.name, fps=float(fps), frame_count=count,
                    duration_seconds=count / fps, width=width, height=height)
    finally:
        capture.release()


def run_real_analysis(input_video: Path, output_root: Path, progress=None) -> dict:
    caps = yolo_capabilities(ROOT)
    if not caps["available"]:
        raise RuntimeError(" ".join(caps["blockers"]))
    import imageio_ffmpeg
    import torch

    metadata = video_metadata(input_video)
    with input_video.open("rb") as source:
        metadata["sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    started = time.monotonic()
    if progress:
        progress("Loading detector and ByteTrack", 0.03)
    device = os.getenv("YOLO_DEVICE", "0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(max(1, int(os.getenv("YOLO_CPU_THREADS", "4"))))
    options = analysis_settings(metadata["fps"])
    stride, image_size = options["stride"], options["image_size"]
    ball_weights = Path(os.environ["YOLO_BALL_MODEL_PATH"]).expanduser().resolve() if os.getenv("YOLO_BALL_MODEL_PATH") else None
    rescue_weights = Path(os.environ['YOLO_BALL_RESCUE_PATH']).expanduser().resolve() if os.getenv('YOLO_BALL_RESCUE_PATH') else None
    ball_confidence = float(os.getenv("BALL_CONFIDENCE", "0.1"))
    object_confidence = float(os.getenv("OBJECT_CONFIDENCE", "0.45"))
    if not 0 < ball_confidence <= 1 or not 0 < object_confidence <= 1:
        raise ValueError("Detection confidence thresholds must be greater than 0 and at most 1.")
    person_weights = person_model_path(ROOT)
    detector = FootballDetector(model_path(ROOT), device=device, image_size=image_size,
                                ball_confidence=ball_confidence, object_confidence=object_confidence,
                                sample_fps=metadata["fps"] / stride, ball_weights=ball_weights,
                                tile_size=options["tile_size"],ball_rescue_weights=rescue_weights,
                                person_weights=person_weights)
    pitch_weights = pitch_model_path(ROOT)
    pitch_model = None
    if pitch_weights:
        from workers.video.pitch_keypoints import PitchKeypointModel
        pitch_model = PitchKeypointModel(pitch_weights, device)
    # Every analysed frame on a GPU; about five per second on a CPU, where the landmark model is slow.
    pitch_stride = max(1, int(os.getenv("PITCH_STRIDE", "1" if device != "cpu" else str(round(metadata["fps"] / stride / 5)))))
    pitch_frames = {}
    run_id = uuid4().hex
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True)
    width, height = metadata["width"], metadata["height"]
    scale = min(1, 1280 / max(width, height))
    out_w, out_h = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
    capture = cv2.VideoCapture(str(input_video))
    counts, track_counts = Counter(), Counter()
    track_classes, first_seen, last_seen = {}, {}, {}
    preview, analyzed, decoded, ball_frames = [], 0, 0, 0
    preview_stride = max(1, math.ceil(metadata["frame_count"] / stride / 70))
    embedder = None
    if os.getenv("TRACK_EMBEDDINGS", "off") == "detector":  # opt-in: small measured gain, extra compute
        from workers.video.reid import BackboneEmbedder
        embedder = BackboneEmbedder(model_path(ROOT), device=device)
    appearance, link_rows = AppearanceBank(embedder=embedder), []
    refiner, warps, warp_scenes = getattr(detector, "refiner", None), {}, {}
    last_rows = []
    try:
        with (out_dir / "encoder.log").open("wb") as log, (out_dir / "tracking.json").open("w", encoding="utf-8") as tracking_file:
            encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo",
                "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
                "-r", str(metadata["fps"]), "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(out_dir / "annotated.mp4")], stdin=subprocess.PIPE, stderr=log, stdout=subprocess.DEVNULL)
            tracking_file.write("[\n")
            first = True
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_idx = decoded
                    decoded += 1
                    # Unsampled frames have no observation; never paint a stale position as current.
                    last_rows = []
                    if frame_idx % stride == 0:
                        last_rows = []
                        detections = detector.detect(frame)
                        if getattr(refiner, "camera_warp", None) is not None:
                            warps[frame_idx], warp_scenes[frame_idx] = refiner.camera_warp, refiner.scene_id
                        for detection in detections:
                            label, identity = detection["object_type"], detection["track_id"]
                            row = dict(frame_idx=frame_idx, timestamp_ms=round(frame_idx / metadata["fps"] * 1000), **detection)
                            row.update(team=detection.get('team'), player_id=None, coordinate_space="image_pixels", review_status="unreviewed")
                            last_rows.append(row)
                            counts[label] += 1
                            if identity:
                                track_counts[identity] += 1
                                track_classes[identity] = label
                                first_seen.setdefault(identity, row["timestamp_ms"])
                                last_seen[identity] = row["timestamp_ms"]
                            if not first:
                                tracking_file.write(",\n")
                            tracking_file.write(json.dumps(row))
                            first = False
                        appearance.observe(frame, last_rows)  # before overlays are painted on the frame
                        if pitch_model and analyzed % pitch_stride == 0:
                            pitch_frames[frame_idx] = pitch_model.calibrate(frame)
                        link_rows.extend({k: r.get(k) for k in ("frame_idx", "track_id", "object_type", "bbox", "team", "scene_id")}
                                         for r in last_rows if r.get("track_id") and r["object_type"] in PEOPLE)
                        ball_frames += int(any(r["object_type"] in {"ball", "sports ball"} for r in last_rows))
                        if analyzed % preview_stride == 0:
                            preview.extend(last_rows)
                        analyzed += 1
                        if progress and (analyzed % 10 == 0 or analyzed == 1):
                            progress(f"Tracking · {frame_idx / metadata['fps']:.0f}s / {metadata['duration_seconds']:.0f}s · {device}",
                                     min(0.9, 0.05 + 0.85 * decoded / metadata["frame_count"]))
                    for row in last_rows:
                        x1, y1, x2, y2 = map(int, row["bbox"])
                        colour = (255,190,70) if row.get('team')=='kit-A' else (70,155,255) if row.get('team')=='kit-B' else (120,240,190)
                        if row['object_type'] in {'ball','sports ball'}:
                            colour = (30,230,255) if row.get('ball_selected') else (160,160,160)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                        label=f"{row['object_type']} {row['track_id'] or '?'} {row.get('team') or ''} {row['confidence']:.2f}"
                        cv2.putText(frame,label,(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,colour,1)
                    if (out_w, out_h) != (width, height):
                        frame = cv2.resize(frame, (out_w, out_h))
                    encoder.stdin.write(frame.tobytes())
                tracking_file.write("\n]")
            finally:
                try:
                    encoder.stdin.close()
                except OSError:
                    pass  # Always reap ffmpeg, including after a broken input pipe.
                if progress:
                    progress("Finalizing browser video", 0.94)
                try:
                    exit_code = encoder.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
                    raise RuntimeError("Video encoding timed out; see encoder.log.")
            if exit_code:
                raise RuntimeError("Video encoding failed; see the run's encoder.log.")
        if decoded < metadata["frame_count"] - max(2, stride):
            raise RuntimeError(f"Video decoding stopped at frame {decoded} of {metadata['frame_count']}. Re-export the source video.")
        if progress:
            progress("Linking fragmented tracks", 0.96)
        # Image-based camera motion is used for the ball path. Track linking keeps its own
        # estimate from player movement, which measured better for joining player tracks.
        # With a general person detector the football weights name a referee or goalkeeper in only some frames of a track,
        # so each track is given one role from all its frames before tracks are linked into identities.
        from workers.video import off_pitch
        from workers.video.people_boxes import PERSON_KINDS, apply_roles, settle_roles
        roles, role_changes, settled = {}, 0, {}
        if getattr(detector, "person_model", None):
            roles = settled = settle_roles(link_rows, detector.default_role)
        # The pitch calibration is steadied over the clip here, before linking, so that people who stand off the pitch
        # throughout (coaches, substitutes, stewards) can be set aside and never become player identities.
        pitch_document, bystanders = None, {}
        if pitch_model:
            from workers.video.pitch_keypoints import document
            with pitch_weights.open("rb") as weights:
                pitch_document = document(pitch_frames, pitch_weights.name, hashlib.file_digest(weights, "sha256").hexdigest(), stride * pitch_stride,
                                          (metadata["width"], metadata["height"], metadata["fps"]))
            (out_dir / "pitch-calibration.json").write_text(json.dumps(pitch_document), encoding="utf-8")
            bystanders = off_pitch.settle(link_rows, pitch_frames, keep={track for track, role in settled.items() if role == "referee"})
            roles = {**roles, **bystanders}
        if roles:
            apply_roles(link_rows, roles)
            apply_roles(preview, roles)
            track_classes.update({track: role for track, role in roles.items() if track in track_classes})
        camera = CameraMotion.from_warps(warps, warp_scenes) if warps else CameraMotion.from_people(link_rows)
        (out_dir / "camera-motion.json").write_text(json.dumps(dict(source=camera.source, warps=warps, scenes=warp_scenes)), encoding="utf-8")
        linking = link_tracks(link_rows, appearance.descriptors(), metadata["fps"], embeddings=appearance.embeddings())
        linking["summary"]["embedding_model"] = embedder.name if embedder else None
        apply_links(out_dir / "tracking.json", linking)
        write_links(out_dir, linking)
        # Analytics are computed after linking so they follow linked identities, not raw tracker IDs.
        linked_rows = json.loads((out_dir / "tracking.json").read_text(encoding="utf-8"))
        if roles:
            role_changes = apply_roles(linked_rows, roles)
            ball_kinds = {kind: n for kind, n in counts.items() if kind not in PERSON_KINDS}
            counts = Counter(row["object_type"] for row in linked_rows if row["object_type"] in PERSON_KINDS | {off_pitch.ROLE})
            counts.update(ball_kinds)
        # Choose the ball path over the whole clip, then store the rows analytics will read.
        ball_path = select_ball_path(linked_rows, metadata, camera=camera, pitch=pitch_frames if pitch_model else None)
        write_rows(out_dir / "tracking.json", linked_rows)
        scope = dict(sample_fps=metadata["fps"] / stride, decoded_frames=decoded)
        warnings = ["Detections and track IDs are unreviewed model predictions, not verified player identities.",
                    "Jersey groups are anonymous colour estimates, not verified teams. Passes, tackles, shots, goals, possession and xG are not verified.",
                    "Coordinates are image pixels. Distance, speed and pitch heatmaps require camera calibration.",
                    "Camera cuts and occlusion can split or switch track IDs. Output video has no audio.",
                    "Timestamps use the reported frame rate. Normalize variable-frame-rate footage before precise temporal review."]
        warnings.append("linked_track_id joins track fragments offline using timing, image position and kit colour. Links are unreviewed and can join different players; track_id keeps the original tracker output. Measurements and insights use the linked identities; the annotated video shows original tracker IDs.")
        warnings.append("Ball candidates pass geometry filtering; ball_selected marks the most consistent observed path chosen over the whole clip; it is provisional, not a confirmed trajectory. Missing positions remain missing; none are interpolated.")
        if pitch_model:
            warnings.append("Pitch positions in the shape view are estimates from automatic per-frame calibration that assumes a 105 x 68 m pitch. They are unreviewed, were off by a median of about 8 m on labelled test clips from another league, and are absent for frames that fail the calibration checks.")
        if stride > 1:
            warnings.append("Only sampled frames have detection overlays. Unsampled frames are not analysed or interpolated.")
        if options["tile_size"]:
            warnings.append("Experimental tiled ball search is enabled. Extra candidates may include false positives; validate against reviewed labels.")
        if "player" not in {str(name).lower() for name in detector.names.values()}:
            warnings.append("These weights detect general people, including officials and spectators; they do not distinguish football players or teams.")
        if not counts:
            warnings.append("No supported objects detected. Check the footage and model class labels.")
        summary = dict(analyzed_frames=analyzed, decoded_frames=decoded, sample_fps=metadata["fps"] / stride,
                       detections_by_class=dict(counts), unique_tracks=len(track_counts), ball_detected_frames=ball_frames,
                       processing_seconds=round(time.monotonic() - started, 2), device=device,
                       model=model_path(ROOT).name, model_classes=detector.names,
                       image_size=image_size, ball_confidence=ball_confidence, object_confidence=object_confidence,
                       analysis_profile=options["profile"], ball_tile_size=options["tile_size"],
                       tracks=[dict(track_id=key, object_type=track_classes[key], observations=n,
                                    first_seen_ms=first_seen[key], last_seen_ms=last_seen[key])
                               for key, n in track_counts.most_common()])
        with model_path(ROOT).open("rb") as weights:
            summary["model_sha256"] = hashlib.file_digest(weights, "sha256").hexdigest()
        summary["ultralytics_version"] = version("ultralytics")
        summary['tracking_config']=getattr(detector,'tracking_config',{})
        summary["tracker"] = summary['tracking_config'].get('tracker_type','bytetrack')+'.yaml'
        summary["ball_processing"] = "raw_detection_independent_of_person_tracker"
        summary["quality_filters"] = getattr(detector, "quality_counts", {})
        summary["quality_version"] = "research-association-2"
        summary["track_linking"] = linking["summary"]
        summary["ball_path"] = ball_path
        summary["pitch_calibration"] = ({key: pitch_document[key] for key in ("schema_version", "method", "model", "model_sha256", "frame_stride", "assumed_pitch", "summary")}
                                        if pitch_document else dict(status="not_run", reason="No pitch landmark model is configured."))
        summary["linked_tracks"] = linking["identities"]
        if person_weights:
            with person_weights.open('rb') as stream:
                summary['person_model_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
            summary['person_model']=person_weights.name
            summary['person_detection']='People come from a general person detector; the football weights supply the ball and the goalkeeper and referee labels. Each track has one role, settled from all its frames; role_hint keeps what each frame was called.'
            summary['role_settling']=dict(tracks=len(settled),referee_tracks=sum(role=='referee' for role in settled.values()),
                                          goalkeeper_tracks=sum(role=='goalkeeper' for role in settled.values()),rows_changed=role_changes)
        summary['off_pitch']=(dict(status='applied',tracks=len(bystanders),observations=counts.get(off_pitch.ROLE,0),
                                   rule=dict(margin_m=off_pitch.MARGIN_M,min_frames=off_pitch.MIN_FRAMES,min_share=off_pitch.MIN_SHARE),
                                   note='Tracks whose foot point stays outside the pitch outline are set aside as off_pitch and are not players. It relies on the pitch calibration.')
                              if pitch_document else dict(status='not_run',reason='No pitch calibration, so people beside the pitch are not told apart from players.'))
        if rescue_weights:
            with rescue_weights.open('rb') as stream:
                summary['secondary_ball_model_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
            summary['secondary_ball_model']=rescue_weights.name
            summary['secondary_ball_sampling']='Global fallback up to 2fps; recent-ball ROI up to 8fps; primary ball confidence below 0.3; image size 640'
        if ball_weights:
            with ball_weights.open("rb") as weights:
                summary["ball_model_sha256"] = hashlib.file_digest(weights, "sha256").hexdigest()
            summary["ball_model"] = ball_weights.name
        metrics = dict(possession=dict(home_pct=None, away_pct=None, timeline=[]), teams={}, players=[], events=[],
                       data_quality=dict(mode="yolo_bytetrack"), phase_count=0,
                       video_measurements=video_measurements.from_tracking(linked_rows, metadata, scope),
                       video_insights=video_insights.from_tracking(linked_rows, metadata, scope, pitch=pitch_frames if pitch_model else None))
        report = dict(title="Video detection and tracking", summary=f"Analyzed {analyzed} frames across {decoded / metadata['fps']:.1f} seconds. Found {len(track_counts)} temporary object tracks.", recommendations=[])
        manifest = dict(schema_version="1.0", run_id=run_id, engine="yolo_bytetrack", metadata=metadata,
                        tracking=preview, detection_summary=summary, warnings=warnings, metrics=metrics, phases=[], report=report,
                        files={name: f"/media/outputs/{run_id}/{filename}" for name, filename in
                               dict(annotated_video="annotated.mp4", tracking_json="tracking.json", metrics_json="metrics.json",
                                    report_json="report.json", manifest_json="manifest.json").items()})
        for name, value in [("metrics", metrics), ("report", report), ("manifest", manifest)]:
            (out_dir / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
        return manifest
    finally:
        capture.release()
