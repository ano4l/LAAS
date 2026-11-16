"""Automatic per-frame pitch calibration from detected pitch landmarks.

A keypoint model marks known pitch landmarks (line junctions, penalty spots, circle
points) in a frame. A homography fitted to them maps image foot points onto the pitch
plane. Each frame is calibrated on its own, and a frame that fails a check has no
pitch positions at all: nothing is carried into it or interpolated. pitch_camera.py then
steadies calibrated frames that have calibrated neighbours, by smoothing the mapping over
about half a second; a frame whose smoothed mapping would fail its landmark checks keeps
its own fit.

The landmark checks only test whether the landmarks agree with each other, and a model can
invent a consistent set. So calibrated frames are also checked against the image
(pitch_lines.py): the pitch's markings drawn through the mapping should land on white lines.
Nothing is rejected on this. A clip where most checked frames miss the lines carries a
warning that its pitch positions are likely several metres off.

The pitch size is not measured. Markings with fixed dimensions (penalty and goal areas,
centre circle, penalty spots) are placed by the Laws of the Game; touchline and goal-line
landmarks assume a 105 x 68 m pitch. Metres are therefore estimates. On labelled SoccerNet clips
they were a median 7 m off (docs/testing/SOCCERNET_GSR_2026-11-15.md).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

VERSION = "pitch-keypoints-4"
LENGTH_M, WIDTH_M = 105., 68.  # assumed, not measured
MIN_CONFIDENCE = .5
MIN_LANDMARKS = 5  # four fit exactly, so a fifth is the first that can disagree
RANSAC_M = 1.
MIN_AGREEMENT = .7  # share of the seen landmarks that must agree with the fit
MAX_MEAN_ERROR_M = .6
MAX_ERROR_M = 1.5
MIN_IMAGE_AREA = .015  # landmark hull as a share of the image
MIN_SPAN_M = 9.  # landmarks must spread along both pitch axes
MARGIN_M = 3.  # foot points this far outside the pitch outline are still drawn
# Checked against the image (pitch_lines.py) and reported, never used to reject. Set on ten labelled SoccerNet clips, where the calibration
# was a median 7 m off and at most 15% of a clip's frames reached this support, and a Bundesliga clip whose drawn lines sit on the real ones (median 0.93).
LINE_TOLERANCE = .015  # a drawn marking point is on a line within this share of the image height (16 px at 1080p)
MIN_LINE_SUPPORT = .75  # share of drawn marking points that must land on a white line
MIN_CLIP_LINE_SHARE = .5  # below this share of checked frames matching, the clip's pitch figures carry a warning
LINE_EVERY = 5  # in the worker, every Nth calibrated frame is checked: enough for the clip-level result at a fifth of the cost
LINES_WARNING = ("In most frames of this clip the pitch lines drawn from the calibration miss the lines in the video (only {share}% of checked "
                 "frames match), so pitch positions are likely several metres off. Treat the pitch map, heat maps, distances and speeds as rough.")
LIMITATIONS = ["Each frame is calibrated from landmarks a model found. A calibrated frame with calibrated neighbours has its mapping smoothed over about half a second, and keeps the smoothed mapping only if it still passes the landmark checks; otherwise it keeps its own. A frame without a calibration gets no positions: nothing is carried into it or interpolated.",
               "The pitch is assumed to be 105 x 68 m. Fixed markings follow the Laws of the Game; a different pitch size shifts positions near the touchlines and goal lines.",
               "Fit error measures agreement among the landmarks used, not the accuracy of player positions. On ten labelled SoccerNet test clips from another league, positions were off by a median of about 8 m, including frames whose landmarks agreed to under half a metre.",
               "Calibrated frames are checked against the video: the pitch's lines drawn through the mapping should land on white lines. A clip where most checked frames miss them is marked as likely several metres off; nothing is removed.",
               "Foot points are the bottom centre of a detection box; occlusion, jumps and lens distortion move them. Positions far from any landmark are less certain."]


def landmarks(length=LENGTH_M, width=WIDTH_M):
    """Pitch coordinates in metres of the model's 32 landmarks, in its output order.

    x runs along the pitch from the left goal line as the main camera sees it, y from
    the far touchline to the near one. Landmarks 11, 12, 19 and 20 are where the penalty
    arc meets the penalty-area line.
    """
    area_w, area_l, goal_w, goal_l, radius, spot = 40.32, 16.5, 18.32, 5.5, 9.15, 11.
    arc = 2 * math.sqrt(radius ** 2 - (area_l - spot) ** 2)
    mid, half = width / 2, length / 2
    left = [(0, 0), (0, mid - area_w / 2), (0, mid - goal_w / 2), (0, mid + goal_w / 2), (0, mid + area_w / 2), (0, width),
            (goal_l, mid - goal_w / 2), (goal_l, mid + goal_w / 2), (spot, mid),
            (area_l, mid - area_w / 2), (area_l, mid - arc / 2), (area_l, mid + arc / 2), (area_l, mid + area_w / 2)]
    centre = [(half, 0), (half, mid - radius), (half, mid + radius), (half, width)]
    right = [(length - area_l, mid - area_w / 2), (length - area_l, mid - arc / 2), (length - area_l, mid + arc / 2),
             (length - area_l, mid + area_w / 2), (length - spot, mid),
             (length - goal_l, mid - goal_w / 2), (length - goal_l, mid + goal_w / 2),
             (length, 0), (length, mid - area_w / 2), (length, mid - goal_w / 2), (length, mid + goal_w / 2),
             (length, mid + area_w / 2), (length, width)]
    return np.array(left + centre + right + [(half - radius, mid), (half + radius, mid)], dtype=float)


def _project(matrix, points):
    return cv2.perspectiveTransform(np.asarray(points, dtype=float).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def _orientation(matrix, points):
    """Sign of the mapping's local orientation at each image point, and whether it lies in front of the horizon."""
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    w = matrix[2, 0] * xy[:, 0] + matrix[2, 1] * xy[:, 1] + matrix[2, 2]
    return np.sign(np.linalg.det(matrix)) * np.sign(w), w


def fit(xy, confidence, width, height):
    """Fit image->pitch for one frame. Returns a record with `status` 'calibrated' or 'invalid' (with a reason)."""
    xy, confidence = np.asarray(xy, dtype=float).reshape(-1, 2), np.asarray(confidence, dtype=float).ravel()
    world = landmarks()
    if len(xy) != len(world) or len(confidence) != len(world):
        return dict(status="invalid", reason="Landmark model output has the wrong shape")
    keep = ((confidence >= MIN_CONFIDENCE) & np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height))
    seen = int(keep.sum())
    if seen < MIN_LANDMARKS:
        return dict(status="invalid", reason=f"Only {seen} pitch landmarks seen; {MIN_LANDMARKS} are needed", landmarks_seen=seen)
    image, pitch = xy[keep], world[keep]
    matrix, mask = cv2.findHomography(image, pitch, cv2.RANSAC, RANSAC_M)
    if matrix is None or mask is None:
        return dict(status="invalid", reason="No consistent pitch fit", landmarks_seen=seen)
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < MIN_LANDMARKS or inliers.mean() < MIN_AGREEMENT:
        return dict(status="invalid", reason=f"Only {int(inliers.sum())} of {seen} landmarks agree with one pitch fit", landmarks_seen=seen)
    image, pitch = image[inliers], pitch[inliers]
    numbers = np.flatnonzero(keep)[inliers] + 1
    matrix, _ = cv2.findHomography(image, pitch, 0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        return dict(status="invalid", reason="Degenerate pitch fit", landmarks_seen=seen)
    record = dict(landmarks_seen=seen, landmarks_used=int(inliers.sum()))
    error = np.linalg.norm(_project(matrix, image) - pitch, axis=1)
    record.update(mean_error_m=round(float(error.mean()), 3), max_error_m=round(float(error.max()), 3))
    span = pitch.max(axis=0) - pitch.min(axis=0)
    area = cv2.contourArea(cv2.convexHull(image.astype(np.float32))) / (width * height)
    orientation, w = _orientation(matrix, image)
    if span.min() < MIN_SPAN_M or area < MIN_IMAGE_AREA:
        reason = "Landmarks cover too little of the pitch to fix its geometry"
    elif error.mean() > MAX_MEAN_ERROR_M or error.max() > MAX_ERROR_M:
        reason = "Landmarks do not agree closely enough with one pitch fit"
    elif not (orientation > 0).all() or not (np.sign(w) == np.sign(w[0])).all():
        reason = "Pitch fit is mirrored or folds across the horizon"
    else:
        back = np.linalg.norm(_project(np.linalg.inv(matrix), pitch) - image, axis=1)
        # Scaled so the landmarks lie at positive depth; to_pitch relies on that sign.
        matrix = matrix * np.sign(w[0]) / np.linalg.norm(matrix)
        return dict(record, status="calibrated", matrix=[float(f"{v:.10g}") for v in matrix.ravel()],
                    mean_reprojection_px=round(float(back.mean()), 2),
                    # [landmark number, x px, y px] of the landmarks used, so later steps can be checked against them.
                    landmarks=[[int(n), round(float(x), 1), round(float(y), 1)] for n, (x, y) in zip(numbers, image)])
    return dict(record, status="invalid", reason=reason)


def to_pitch(record, points):
    """Project image points with a calibrated frame record. Returns (xy_m, usable) per point.

    A point is usable when it lies in front of the horizon, keeps the fit's orientation
    and lands within MARGIN_M of the assumed pitch outline.
    """
    matrix = np.asarray(record["matrix"], dtype=float).reshape(3, 3)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    orientation, w = _orientation(matrix, points)
    safe = (orientation > 0) & (w > 1e-12)  # same side of the horizon as the landmarks
    xy = np.full((len(points), 2), np.nan)
    if safe.any():
        xy[safe] = _project(matrix, points[safe])
    usable = safe & (xy[:, 0] >= -MARGIN_M) & (xy[:, 0] <= LENGTH_M + MARGIN_M) & (xy[:, 1] >= -MARGIN_M) & (xy[:, 1] <= WIDTH_M + MARGIN_M)
    return xy, usable


def visible_area(record, width, height, step=2.5):
    """Outline, in pitch metres, of the part of the assumed pitch that this frame's image covers."""
    matrix = np.asarray(record["matrix"], dtype=float).reshape(3, 3)
    inverse = np.linalg.inv(matrix)
    xs, ys = np.linspace(0, LENGTH_M, round(LENGTH_M / step) + 1), np.linspace(0, WIDTH_M, round(WIDTH_M / step) + 1)
    grid = np.array([(x, y) for x in xs for y in ys], dtype=float)
    w = inverse[2, 0] * grid[:, 0] + inverse[2, 1] * grid[:, 1] + inverse[2, 2]
    front = np.abs(w) > 1e-9
    image = np.full((len(grid), 2), np.nan)
    image[front] = _project(inverse, grid[front])
    inside = front & (image[:, 0] >= 0) & (image[:, 0] < width) & (image[:, 1] >= 0) & (image[:, 1] < height)
    # Keep only pitch points whose image position maps back to themselves (rejects the far side of the horizon).
    if inside.any():
        back, usable = to_pitch(record, image[inside])
        ok = usable & (np.linalg.norm(back - grid[inside], axis=1) < .05)
        inside[np.flatnonzero(inside)[~ok]] = False
    if inside.sum() < 3:
        return []
    hull = cv2.convexHull(grid[inside].astype(np.float32)).reshape(-1, 2)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in hull]


class PitchKeypointModel:
    """Thin wrapper around the pose model: one frame in, one calibration record out."""

    def __init__(self, weights, device, line_every=None):
        from ultralytics import YOLO
        exported = str(weights).lower().endswith(".onnx")  # same network, run by ONNX Runtime
        self.model, self.device, self.name = YOLO(str(weights), task="pose") if exported else YOLO(str(weights)), device, getattr(weights, "name", str(weights))
        self.line_every, self.calibrated = line_every or LINE_EVERY, 0
        network = self.model.model
        if exported:
            # An export carries its landmark layout in metadata that is only read when the runtime loads it.
            import numpy as np
            self.model.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=640, device=device, verbose=False)
            network = self.model.predictor.model
        if getattr(self.model, "task", None) != "pose" or list(getattr(network, "kpt_shape", [])[:1]) != [len(landmarks())]:
            raise ValueError("The pitch model must be a pose model with 32 pitch landmarks")

    def calibrate(self, frame):
        height, width = frame.shape[:2]
        result = self.model.predict(frame, imgsz=640, conf=.3, device=self.device, verbose=False)[0]
        if result.keypoints is None or not len(result.keypoints) or result.keypoints.conf is None:
            return dict(status="invalid", reason="No pitch found in the frame", landmarks_seen=0)
        best = int(result.boxes.conf.argmax()) if len(result.boxes) > 1 else 0
        record = fit(result.keypoints.xy[best].cpu().numpy(), result.keypoints.conf[best].cpu().numpy(), width, height)
        if record["status"] == "calibrated":
            self.calibrated += 1
            if (self.calibrated - 1) % self.line_every == 0:
                record = check_lines(record, frame)
        return record


def check_lines(record, frame):
    """Measure how well a calibrated record's mapping, drawn into `frame`, lands on the white lines. It never changes the status."""
    if record.get("status") != "calibrated":
        return record
    from workers.video import pitch_lines
    support = pitch_lines.support(frame, record["matrix"], tolerance=LINE_TOLERANCE)
    return dict(record, line_support=support["share"], line_markings=[support["markings_on_lines"], support["markings_seen"]])


def lines_summary(frames):
    """Clip-level result of the line check over the calibrated frames that were checked. Changes nothing."""
    out = dict(tolerance=LINE_TOLERANCE, min_line_support=MIN_LINE_SUPPORT, min_clip_share=MIN_CLIP_LINE_SHARE)
    checked = [record for record in frames.values() if record.get("status") == "calibrated" and "line_support" in record]
    if not checked:
        return dict(out, status="not_run", warning=None,
                    reason="This calibration was made before frames were checked against the video's lines.")
    matching = sum(record["line_support"] is not None and record["line_support"] >= MIN_LINE_SUPPORT for record in checked)
    share = matching / len(checked)
    mismatch = share < MIN_CLIP_LINE_SHARE
    return dict(out, status="lines_do_not_match" if mismatch else "lines_match", frames_checked=len(checked), frames_matching=matching,
                share_matching=round(share, 3), warning=LINES_WARNING.format(share=round(100 * share)) if mismatch else None)


def document(frames, model_name, model_sha256=None, stride=1, video=None):
    """The per-run calibration artifact. `frames`: {frame_idx: record}.

    With `video` = (width, height, fps), the calibrated records are first steadied in
    place by smoothing over time (see pitch_camera.py).
    """
    camera = None
    if video:
        from workers.video.pitch_camera import stabilise
        try:
            camera = stabilise(frames, *video)
        except Exception as error:  # steadying is an extra; a failure leaves the per-frame fits as they were
            for record in frames.values():
                if "frame_matrix" in record:
                    record["matrix"] = record.pop("frame_matrix")
                if record["status"] == "calibrated":
                    record["stabilised"] = False
            camera = dict(status="not_stabilised", reason=f"Steadying failed: {type(error).__name__}", frames_stabilised=0)
    calibrated = [r for r in frames.values() if r["status"] == "calibrated"]
    reasons = {}
    for record in frames.values():
        if record["status"] != "calibrated":
            # Count "Only 3 ..." and "Only 4 ..." together.
            reason = ("Too few landmarks agree with one pitch fit" if "agree" in record["reason"] else "Too few pitch landmarks") if record["reason"].startswith("Only") else record["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
    return dict(schema_version=VERSION, method="landmark_homography_smoothed" if camera and camera["frames_stabilised"] else "per_frame_landmark_homography", review_status="unreviewed", accuracy_evaluated=False,
                model=model_name, model_sha256=model_sha256, frame_stride=stride,
                assumed_pitch=dict(length_m=LENGTH_M, width_m=WIDTH_M, measured=False),
                parameters=dict(min_landmark_confidence=MIN_CONFIDENCE, min_landmarks=MIN_LANDMARKS, ransac_threshold_m=RANSAC_M, min_landmark_agreement=MIN_AGREEMENT,
                                max_mean_error_m=MAX_MEAN_ERROR_M, max_error_m=MAX_ERROR_M, min_landmark_image_area=MIN_IMAGE_AREA,
                                min_landmark_span_m=MIN_SPAN_M, outside_pitch_margin_m=MARGIN_M),
                summary=dict(frames_tried=len(frames), frames_calibrated=len(calibrated),
                             mean_error_m=round(float(np.mean([r["mean_error_m"] for r in calibrated])), 3) if calibrated else None,
                             rejected=reasons, steadying=camera, line_check=lines_summary(frames)),
                limitations=LIMITATIONS,
                frames={str(index): record for index, record in sorted(frames.items())})


def upgrade(doc, video):
    """Bring a stored calibration document up to this version by redoing the steadying step on its stored frames.

    No model is needed: the per-frame fits and their landmarks are already in the document.
    Returns whether the document changed.
    """
    if doc.get("schema_version") == VERSION or "frames" not in doc:
        return False
    frames = {int(index): record for index, record in doc["frames"].items()}
    new = document(frames, doc.get("model"), doc.get("model_sha256"), doc.get("frame_stride", 1), video)
    doc.clear()
    doc.update(new)
    return True


SUMMARY_KEYS = ("schema_version", "method", "model", "model_sha256", "frame_stride", "assumed_pitch", "summary")
CALIBRATION_WARNING = ("Pitch positions in the shape view are estimates from automatic per-frame calibration that assumes a 105 x 68 m pitch. "
                       "They are unreviewed, were off by a median of about 8 m on labelled test clips from another league, and are absent for frames that fail the calibration checks.")


def _write_json(path, value):
    import json
    import os
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream)
        temporary = stream.name
    try:
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def calibrate_run(video, run_directory, weights, device="cpu", stride=1, progress=None):
    """Add pitch calibration to an existing run without re-tracking.

    Calibrates the frames the run analysed, writes pitch-calibration.json and
    recomputes the insights that use it. Detections and tracks are untouched.
    Nothing is written unless the whole video decodes. Returns the manifest.
    """
    import hashlib
    import json
    from pathlib import Path

    video, run_directory, weights = Path(video), Path(run_directory), Path(weights)
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    metadata, summary = manifest["metadata"], manifest["detection_summary"]
    with video.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != metadata.get("sha256"):
            raise ValueError("This video is not the run's source: SHA-256 differs")
    frame_stride = max(1, round(metadata["fps"] / summary["sample_fps"]))
    wanted = set(range(0, summary["decoded_frames"], frame_stride * stride))
    model, frames, index = PitchKeypointModel(weights, device), {}, 0
    capture = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = model.calibrate(frame)
                if progress and len(frames) % 25 == 0:
                    progress(len(frames), len(wanted))
            index += 1
    finally:
        capture.release()
    if index < summary["decoded_frames"] - max(2, frame_stride):
        raise ValueError(f"Video decoding stopped at frame {index} of {summary['decoded_frames']}; nothing was written.")
    with weights.open("rb") as stream:
        calibration = document(frames, weights.name, hashlib.file_digest(stream, "sha256").hexdigest(), frame_stride * stride,
                               (metadata["width"], metadata["height"], metadata["fps"]))
    return _store(run_directory, manifest, calibration)


def _store(run_directory, manifest, calibration):
    """Write a calibration document and recompute the insights that use it."""
    import json
    from workers.video.insights import from_tracking
    from workers.video.sequence_context import load_context
    _write_json(run_directory / "pitch-calibration.json", calibration)
    summary = manifest["detection_summary"]
    summary["pitch_calibration"] = {key: calibration[key] for key in SUMMARY_KEYS}
    if CALIBRATION_WARNING not in manifest.setdefault("warnings", []):
        manifest["warnings"].append(CALIBRATION_WARNING)
    context = load_context(run_directory / "sequence-context.json", manifest)
    frames = {int(index): record for index, record in calibration["frames"].items()}
    manifest.setdefault("metrics", {})["video_insights"] = from_tracking(
        json.loads((run_directory / "tracking.json").read_text(encoding="utf-8")), manifest["metadata"], summary, context, frames)
    _write_json(run_directory / "metrics.json", manifest["metrics"])
    _write_json(run_directory / "manifest.json", manifest)
    return manifest


def check_run_lines(video, run_directory, every=None):
    """Check a stored run's calibration against its video's lines, without the landmark model, and store the result.

    Every `every`th calibrated frame (LINE_EVERY by default) is measured with the mapping it
    already has; nothing is rejected or refitted. The document is brought up to this version
    and the insights that use it are recomputed. Returns the line-check summary.
    """
    import hashlib
    import json
    from pathlib import Path
    video, run_directory = Path(video), Path(run_directory)
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    with video.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != metadata.get("sha256"):
            raise ValueError("This video is not the run's source: SHA-256 differs")
    doc = json.loads((run_directory / "pitch-calibration.json").read_text(encoding="utf-8"))
    frames = {int(index): record for index, record in doc["frames"].items()}
    wanted = set(sorted(index for index, record in frames.items() if record.get("status") == "calibrated")[::every or LINE_EVERY])
    capture, index = cv2.VideoCapture(str(video)), 0
    try:
        while wanted and index <= max(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = check_lines(frames[index], frame)
            index += 1
    finally:
        capture.release()
    calibration = document(frames, doc.get("model"), doc.get("model_sha256"), doc.get("frame_stride", 1),
                           (metadata["width"], metadata["height"], metadata["fps"]))
    _store(run_directory, manifest, calibration)
    return calibration["summary"]["line_check"]
