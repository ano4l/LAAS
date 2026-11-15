"""Automatic per-frame pitch calibration from detected pitch landmarks.

A keypoint model marks known pitch landmarks (line junctions, penalty spots, circle
points) in a frame. A homography fitted to them maps image foot points onto the pitch
plane. Each frame is calibrated on its own: nothing is carried over, smoothed or
interpolated, and a frame that fails a check has no pitch positions at all.

The pitch size is not measured. Markings with fixed dimensions (penalty and goal areas,
centre circle, penalty spots) are placed by the Laws of the Game; touchline and goal-line
landmarks assume a 105 x 68 m pitch. Metres are therefore estimates, and accuracy has not
been evaluated against measured positions.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

VERSION = "pitch-keypoints-1"
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
LIMITATIONS = ["Each frame is calibrated on its own from landmarks a model found; nothing is smoothed, carried over or interpolated.",
               "The pitch is assumed to be 105 x 68 m. Fixed markings follow the Laws of the Game; a different pitch size shifts positions near the touchlines and goal lines.",
               "Fit error measures agreement among the landmarks used, not the accuracy of player positions. It has not been checked against measured positions.",
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
                    mean_reprojection_px=round(float(back.mean()), 2))
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

    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model, self.device, self.name = YOLO(str(weights)), device, getattr(weights, "name", str(weights))
        if getattr(self.model, "task", None) != "pose" or list(getattr(self.model.model, "kpt_shape", [])[:1]) != [len(landmarks())]:
            raise ValueError("The pitch model must be a pose model with 32 pitch landmarks")

    def calibrate(self, frame):
        height, width = frame.shape[:2]
        result = self.model.predict(frame, imgsz=640, conf=.3, device=self.device, verbose=False)[0]
        if result.keypoints is None or not len(result.keypoints) or result.keypoints.conf is None:
            return dict(status="invalid", reason="No pitch found in the frame", landmarks_seen=0)
        best = int(result.boxes.conf.argmax()) if len(result.boxes) > 1 else 0
        return fit(result.keypoints.xy[best].cpu().numpy(), result.keypoints.conf[best].cpu().numpy(), width, height)


def document(frames, model_name, model_sha256=None, stride=1):
    """The per-run calibration artifact. `frames`: {frame_idx: record}."""
    calibrated = [r for r in frames.values() if r["status"] == "calibrated"]
    reasons = {}
    for record in frames.values():
        if record["status"] != "calibrated":
            # Count "Only 3 ..." and "Only 4 ..." together.
            reason = ("Too few landmarks agree with one pitch fit" if "agree" in record["reason"] else "Too few pitch landmarks") if record["reason"].startswith("Only") else record["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
    return dict(schema_version=VERSION, method="per_frame_landmark_homography", review_status="unreviewed", accuracy_evaluated=False,
                model=model_name, model_sha256=model_sha256, frame_stride=stride,
                assumed_pitch=dict(length_m=LENGTH_M, width_m=WIDTH_M, measured=False),
                parameters=dict(min_landmark_confidence=MIN_CONFIDENCE, min_landmarks=MIN_LANDMARKS, ransac_threshold_m=RANSAC_M, min_landmark_agreement=MIN_AGREEMENT,
                                max_mean_error_m=MAX_MEAN_ERROR_M, max_error_m=MAX_ERROR_M, min_landmark_image_area=MIN_IMAGE_AREA,
                                min_landmark_span_m=MIN_SPAN_M, outside_pitch_margin_m=MARGIN_M),
                summary=dict(frames_tried=len(frames), frames_calibrated=len(calibrated),
                             mean_error_m=round(float(np.mean([r["mean_error_m"] for r in calibrated])), 3) if calibrated else None,
                             rejected=reasons),
                limitations=LIMITATIONS,
                frames={str(index): record for index, record in sorted(frames.items())})
