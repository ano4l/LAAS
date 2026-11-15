"""Steadier pitch mapping: smooth each frame's own calibration over a short time window.

A frame-by-frame homography follows the small errors of that frame's landmarks, so the
whole pitch map wobbles from frame to frame. The camera itself moves smoothly. This
module follows where four fixed image points land on the pitch through an unbroken
stretch of calibrated frames, fits a short straight line through each point's path
around every frame, and rebuilds that frame's mapping from the smoothed points.

A frame keeps the smoothed mapping only if it still passes the same landmark checks its
own fit had to pass (pitch_keypoints.MAX_MEAN_ERROR_M and MAX_ERROR_M). Otherwise it
keeps its own fit. Frames that were never calibrated stay uncalibrated: nothing is
carried into them. Smoothing never crosses a gap in calibration or a jump in the view.

An earlier version fixed one camera position for the whole clip and refitted turn and
zoom only. On real footage that model missed the landmarks by about 0.7 to 1.3 m (a free
fit: 0.2 to 0.3 m), even with the position refined and a lens-distortion term, so it was
dropped. Accuracy has not been evaluated against measured positions.
"""
from __future__ import annotations

import cv2
import numpy as np

from workers.video import pitch_keypoints as pk

SMOOTH_SIGMA_S = .15
SMOOTH_WINDOW_S = .45
MIN_NEIGHBOURS = 5  # calibrated frames in the window, the frame itself included
MAX_GAP_S = .5  # a longer gap between calibrated frames starts a new stretch
MAX_JUMP_M = 5.  # the anchors moving further than this between neighbouring samples, and faster than MAX_PAN_M_S, is a cut
MAX_PAN_M_S = 60.
MAX_DISAGREEMENT_M = .6  # used instead of the landmark checks when a frame's landmarks were not stored
ANCHORS = ((.2, .45), (.8, .45), (.2, .92), (.8, .92))  # image points, as shares of width and height


def _targets(record, matrix, width, height):
    """(pitch metres, image pixels, from_landmarks) the smoothed mapping is checked against."""
    if record.get("landmarks"):
        rows = np.asarray(record["landmarks"], dtype=float)
        return pk.landmarks()[rows[:, 0].astype(int) - 1], rows[:, 1:3], True
    grid = np.array([(x, y) for x in np.linspace(.08, .92, 7) * width for y in np.linspace(.2, .95, 5) * height])
    world, usable = pk.to_pitch(dict(matrix=matrix.ravel()), grid)
    return world[usable], grid[usable], False


def _jitter(frames, key, width, height, fps):
    """Mean second difference (m) of where one fixed image point lands on the pitch, over evenly spaced neighbours."""
    point, rows = np.array([[width / 2, height * .7]]), []
    for index in sorted(frames):
        record = frames[index]
        if record["status"] == "calibrated":
            xy, usable = pk.to_pitch(dict(matrix=record.get(key) or record["matrix"]), point)
            if usable[0]:
                rows.append((index, xy[0]))
    values = [np.linalg.norm(a[1] - 2 * b[1] + c[1]) for a, b, c in zip(rows, rows[1:], rows[2:])
              if b[0] - a[0] == c[0] - b[0] and (c[0] - a[0]) / fps <= 2 * MAX_GAP_S]
    return round(float(np.mean(values)), 3) if values else None


def stabilise(frames, width, height, fps):
    """Steady the calibrated records in `frames` ({frame_idx: record}) in place. Returns a summary.

    A steadied record keeps its own fit as `frame_matrix`, gets the smoothed mapping as
    `matrix`, and `stabilised` True. Every other calibrated record gets `stabilised` False.
    """
    parameters = dict(smooth_sigma_s=SMOOTH_SIGMA_S, smooth_window_s=SMOOTH_WINDOW_S, min_neighbours=MIN_NEIGHBOURS, max_gap_s=MAX_GAP_S,
                      max_jump_m=MAX_JUMP_M, max_pan_m_s=MAX_PAN_M_S, max_mean_error_m=pk.MAX_MEAN_ERROR_M, max_error_m=pk.MAX_ERROR_M,
                      max_disagreement_m=MAX_DISAGREEMENT_M, anchors=[list(a) for a in ANCHORS])
    anchors = np.array([(x * width, y * height) for x, y in ANCHORS])
    rows = []
    for index in sorted(frames):
        record = frames[index]
        if record["status"] != "calibrated":
            continue
        if "frame_matrix" in record:  # steadied before: start again from the frame's own fit
            record["matrix"] = record.pop("frame_matrix")
        record["stabilised"] = False
        for key in ("stable_error_m", "focal_px"):
            record.pop(key, None)
        matrix = np.asarray(record["matrix"], dtype=float).reshape(3, 3)
        orientation, w = pk._orientation(matrix, anchors)
        if not (orientation > 0).all() or not (w > 1e-12).all():
            continue  # an anchor lies beyond the horizon in this frame; it keeps its own fit and ends the stretch
        rows.append(dict(index=index, time=index / fps, matrix=matrix, points=pk._project(matrix, anchors)))
    calibrated = sum(record["status"] == "calibrated" for record in frames.values())
    stretches, current = [], []
    for row in rows:
        if current:
            before = current[-1]
            elapsed = row["time"] - before["time"]
            moved = float(np.linalg.norm(row["points"] - before["points"], axis=1).mean())
            if elapsed > MAX_GAP_S or (moved > MAX_JUMP_M and moved > MAX_PAN_M_S * elapsed):
                stretches.append(current)
                current = []
        current.append(row)
    if current:
        stretches.append(current)
    errors, stabilised = [], 0
    for stretch in stretches:
        times = np.array([row["time"] for row in stretch])
        values = np.array([row["points"].ravel() for row in stretch])
        for i, row in enumerate(stretch):
            near = np.abs(times - times[i]) <= SMOOTH_WINDOW_S
            if near.sum() < MIN_NEIGHBOURS:
                continue
            dt = times[near] - times[i]
            weight = np.sqrt(np.exp(-.5 * (dt / SMOOTH_SIGMA_S) ** 2))
            design = np.column_stack([np.ones(near.sum()), dt]) * weight[:, None]
            smooth = np.linalg.lstsq(design, values[near] * weight[:, None], rcond=None)[0][0].reshape(4, 2)  # local line, read at this frame's time
            matrix = cv2.getPerspectiveTransform(anchors.astype(np.float32), smooth.astype(np.float32)).astype(float)
            record = frames[row["index"]]
            world, image, measured = _targets(record, row["matrix"], width, height)
            if len(world) < 4 or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                continue
            orientation, w = pk._orientation(matrix, image)
            if not (orientation > 0).all() or not (np.sign(w) == np.sign(w[0])).all():
                continue
            error = np.linalg.norm(pk._project(matrix, image) - world, axis=1)
            if (error.mean() > pk.MAX_MEAN_ERROR_M or error.max() > pk.MAX_ERROR_M) if measured else error.mean() > MAX_DISAGREEMENT_M:
                continue  # smoothing would break this frame's landmark checks; it keeps its own fit
            matrix = matrix * np.sign(w[0]) / np.linalg.norm(matrix)
            record.update(frame_matrix=record["matrix"], matrix=[float(f"{v:.10g}") for v in matrix.ravel()], stabilised=True)
            if measured:
                record["stable_error_m"] = round(float(error.mean()), 3)
                errors.append(float(error.mean()))
            stabilised += 1
    summary = dict(status="stabilised" if stabilised else "not_stabilised", frames_calibrated=calibrated, frames_stabilised=stabilised,
                   frames_kept_per_frame=calibrated - stabilised, stretches=len(stretches),
                   mean_stable_error_m=round(float(np.mean(errors)), 3) if errors else None,
                   jitter_before_m=_jitter(frames, "frame_matrix", width, height, fps), jitter_after_m=_jitter(frames, "matrix", width, height, fps),
                   parameters=parameters)
    if not stabilised:
        summary["reason"] = "No calibrated frame had enough calibrated neighbours to smooth against, or none kept its landmark checks"
    return summary
