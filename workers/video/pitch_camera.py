"""One camera for the whole clip: steadier pitch mapping than fitting every frame alone.

A frame-by-frame homography has eight free numbers, so small landmark errors move the
whole pitch map from frame to frame. A broadcast camera has far fewer: it stands in one
place and only turns and zooms. This module

1. reads a camera position and focal length out of every calibrated frame's own fit,
2. takes the median position as the clip's camera, if the frames agree on one,
3. refits each frame with that position fixed (rotation and zoom only),
4. smooths rotation and zoom over a short time window, within unbroken stretches only,
5. keeps the result for a frame only if it still agrees with that frame's landmarks.

A frame that fails step 5, and every frame of a clip that fails step 2, keeps its own
per-frame fit. Frames that were never calibrated stay uncalibrated: nothing is carried
into them. Assumes a pinhole camera with square pixels, the principal point at the image
centre and no lens distortion. Accuracy has not been evaluated against measured positions.
"""
from __future__ import annotations

import cv2
import numpy as np

from workers.video import pitch_keypoints as pk

MIN_FRAMES = 8  # calibrated frames needed before a clip camera is trusted
MAX_POSITION_SPREAD = .2  # median distance of per-frame camera positions from their median, as a share of the camera's distance to the pitch centre
SMOOTH_SIGMA_S = .15
SMOOTH_WINDOW_S = .45
MAX_GAP_S = .5  # a longer gap between calibrated frames starts a new stretch
MAX_STEP_DEG = 4.  # a larger turn between neighbouring samples is treated as a cut
MAX_STEP_ZOOM = .15  # likewise for a larger change in log focal length
MAX_MEAN_ERROR_M = 1.  # landmark agreement the steadied mapping must keep
MAX_ERROR_M = 2.5
MAX_DISAGREEMENT_M = 1.5  # used instead when a frame's landmarks were not stored


def _centre(width, height):
    return np.array([[1., 0., -width / 2], [0., 1., -height / 2], [0., 0., 1.]])


def _targets(record, matrix, width, height):
    """(pitch metres, image pixels, from_landmarks) the steadied mapping is fitted to and checked against."""
    if record.get("landmarks"):
        rows = np.asarray(record["landmarks"], dtype=float)
        return pk.landmarks()[rows[:, 0].astype(int) - 1], rows[:, 1:3], True
    grid = np.array([(x, y) for x in np.linspace(.08, .92, 7) * width for y in np.linspace(.2, .95, 5) * height])
    world, usable = pk.to_pitch(dict(matrix=matrix.ravel()), grid)
    return world[usable], grid[usable], False


def decompose(matrix, width, height, world_point):
    """Focal length (px), rotation and camera position (pitch metres) implied by one image->pitch homography, or None."""
    g = _centre(width, height) @ np.linalg.inv(matrix)
    g1, g2, g3 = g[:, 0], g[:, 1], g[:, 2]
    # The two pitch axes are perpendicular and equally scaled; both conditions are linear in 1/f^2.
    a = np.array([g1[0] * g2[0] + g1[1] * g2[1], (g1[0] ** 2 + g1[1] ** 2) - (g2[0] ** 2 + g2[1] ** 2)])
    b = np.array([-g1[2] * g2[2], g2[2] ** 2 - g1[2] ** 2])
    if not a @ a > 0:
        return None
    inverse_f2 = (a @ b) / (a @ a)
    if not np.isfinite(inverse_f2) or inverse_f2 <= 0:
        return None
    f = 1 / np.sqrt(inverse_f2)
    if not .2 * width < f < 40 * width:
        return None
    r1, r2, t = (np.array([v[0] / f, v[1] / f, v[2]]) for v in (g1, g2, g3))
    scale = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    r1, r2, t = r1 / scale, r2 / scale, t / scale
    if r1[2] * world_point[0] + r2[2] * world_point[1] + t[2] < 0:  # the pitch must lie in front of the camera
        r1, r2, t = -r1, -r2, -t
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        return None
    return float(f), rotation, -rotation.T @ t


def homography(f, rotation, position, width, height):
    """Pitch -> image for a camera at `position`."""
    k = np.array([[f, 0., width / 2], [0., f, height / 2], [0., 0., 1.]])
    return k @ np.column_stack([rotation[:, 0], rotation[:, 1], -rotation @ position])


def _rotation(vector):
    return cv2.Rodrigues(np.asarray(vector, dtype=float).reshape(3, 1))[0]


def _vector(rotation):
    return cv2.Rodrigues(rotation)[0].ravel()


def _refit(f, rotation, position, world, image, width, height):
    """Rotation and zoom that best explain the targets from a fixed camera position."""
    def residual(p):
        g = homography(np.exp(p[3]), _rotation(p[:3]) @ rotation, position, width, height)
        return (pk._project(g, world) - image).ravel()

    # Levenberg-Marquardt on four numbers, with a numerical Jacobian. Small enough not to need a solver library.
    p, damping = np.array([0., 0., 0., np.log(f)]), 1e-3
    r = residual(p)
    for _ in range(40):
        jacobian = np.column_stack([(residual(p + step) - r) / 1e-6 for step in np.eye(4) * 1e-6])
        normal = jacobian.T @ jacobian
        try:
            delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -jacobian.T @ r)
        except np.linalg.LinAlgError:
            break
        candidate = residual(p + delta)
        if np.isfinite(candidate).all() and candidate @ candidate < r @ r:
            p, r, damping = p + delta, candidate, damping / 3
            if np.abs(delta).max() < 1e-9:
                break
        else:
            damping *= 5
            if damping > 1e8:
                break
    return float(np.exp(p[3])), _rotation(p[:3]) @ rotation


def _image_to_pitch(f, rotation, position, width, height, image):
    """The image->pitch matrix in the sign convention pitch_keypoints.to_pitch relies on, or None."""
    g = homography(f, rotation, position, width, height)
    if not np.isfinite(g).all() or abs(np.linalg.det(g)) < 1e-12:
        return None
    matrix = np.linalg.inv(g)
    orientation, w = pk._orientation(matrix, image)
    if not (orientation > 0).all() or not (np.sign(w) == np.sign(w[0])).all():
        return None
    return matrix * np.sign(w[0]) / np.linalg.norm(matrix)


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

    A steadied record keeps its own fit as `frame_matrix`, gets the clip-camera mapping as
    `matrix`, and `stabilised` True. Every other calibrated record gets `stabilised` False.
    """
    parameters = dict(min_frames=MIN_FRAMES, max_position_spread=MAX_POSITION_SPREAD, smooth_sigma_s=SMOOTH_SIGMA_S,
                      smooth_window_s=SMOOTH_WINDOW_S, max_gap_s=MAX_GAP_S, max_step_deg=MAX_STEP_DEG, max_step_zoom=MAX_STEP_ZOOM,
                      max_mean_error_m=MAX_MEAN_ERROR_M, max_error_m=MAX_ERROR_M, max_disagreement_m=MAX_DISAGREEMENT_M)
    rows = []
    for index in sorted(frames):
        record = frames[index]
        if record["status"] != "calibrated":
            continue
        if "frame_matrix" in record:  # steadied before: start again from the frame's own fit
            record["matrix"] = record.pop("frame_matrix")
        record["stabilised"] = False
        record.pop("stable_error_m", None)
        matrix = np.asarray(record["matrix"], dtype=float).reshape(3, 3)
        world, image, measured = _targets(record, matrix, width, height)
        if len(world) < 4:
            continue
        camera = decompose(matrix, width, height, world.mean(axis=0))
        if camera:
            rows.append(dict(index=index, matrix=matrix, world=world, image=image, measured=measured, f=camera[0], rotation=camera[1], position=camera[2]))
    calibrated = sum(record["status"] == "calibrated" for record in frames.values())

    def refused(reason, **extra):
        return dict(status="not_stabilised", reason=reason, frames_calibrated=calibrated, frames_stabilised=0, parameters=parameters, **extra)

    if len(rows) < MIN_FRAMES:
        return refused(f"Only {len(rows)} calibrated frames give a camera position; {MIN_FRAMES} are needed")
    positions = np.array([row["position"] for row in rows])
    position = np.median(positions, axis=0)
    distance = float(np.linalg.norm(position - np.array([pk.LENGTH_M / 2, pk.WIDTH_M / 2, 0.])))
    spread = float(np.median(np.linalg.norm(positions - position, axis=1)) / max(distance, 1e-9))
    if spread > MAX_POSITION_SPREAD:
        return refused("Frames do not agree on one camera position; the clip may mix cameras", position_spread=round(spread, 3))

    for row in rows:
        row["f"], row["rotation"] = _refit(row["f"], row["rotation"], position, row["world"], row["image"], width, height)
    # Unbroken stretches: no long gap, no sudden turn or zoom between neighbouring samples.
    stretches, current = [], [rows[0]]
    for before, row in zip(rows, rows[1:]):
        turn = np.degrees(np.linalg.norm(_vector(row["rotation"] @ before["rotation"].T)))
        if (row["index"] - before["index"]) / fps > MAX_GAP_S or turn > MAX_STEP_DEG or abs(np.log(row["f"] / before["f"])) > MAX_STEP_ZOOM:
            stretches.append(current)
            current = []
        current.append(row)
    stretches.append(current)
    for stretch in stretches:
        reference = stretch[0]["rotation"]
        times = np.array([row["index"] / fps for row in stretch])
        values = np.array([[*_vector(row["rotation"] @ reference.T), np.log(row["f"])] for row in stretch])
        for i, row in enumerate(stretch):
            near = np.abs(times - times[i]) <= SMOOTH_WINDOW_S
            if near.sum() < 3:
                continue  # too few neighbours to smooth against; the fixed-position fit stands
            dt = times[near] - times[i]
            weight = np.sqrt(np.exp(-.5 * (dt / SMOOTH_SIGMA_S) ** 2))
            design = np.column_stack([np.ones(near.sum()), dt]) * weight[:, None]
            value = np.linalg.lstsq(design, values[near] * weight[:, None], rcond=None)[0][0]  # local line, read at this frame's time
            row["smooth"] = (float(np.exp(value[3])), _rotation(value[:3]) @ reference)
    errors, stabilised = [], 0
    for row in rows:
        f, rotation = row.get("smooth", (row["f"], row["rotation"]))
        matrix = _image_to_pitch(f, rotation, position, width, height, row["image"])
        if matrix is None:
            continue
        error = np.linalg.norm(pk._project(matrix, row["image"]) - row["world"], axis=1)
        if (error.mean() > MAX_MEAN_ERROR_M or error.max() > MAX_ERROR_M) if row["measured"] else error.mean() > MAX_DISAGREEMENT_M:
            continue  # the clip camera does not explain this frame; it keeps its own fit
        record = frames[row["index"]]
        record.update(frame_matrix=record["matrix"], matrix=[float(f"{v:.10g}") for v in matrix.ravel()], stabilised=True,
                      focal_px=round(float(f), 1))
        if row["measured"]:
            record["stable_error_m"] = round(float(error.mean()), 3)
            errors.append(float(error.mean()))
        stabilised += 1
    return dict(status="stabilised" if stabilised else "not_stabilised", frames_calibrated=calibrated, frames_stabilised=stabilised,
                frames_kept_per_frame=calibrated - stabilised, stretches=len(stretches),
                camera_position_m=[round(float(v), 1) for v in position], position_spread=round(spread, 3),
                mean_stable_error_m=round(float(np.mean(errors)), 3) if errors else None,
                jitter_before_m=_jitter(frames, "frame_matrix", width, height, fps), jitter_after_m=_jitter(frames, "matrix", width, height, fps),
                parameters=parameters, **({} if stabilised else dict(reason="The clip camera did not agree with any frame's landmarks")))
