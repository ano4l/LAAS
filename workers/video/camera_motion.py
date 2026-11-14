"""Per-frame camera motion, used to compare image positions across frames.

Positions in a panning or zooming shot are only comparable once camera movement is
removed. The preferred source is an affine transform between consecutive frames
estimated from background image features (the same global motion compensation the
tracker uses). When that is unavailable, the median movement of tracked people is a
weak fallback: players tend to run with the play, so it underestimates pans.

Stabilised coordinates are relative to the first frame of each scene and drift over
long spans, so only compare positions that are a few seconds apart.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

import numpy as np

PEOPLE = {"person", "player", "goalkeeper", "referee"}


class CameraMotion:
    def __init__(self, transforms, source):
        """`transforms`: {frame_idx: 3x3 matrix mapping that frame's pixels to stabilised coordinates}."""
        self.transforms, self.source = transforms, source
        self.frames = sorted(transforms)

    def stabilise(self, frame, point):
        if not self.frames:
            return np.asarray(point, dtype=float)
        index = bisect_right(self.frames, frame)
        matrix = self.transforms[self.frames[max(0, index - 1)]]  # nearest earlier analysed frame
        x, y = point
        return matrix[:2, :2] @ np.array([x, y], dtype=float) + matrix[:2, 2]

    def scale(self, frame):
        """Zoom factor of the frame relative to the stabilised coordinates (1 = no zoom change)."""
        if not self.frames:
            return 1.0
        index = bisect_right(self.frames, frame)
        matrix = self.transforms[self.frames[max(0, index - 1)]]
        return float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))

    @classmethod
    def from_warps(cls, warps, scenes=None):
        """`warps`: {frame_idx: 2x3 affine mapping the previous analysed frame's pixels to this frame's}."""
        transforms, to_frame, scene = {}, np.eye(3), None
        for frame in sorted(warps):
            current = (scenes or {}).get(frame, 0)
            warp = np.vstack([np.asarray(warps[frame], dtype=float).reshape(2, 3), [0, 0, 1]])
            # A new scene restarts the reference; its first warp spans the cut and is meaningless.
            to_frame = np.eye(3) if current != scene else warp @ to_frame
            scene = current
            transforms[frame] = np.linalg.inv(to_frame)
        return cls(transforms, "image_features")

    @classmethod
    def from_people(cls, rows):
        """Fallback: cumulative median frame-to-frame movement of tracked people (translation only)."""
        by_frame = defaultdict(dict)
        for row in rows:
            if row.get("track_id") and row["object_type"] in PEOPLE:
                box = row["bbox"]
                by_frame[row["frame_idx"]][row["track_id"]] = ((box[0] + box[2]) / 2, box[3])
        transforms, total, previous = {}, np.zeros(2), None
        for frame in sorted(by_frame):
            if previous is not None:
                shared = by_frame[frame].keys() & by_frame[previous].keys()
                if len(shared) >= 4:
                    total = total + np.median([np.subtract(by_frame[frame][k], by_frame[previous][k]) for k in shared], axis=0)
            matrix = np.eye(3)
            matrix[:2, 2] = -total
            transforms[frame] = matrix
            previous = frame
        return cls(transforms, "tracked_people_median")


def estimate_warps(video_path, stride=1):
    """Frame-to-frame affine warps for a video, using the tracker's sparse optical-flow estimator."""
    import cv2
    from ultralytics.trackers.utils.gmc import GMC

    estimator, capture, warps, index = GMC(method="sparseOptFlow"), cv2.VideoCapture(str(video_path)), {}, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                warps[index] = estimator.apply(frame).tolist()
            index += 1
    finally:
        capture.release()
    return warps
