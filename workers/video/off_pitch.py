"""People who stand off the pitch for the whole clip (coaches, substitutes, stewards, ball boys) are not players.

A general person detector finds everyone in view, so people beside the pitch were tracked as players and
took kit groups, touches and distance. Where the clip has a pitch calibration, each track's foot point is
placed on the pitch map; a track that is outside the outline in nearly all of its frames is set aside.
A player who steps out for a throw-in or a corner is inside for most of the track and is kept.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from workers.video import pitch_keypoints as pk

ROLE = "off_pitch"
# Chosen on labelled SoccerNet clips, whose calibration fails the line check, so this is the hard case: ten clips lost
# 0.11% of player observations (one track) and dropped 20% of the boxes on nobody labelled; three held-out clips
# lost none and dropped 16%. Wider margins and longer tracks removed fewer bystanders and saved no players.
MARGIN_M = 1.  # a foot point within this of the outline counts as on the pitch
MIN_FRAMES = 10  # placed frames a track needs before it can be set aside
MIN_SHARE = .8  # of those frames that must be outside
STEPS_M = (0., 1., 2., 3., 5.)  # margins reported by `measure`, for choosing and checking MARGIN_M


def outside_m(record, boxes):
    """Metres each box's foot point lies outside the assumed pitch outline: 0 on the pitch, inf beyond the horizon."""
    feet = [((box[0] + box[2]) / 2, box[3]) for box in boxes]
    xy, _ = pk.to_pitch(record, feet)
    dx = np.maximum(0., np.maximum(-xy[:, 0], xy[:, 0] - pk.LENGTH_M))
    dy = np.maximum(0., np.maximum(-xy[:, 1], xy[:, 1] - pk.WIDTH_M))
    return np.where(np.isnan(xy[:, 0]), np.inf, np.hypot(dx, dy))


def measure(rows, pitch_frames):
    """{track_id: dict(placed=n, beyond=[n further out than each of STEPS_M])} over the calibrated frames."""
    by_frame = defaultdict(list)
    for row in rows:
        if row.get("track_id") and row.get("bbox"):
            by_frame[row["frame_idx"]].append(row)
    out = {}
    for frame, people in by_frame.items():
        record = pitch_frames.get(frame)
        if not record or record.get("status") != "calibrated":
            continue
        for row, distance in zip(people, outside_m(record, [row["bbox"] for row in people])):
            item = out.setdefault(row["track_id"], dict(placed=0, beyond=[0] * len(STEPS_M)))
            item["placed"] += 1
            for index, step in enumerate(STEPS_M):
                item["beyond"][index] += bool(distance > step)
    return out


def settle(rows, pitch_frames, keep=()):
    """{track_id: ROLE} for tracks that stay off the pitch. `keep` are tracks never set aside (settled referees)."""
    index = STEPS_M.index(MARGIN_M)
    return {track: ROLE for track, item in measure(rows, pitch_frames).items()
            if track not in keep and item["placed"] >= MIN_FRAMES and item["beyond"][index] / item["placed"] >= MIN_SHARE}
