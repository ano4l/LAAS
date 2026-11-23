"""Who was nearest the ball at a moment in the video, to save typing when a person tags an event.

This reads the stored detections; it decides nothing. The ball is seen in well under half of
analysed frames and the nearest box is not always the player on the ball, so the answer is a
shortlist for the person tagging, who still chooses. Only identities a person has already
named can be offered by name.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path

VERSION = "tagging-hints-1"
PEOPLE = {"player", "goalkeeper"}
BALLS = {"ball", "sports ball"}
WINDOW_SECONDS = .4  # how far from the asked time a frame with the ball may be
MAX_DISTANCE_HEIGHTS = 1.5  # feet further from the ball than this are not offered
LIMIT = 3


@lru_cache(maxsize=4)
def _frames(path: str, modified: float):
    """{frame index: detections} for one stored run. Cached per file version, because tagging asks many times."""
    frames = defaultdict(list)
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        frames[row["frame_idx"]].append(row)
    return dict(frames)


def load_frames(tracking_path):
    path = Path(tracking_path)
    return _frames(str(path), path.stat().st_mtime) if path.is_file() else None


def _ball(rows):
    """The one ball the analysis settled on in this frame, by the same rule the insights use."""
    balls = [row for row in rows if row.get("object_type") in BALLS and row.get("bbox")]
    chosen = [row for row in balls if row.get("ball_selected")] if any("ball_selected" in row for row in balls) else balls
    if len(chosen) != 1 or not (chosen[0].get("confidence", 0) >= .25 or chosen[0].get("ball_path_support")):
        return None
    a, b, c, d = chosen[0]["bbox"]
    return (a + c) / 2, (b + d) / 2


def near_ball(frames, fps, seconds, labels):
    """Named and unnamed identities nearest the ball around `seconds`. `labels` are the saved player names for the run."""
    base = dict(schema_version=VERSION, asked_seconds=round(float(seconds), 3), candidates=[],
                note="A shortlist from stored detections, nearest first. The nearest box is not always the player on the ball: check the footage.")
    if frames is None:
        return dict(base, status="no_tracking", reason="No detections are stored for this video.")
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid time")
    target = seconds * fps
    # The closest analysed frame, within the window, in which the ball was seen.
    seen = sorted((abs(index - target), index) for index in frames if abs(index - target) <= WINDOW_SECONDS * fps)
    hit = next(((index, ball) for _, index in seen for ball in [_ball(frames[index])] if ball), None)
    if not hit:
        return dict(base, status="no_ball", reason=f"The ball was not seen within {WINDOW_SECONDS} seconds of this moment.")
    index, (bx, by) = hit
    named = {track: player for player in labels.get("players", []) for track in player["tracks"]}
    ranked = []
    for row in frames[index]:
        track = row.get("linked_track_id") or row.get("track_id")
        if row.get("object_type") not in PEOPLE or not track or not row.get("bbox"):
            continue
        a, b, c, d = row["bbox"]
        distance = math.hypot((a + c) / 2 - bx, d - by) / max(1., d - b)
        if distance <= MAX_DISTANCE_HEIGHTS:
            ranked.append((distance, track))
    ranked.sort()
    if not ranked:
        return dict(base, status="no_player_near", reason="The ball was seen, but no tracked player was close to it.", frame_seconds=round(index / fps, 3))
    candidates = []
    for distance, track in ranked[:LIMIT]:
        player = named.get(track)
        candidates.append(dict(track_id=track, distance_heights=round(distance, 2), named=bool(player), name=player["name"] if player else None,
                               number=player["number"] if player else None, team=player.get("team") if player else None,
                               lineup_id=player.get("lineup_id") if player else None))
    return dict(base, status="ok", frame_seconds=round(index / fps, 3), candidates=candidates)
