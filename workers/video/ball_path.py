"""Offline selection of one ball candidate per frame using the whole clip.

The online selector decides frame by frame and gives up whenever two candidates score
similarly. Once every frame is known, the most consistent path through the candidates
can be found instead: a candidate is supported by nearby candidates in the frames
before and after it, and an isolated weak detection is not worth starting a path for.

Only observed detections are ever selected. Gaps stay gaps; nothing is interpolated.
Selection is not proof that a candidate is the ball.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass
import math

import numpy as np

from workers.video.camera_motion import CameraMotion

BALLS = {"ball", "sports ball"}
VERSION = "ball-path-2"


@dataclass(frozen=True)
class PathSettings:
    min_confidence: float = 0.1
    base_cost: float = 0.15            # confidence a candidate must exceed to be worth selecting on its own merits
    start_cost: float = 0.25           # extra cost of starting a new path segment
    outside_field_cost: float = 0.1
    continuity_bonus: float = 0.15     # reward for following closely from the previous selected candidate
    max_link_seconds: float = 1.0
    base_reach: float = 0.02           # allowed movement at zero gap, as a share of the image diagonal
    reach_per_second: float = 0.5
    # A run of weak detections with no confident one among them is usually a pitch
    # marking, a boot or a sideline object, so every segment needs one such anchor.
    anchor_confidence: float = 0.35
    # A weak detection is kept only when confident ones sit shortly before and after
    # it on the same segment (a blurred ball in flight). Weak runs at the start or end
    # of a segment are where the path tends to wander onto boots and markings.
    bridge_seconds: float = 0.6
    # Weak detections that keep recurring at one stabilised spot (a sideline object, a
    # pitch marking) are removed before the path is chosen. A resting ball that is
    # detected confidently is kept.
    static_window_seconds: float = 3.0
    static_radius: float = 0.008       # share of the image diagonal
    static_min_frames: int = 6
    static_min_seconds: float = 1.5
    static_max_confidence: float = 0.5   # no detection at the spot reaches this
    # A painted penalty spot or centre spot is round and white, and is detected as a ball with high confidence.
    # With a pitch calibration, candidates that sit still on one of those spots on the pitch map for a second or
    # more, with nobody beside them most of that time, are the paint. A ball placed there with a player standing
    # over it (a kick-off) is kept; a ball left alone on a penalty spot is not seen until it is struck.
    spot_radius_m: float = 2.5           # how far from the spot's place on the pitch map a candidate may sit
    spot_spread_m: float = 0.75          # it is still when candidates this close to it
    spot_still_seconds: float = 0.6      # are found over this long, within a second either side
    spot_min_frames: int = 6
    spot_min_seconds: float = 1.5
    spot_player_m: float = 2.0           # a player this close is standing over it
    spot_unattended_share: float = 0.7   # of its frames with nobody that close


def _centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _best_chain(frames, points, scenes, emit, active, fps, settings):
    """Highest-scoring chain through the active candidates, at most one per frame.

    A candidate either continues from a recent nearby candidate (continuity bonus) or
    starts a new segment (start cost) after the best chain ending in any earlier frame.
    Returns [(candidate index, continued_from_previous)].
    """
    by_frame = defaultdict(list)
    for index in np.flatnonzero(active):
        by_frame[int(frames[index])].append(int(index))
    best, previous, continued = {}, {}, {}
    running_score, running_node = 0.0, None
    recent = []  # candidates from frames inside the link window, oldest first
    max_gap = settings.max_link_seconds * fps
    for frame in sorted(by_frame):
        recent = [j for j in recent if frames[j] >= frame - max_gap]
        for i in by_frame[frame]:
            score, parent, linked = running_score - settings.start_cost, running_node, False
            for j in recent:
                if scenes[j] != scenes[i]:
                    continue
                gap = (frame - frames[j]) / fps
                reach = settings.base_reach + settings.reach_per_second * gap
                distance = float(np.linalg.norm(points[i] - points[j]))
                if distance > reach:
                    continue
                candidate = best[j] + settings.continuity_bonus * (1 - distance / reach)
                if candidate > score:
                    score, parent, linked = candidate, j, True
            best[i], previous[i], continued[i] = score + emit[i], parent, linked
        top = max(by_frame[frame], key=lambda i: best[i])
        if best[top] > running_score:
            running_score, running_node = best[top], top
        recent.extend(by_frame[frame])
    chain, node = [], running_node
    while node is not None:
        chain.append((node, continued[node]))
        node = previous[node]
    return chain[::-1]


def _static(frames, points, scenes, confidence, fps, settings):
    """Candidates in a spot where detections keep recurring for a second or more, none of them confident."""
    frame_ids = sorted(set(frames.tolist()))
    by_frame = defaultdict(list)
    for index, frame in enumerate(frames.tolist()):
        by_frame[frame].append(index)
    window = settings.static_window_seconds * fps
    static = np.zeros(len(frames), bool)
    for i in np.flatnonzero(confidence < settings.static_max_confidence):
        low, high = bisect_left(frame_ids, frames[i] - window), bisect_left(frame_ids, frames[i] + window + 1)
        hit_frames, strongest = [], 0.0
        for frame in frame_ids[low:high]:
            near = [j for j in by_frame[frame]
                    if scenes[j] == scenes[i] and np.linalg.norm(points[j] - points[i]) <= settings.static_radius]
            if near:
                hit_frames.append(frame)
                strongest = max(strongest, float(confidence[near].max()))
        # A resting ball is usually detected confidently at least once; a marking or sideline object is not.
        static[i] = (len(hit_frames) >= settings.static_min_frames
                     and (hit_frames[-1] - hit_frames[0]) / fps >= settings.static_min_seconds
                     and strongest < settings.static_max_confidence)
    return static


PEOPLE = {"person", "player", "goalkeeper", "referee"}


def _painted(nodes, rows, pitch, fps, settings):
    """Which ball candidates are a painted spot on the pitch (see PathSettings). `pitch` is {frame_idx: calibration record}."""
    from workers.video import pitch_keypoints as pk
    painted = np.zeros(len(nodes), bool)
    feet = defaultdict(list)
    for row in rows:
        if row["object_type"] in PEOPLE and row.get("bbox"):
            feet[row["frame_idx"]].append(((row["bbox"][0] + row["bbox"][2]) / 2, row["bbox"][3]))
    by_frame = defaultdict(list)
    for index, node in enumerate(nodes):
        by_frame[node["frame_idx"]].append(index)
    place, attended = {}, {}
    for frame, indices in by_frame.items():
        record = pitch.get(frame)
        if not record or record.get("status") != "calibrated":
            continue
        xy, _ = pk.to_pitch(record, [_centre(nodes[i]["bbox"]) for i in indices] + feet[frame])
        people = xy[len(indices):]
        people = people[~np.isnan(people[:, 0])]
        for i, point in zip(indices, xy):
            if not np.isnan(point[0]):
                place[i] = point
                attended[i] = bool(len(people)) and float(np.linalg.norm(people - point, axis=1).min()) <= settings.spot_player_m
    for spot in ((11., pk.WIDTH_M / 2), (pk.LENGTH_M - 11., pk.WIDTH_M / 2), (pk.LENGTH_M / 2, pk.WIDTH_M / 2)):
        near = [i for i, point in place.items() if np.linalg.norm(point - spot) <= settings.spot_radius_m]
        if len(near) < settings.spot_min_frames:
            continue
        # Still is judged a second at a time, so a calibration that drifts over the clip does not hide the spot,
        # and a ball rolling through the area, which is not still, is kept.
        still = []
        for i in near:
            beside = [nodes[j]["frame_idx"] for j in near if abs(nodes[j]["frame_idx"] - nodes[i]["frame_idx"]) <= fps
                      and np.linalg.norm(place[j] - place[i]) <= settings.spot_spread_m]
            if (max(beside) - min(beside)) / fps >= settings.spot_still_seconds:
                still.append(i)
        if not still:
            continue
        frames = sorted({nodes[i]["frame_idx"] for i in still})
        if (len(frames) >= settings.spot_min_frames and (frames[-1] - frames[0]) / fps >= settings.spot_min_seconds
                and sum(not attended[i] for i in still) / len(still) >= settings.spot_unattended_share):
            painted[still] = True
    return painted


def _segments(chain):
    segments = []
    for index, continued in chain:
        if not continued or not segments:
            segments.append([])
        segments[-1].append(index)
    return segments


def select_ball_path(rows, metadata, settings=PathSettings(), camera=None, pitch=None):
    """Mark the chosen ball candidates in `rows` (in place) and return a summary.

    `pitch` ({frame_idx: calibration record}) lets painted spots be told from the ball; without it they are not.

    Each ball row gets ball_selected (offline choice), ball_selected_online (the
    previous frame-by-frame choice, when present), ball_path_segment and
    ball_path_support ("confident" or "weak_between_confident").
    """
    fps = metadata["fps"]
    diagonal = math.hypot(metadata["width"], metadata["height"])
    camera = camera or CameraMotion.from_people(rows)
    balls = [r for r in rows if r["object_type"] in BALLS]
    online = sum(1 for r in balls if r.get("ball_selected"))
    for row in balls:
        if "ball_selected" in row and "ball_selected_online" not in row:
            row["ball_selected_online"] = bool(row["ball_selected"])
        row["ball_selected"] = False
        row["ball_path_segment"] = None
        row["ball_path_support"] = None
        row.pop("ball_rejected", None)
    nodes = sorted((r for r in balls if r["confidence"] >= settings.min_confidence), key=lambda r: r["frame_idx"])
    summary = dict(version=VERSION, candidate_rows=len(balls), candidate_frames=len({r["frame_idx"] for r in balls}),
                   selected_online=online, settings=asdict(settings),
                   limitations=["Selection picks the most consistent observed candidates; it is not validated against labelled ball positions.",
                                "Missing positions stay missing; nothing is interpolated.",
                                "A wrong candidate that moves plausibly near a confident detection can still be selected."])
    if not nodes:
        return dict(summary, selected=0, segments=0, unanchored_candidates_rejected=0, static_candidates_rejected=0,
                    painted_spot_candidates_rejected=0, weak_candidates_not_bridged=0)
    frames = np.array([r["frame_idx"] for r in nodes])
    # Positions with camera motion removed, as a share of the image diagonal.
    points = np.array([camera.stabilise(r["frame_idx"], _centre(r["bbox"])) / diagonal for r in nodes])
    scenes = [r.get("scene_id", 0) for r in nodes]
    confidence = np.array([r["confidence"] for r in nodes])
    emit = confidence - settings.base_cost - settings.outside_field_cost * np.array(
        [r.get("field_plausibility") == "outside_ground_region" for r in nodes])

    # Segments without a confident anchor are removed from the pool and the path is
    # recomputed, so a real ball they were hiding can be picked up instead.
    active = ~_static(frames, points, scenes, confidence, fps, settings)
    static_rejected = int((~active).sum())
    painted = _painted(nodes, rows, pitch, fps, settings) if pitch else np.zeros(len(nodes), bool)
    for index in np.flatnonzero(painted):
        nodes[index]["ball_rejected"] = "painted_spot"
    painted_rejected = int((painted & active).sum())
    active &= ~painted
    while True:
        segments = _segments(_best_chain(frames, points, scenes, emit, active, fps, settings))
        weak = [s for s in segments if confidence[s].max() < settings.anchor_confidence]
        if not weak:
            break
        for segment in weak:
            active[segment] = False
    unbridged = 0
    for number, segment in enumerate(segments):
        anchors = [frames[i] for i in segment if confidence[i] >= settings.anchor_confidence]
        for index in segment:
            if confidence[index] < settings.anchor_confidence:
                limit = settings.bridge_seconds * fps
                before = any(0 < frames[index] - a <= limit for a in anchors)
                after = any(0 < a - frames[index] <= limit for a in anchors)
                if not (before and after):
                    unbridged += 1
                    continue
            nodes[index]["ball_selected"] = True
            nodes[index]["ball_path_segment"] = number
            nodes[index]["ball_path_support"] = "confident" if confidence[index] >= settings.anchor_confidence else "weak_between_confident"
    return dict(summary, selected=sum(1 for r in nodes if r["ball_selected"]), segments=len(segments),
                weak_candidates_not_bridged=unbridged,
                unanchored_candidates_rejected=int((~active).sum()) - static_rejected - painted_rejected,
                static_candidates_rejected=static_rejected, painted_spot_candidates_rejected=painted_rejected, camera_motion=camera.source)
