"""Score a finished run against SoccerNet Game State Reconstruction labels (`Labels-GameState.json`, version 1.3).

The labels give, for every frame of a 30 s clip, each person's box, role, team side, shirt
number when visible, and foot position on the pitch in metres. They were made by people with
an annotated camera per frame, so they are a reference, not perfect truth: their positions
carry annotation error too.

Our people are matched to labelled players and goalkeepers frame by frame on their boxes
(intersection over union of at least 0.5, best first). Everything below is measured on those
matches:

- positions: our pitch position against the labelled one, and our calibration alone (the
  labelled foot point pushed through our mapping);
- distance and speed: our pitch_motion measure run on our positions and on the labelled
  positions over exactly the same frames, so only position error differs;
- identities: which labelled player each of our identities mostly follows;
- merge suggestions: whether both identities of a pair follow the same labelled player;
- kit groups and shirt-number proposals, against the labelled team side and number.

CPU only: no model, no video.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from workers.video import merge_suggestions as ms
from workers.video import pitch_keypoints as pk
from workers.video import pitch_motion as pm

VERSION = "gsr-benchmark-1"
PEOPLE = {"player", "goalkeeper"}
MATCH_IOU = .5
PITCH_LIMIT = (60., 40.)  # labelled positions further out than this are projection artefacts, not places on the pitch
MIN_IDENTITY_MATCHES = 5
MIN_PURITY = .5  # share of an identity's observations that must be one labelled player for it to count as following them
ORIENTATIONS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
SMOOTHING_TRIED = (.5, 1., 1.5, 2.)  # position averaging windows (s) compared for distance and speed


def _iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def load_labels(doc):
    """{frame_idx: [person, ...]} for labelled players, goalkeepers and referees, plus clip info. Frame 0 is file 000001."""
    frame_of = {image["image_id"]: int(Path(image["file_name"]).stem) - 1 for image in doc["images"]}
    frames = defaultdict(list)
    for item in doc["annotations"]:
        attributes = item.get("attributes") or {}
        if item.get("supercategory") != "object" or attributes.get("role") not in PEOPLE | {"referee"}:
            continue
        box = item["bbox_image"]
        place = item.get("bbox_pitch") or {}
        xy = (place.get("x_bottom_middle"), place.get("y_bottom_middle"))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy) or abs(xy[0]) > PITCH_LIMIT[0] or abs(xy[1]) > PITCH_LIMIT[1]:
            xy = None
        jersey = attributes.get("jersey")
        frames[frame_of[item["image_id"]]].append(dict(
            track=int(item["track_id"]), role=attributes["role"], team=attributes.get("team"),
            jersey=jersey if isinstance(jersey, str) and jersey.isdigit() else None,
            box=(box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]), pitch=xy))
    info = doc.get("info", {})
    return dict(frames=dict(frames), fps=float(info.get("frame_rate") or 25), name=info.get("name"),
                action=info.get("action_class"), visibility=info.get("visibility"), frame_count=len(doc["images"]))


def match(rows, labels):
    """[(row, person)] per frame, greedy on box overlap, best first. Only our players and goalkeepers against labelled ones."""
    ours = defaultdict(list)
    for row in rows:
        if row.get("object_type") in PEOPLE and row.get("bbox"):
            ours[row["frame_idx"]].append(row)
    pairs = []
    for frame, people in labels["frames"].items():
        wanted = [p for p in people if p["role"] in PEOPLE]
        scored = sorted(((_iou(r["bbox"], p["box"]), i, j) for i, r in enumerate(ours.get(frame, [])) for j, p in enumerate(wanted)), reverse=True)
        used_r, used_p = set(), set()
        for overlap, i, j in scored:
            if overlap < MATCH_IOU:
                break
            if i in used_r or j in used_p:
                continue
            used_r.add(i)
            used_p.add(j)
            pairs.append((ours[frame][i], wanted[j]))
    return pairs


def identity(row):
    return row.get("linked_track_id") or row.get("track_id")


def _foot(box):
    return ((box[0] + box[2]) / 2, box[3])


def _ours(record, box):
    """Our pitch position for one box, centred on the halfway spot like the labels, or None."""
    if not record or record.get("status") != "calibrated":
        return None
    xy, usable = pk.to_pitch(record, [_foot(box)])
    return (float(xy[0][0]) - pk.LENGTH_M / 2, float(xy[0][1]) - pk.WIDTH_M / 2) if usable[0] else None


def _stats(errors):
    if not errors:
        return dict(count=0)
    e = np.array(errors)
    return dict(count=int(len(e)), median_m=round(float(np.median(e)), 2), mean_m=round(float(e.mean()), 2),
                p90_m=round(float(np.percentile(e, 90)), 2),
                within_1m=round(float((e <= 1).mean()), 3), within_2m=round(float((e <= 2).mean()), 3), within_5m=round(float((e <= 5).mean()), 3))


def score(rows, labels, pitch, suggestions=None, numbers=None):
    """Everything measured for one clip. `pitch` is {frame_idx: calibration record} or None."""
    pitch = pitch or {}
    fps = labels["fps"]
    pairs = match(rows, labels)
    labelled = [p for people in labels["frames"].values() for p in people if p["role"] in PEOPLE]
    our_people = [r for r in rows if r.get("object_type") in PEOPLE]

    # Positions. Which end and which touchline the labels call positive is learned once per clip.
    placed = []  # (frame, person, ours, calibration-only)
    for row, person in pairs:
        record = pitch.get(row["frame_idx"])
        ours = _ours(record, row["bbox"])
        if ours is not None and person["pitch"] is not None:
            placed.append((row["frame_idx"], row, person, ours, _ours(record, person["box"])))

    def error(a, b, orientation):
        return math.hypot(orientation[0] * a[0] - b[0], orientation[1] * a[1] - b[1])

    orientation = min(ORIENTATIONS, key=lambda o: np.median([error(ours, p["pitch"], o) for _, _, p, ours, _ in placed]) if placed else 0)
    full = [error(ours, p["pitch"], orientation) for _, _, p, ours, _ in placed]
    calibration_only = [error(c, p["pitch"], orientation) for _, _, p, _, c in placed if c is not None]
    # A frame where another orientation fits its players far better has the ends or touchlines swapped.
    by_frame = defaultdict(list)
    for frame, _, p, ours, _ in placed:
        by_frame[frame].append((ours, p["pitch"]))
    flipped = sum(1 for items in by_frame.values()
                  if min(np.median([error(a, b, o) for a, b in items]) for o in ORIENTATIONS) + 5 < np.median([error(a, b, orientation) for a, b in items]))
    calibrated = sum(1 for record in pitch.values() if record.get("status") == "calibrated")
    tried = len(pitch)

    # Distance and speed over the same frames: our boxes, the labelled boxes through our mapping, and the labelled positions.
    sides = ("ours", "calibration_only", "labelled")
    series = defaultdict(list)  # labelled track -> [(frame, ours, calibration only, labelled)]
    for frame, _, p, ours, c in placed:
        if c is not None:
            series[p["track"]].append((frame, (orientation[0] * ours[0], orientation[1] * ours[1]),
                                       (orientation[0] * c[0], orientation[1] * c[1]), p["pitch"]))
    step = 1 / fps
    motion = {side: dict(distance_m=0., seconds=0., bands=[0.] * len(pm.SPEED_BANDS), fastest=[], rejected=0) for side in sides}
    smoothing = {seconds: {side: [0., 0.] for side in sides} for seconds in SMOOTHING_TRIED}
    per_player = []
    for track, items in series.items():
        items.sort()
        stretches, last = [], None
        for item in items:
            if last is None or item[0] - last > 1:
                stretches.append([])
            stretches[-1].append(item)
            last = item[0]
        totals = {}
        for index, side in enumerate(sides, start=1):
            d = s = 0.
            fast = None
            for stretch in stretches:
                samples = [(x[0] * step, *x[index]) for x in stretch]
                distance, seconds, fastest, rejected, bands = pm._measure(samples, step)
                d, s = d + distance, s + seconds
                motion[side]["distance_m"] += distance
                motion[side]["seconds"] += seconds
                motion[side]["rejected"] += rejected
                motion[side]["bands"] = [a + b for a, b in zip(motion[side]["bands"], bands)]
                if fastest is not None:
                    fast = max(fast or 0., fastest)
                for window in SMOOTHING_TRIED:
                    distance, seconds, *_ = pm._measure(samples, step, smooth_s=window)
                    smoothing[window][side][0] += distance
                    smoothing[window][side][1] += seconds
            if fast is not None:
                motion[side]["fastest"].append(fast)
            totals[side] = (d, s, fast)
        if totals["labelled"][1] >= pm.MIN_TRACK_S:
            per_player.append(dict(track=track, seconds=round(totals["labelled"][1], 2),
                                   labelled_m=round(totals["labelled"][0], 1), ours_m=round(totals["ours"][0], 1),
                                   calibration_only_m=round(totals["calibration_only"][0], 1),
                                   labelled_fastest=round(totals["labelled"][2] or 0., 2), ours_fastest=round(totals["ours"][2] or 0., 2),
                                   calibration_only_fastest=round(totals["calibration_only"][2] or 0., 2)))
    for side in motion.values():
        side["mean_speed_m_s"] = round(side["distance_m"] / side["seconds"], 2) if side["seconds"] else None
        side["band_share"] = [round(b / side["seconds"], 3) if side["seconds"] else None for b in side["bands"]]
        side["fastest_median"] = round(float(np.median(side["fastest"])), 2) if side["fastest"] else None
        side["fastest_max"] = round(max(side["fastest"]), 2) if side["fastest"] else None
        side["distance_m"], side["seconds"] = round(side["distance_m"], 1), round(side["seconds"], 1)
        side["bands"] = [round(b, 1) for b in side["bands"]]
        del side["fastest"]
    mean_speed_by_smoothing = {str(window): {side: dict(distance_m=round(d, 1), seconds=round(s, 1), mean_speed_m_s=round(d / s, 3) if s else None)
                                             for side, (d, s) in values.items()} for window, values in smoothing.items()}
    ratios = [p["ours_m"] / p["labelled_m"] for p in per_player if p["labelled_m"] > 5]

    # Identities: which labelled player each of ours mostly follows.
    follows = defaultdict(Counter)
    for row, person in pairs:
        follows[identity(row)][person["track"]] += 1
    sizes = Counter(identity(r) for r in our_people)
    owner = {}
    for ident, counts in follows.items():
        track, count = counts.most_common(1)[0]
        if count >= MIN_IDENTITY_MATCHES and count / sizes[ident] >= MIN_PURITY:
            owner[ident] = track
    labelled_tracks = {p["track"] for p in labelled}
    pieces = Counter(owner.values())
    observed = Counter(p["track"] for p in labelled)

    # Merge suggestions against the labels.
    judged = []
    for s in (suggestions or {}).get("suggestions", []):
        a, b = owner.get(s["from_track"]), owner.get(s["to_track"])
        verdict = "unknown" if a is None or b is None else "same" if a == b else "different"
        judged.append(dict({k: s.get(k) for k in ("from_track", "to_track", "gap_seconds", "basis", "distance_m", "distance_heights", "fit", "rivals", "same_kit")}, verdict=verdict))
    spans = {}
    for r in our_people:
        ident = identity(r)
        first, last = spans.get(ident, (r["frame_idx"], r["frame_idx"]))
        spans[ident] = (min(first, r["frame_idx"]), max(last, r["frame_idx"]))
    truth = set()
    unreachable = 0
    for track in labelled_tracks:
        mine = sorted((spans[i][0], spans[i][1], i) for i, t in owner.items() if t == track and sizes[i] >= ms.MIN_OBSERVATIONS)
        for (_, end, a), (start, _, b) in zip(mine, mine[1:]):
            if start <= end:
                continue
            if (start - end) / fps <= ms.MAX_GAP_S:
                truth.add((a, b))
            else:
                unreachable += 1
    suggested = {(s["from_track"], s["to_track"]) for s in judged}

    # Kit groups against labelled team sides, with the better of the two ways round.
    kit_of, side_of = {}, defaultdict(Counter)
    for r in our_people:
        kit = r.get("linked_team") or r.get("team")
        if kit:
            kit_of.setdefault(identity(r), Counter())[kit] += 1
    for row, person in pairs:
        if person["team"]:
            side_of[identity(row)][person["team"]] += 1
    both = [(kit_of[i].most_common(1)[0][0], side_of[i].most_common(1)[0][0]) for i in owner if i in kit_of and side_of[i]]
    kits = sorted({k for k, _ in both})
    best = 0
    if kits:
        for flip in (False, True):
            mapping = {k: ("left" if (n % 2 == 0) != flip else "right") for n, k in enumerate(kits)}
            best = max(best, sum(1 for k, s in both if mapping[k] == s))

    # Shirt-number proposals against the labelled numbers.
    jersey = defaultdict(Counter)
    for people in labels["frames"].values():
        for p in people:
            if p["jersey"]:
                jersey[p["track"]][p["jersey"]] += 1
    shirts = []
    for proposal in (numbers or {}).get("proposals", []):
        if not proposal.get("proposed_number"):
            continue
        track = owner.get(proposal["track_id"])
        known = jersey[track].most_common(1)[0][0] if track is not None and jersey[track] else None
        shirts.append(dict(track_id=proposal["track_id"], proposed=proposal["proposed_number"], labelled=known,
                           digits=len(proposal["proposed_number"]), confident_reads=proposal.get("confident_reads"), agreement=proposal.get("agreement"),
                           verdict="unknown" if known is None else "right" if known == proposal["proposed_number"] else "wrong"))
    numbered = {i for i, t in owner.items() if jersey[t]}

    # Roles: which labelled kind each of our tracks sits on (referees included), beside what the pipeline's own cues said
    # about it frame by frame. Used to choose and check the rule that settles one role per track.
    kinds = PEOPLE | {"referee", "person"}
    everyone = defaultdict(list)
    followed, hinted, kit_votes = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        if row.get("object_type") in kinds and row.get("bbox") and row.get("track_id"):
            everyone[row["frame_idx"]].append(row)
            hinted[row["track_id"]][row["object_type"]] += 1
            kit_votes[row["track_id"]][row.get("team") or "none"] += 1
    for frame, people in labels["frames"].items():
        candidates = everyone.get(frame, [])
        scored = sorted(((_iou(r["bbox"], p["box"]), i, j) for i, r in enumerate(candidates) for j, p in enumerate(people)), reverse=True)
        used_r, used_p = set(), set()
        for overlap, i, j in scored:
            if overlap < MATCH_IOU:
                break
            if i not in used_r and j not in used_p:
                used_r.add(i)
                used_p.add(j)
                followed[candidates[i]["track_id"]][people[j]["role"]] += 1
    role_items = [dict(track_id=track, observations=sum(hinted[track].values()), labelled=dict(followed[track]), hinted=dict(hinted[track]), kits=dict(kit_votes[track]))
                  for track in sorted(hinted)]

    return dict(
        schema_version=VERSION, clip=labels["name"], action=labels["action"], visibility=labels["visibility"], frames=labels["frame_count"],
        roles=dict(tracks=len(role_items), items=role_items),
        detection=dict(labelled_boxes=len(labelled), our_boxes=len(our_people), matched=len(pairs),
                       recall=round(len(pairs) / len(labelled), 3) if labelled else None,
                       precision=round(len(pairs) / len(our_people), 3) if our_people else None),
        calibration=dict(frames_tried=tried, frames_calibrated=calibrated, share=round(calibrated / tried, 3) if tried else None,
                         matched_boxes_placed=len(placed), share_of_matched_placed=round(len(placed) / len(pairs), 3) if pairs else None,
                         orientation=list(orientation), frames_with_swapped_orientation=flipped, frames_with_positions=len(by_frame)),
        position_error=_stats(full), calibration_only_error=_stats(calibration_only),
        motion=dict(motion, players=len(per_player), mean_speed_by_smoothing=mean_speed_by_smoothing, distance_ratio_median=round(float(np.median(ratios)), 3) if ratios else None,
                    distance_ratio_p10_p90=[round(float(np.percentile(ratios, q)), 3) for q in (10, 90)] if ratios else None, per_player=per_player),
        identities=dict(labelled_players=len(labelled_tracks), our_identities=len(sizes), following_a_player=len(owner),
                        identities_per_followed_player=round(sum(pieces.values()) / len(pieces), 2) if pieces else None,
                        labelled_players_followed=len(pieces), labelled_players_seen_2s=sum(1 for t in labelled_tracks if observed[t] >= 2 * fps)),
        suggestions=dict(total=len(judged), same=sum(j["verdict"] == "same" for j in judged), different=sum(j["verdict"] == "different" for j in judged),
                         unknown=sum(j["verdict"] == "unknown" for j in judged), true_pairs=len(truth), true_pairs_suggested=len(truth & suggested),
                         true_pairs_beyond_gap_limit=unreachable, items=judged),
        kits=dict(identities=len(both), agreeing=best, share=round(best / len(both), 3) if both else None),
        shirt_numbers=dict(identities_with_labelled_number=len(numbered), proposals=len(shirts),
                           right=sum(s["verdict"] == "right" for s in shirts), wrong=sum(s["verdict"] == "wrong" for s in shirts),
                           unknown=sum(s["verdict"] == "unknown" for s in shirts), items=shirts))
