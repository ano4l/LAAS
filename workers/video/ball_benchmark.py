"""How often the ball is found on labelled clips, and what kind of ball is missed.

Each frame with a labelled ball goes in one bin: the chosen ball is right, the chosen ball is something
else, the ball was detected but nothing was chosen, or the ball was not detected at all. The same frames
are also counted by what the ball was doing (its size, its speed in the image, whether it was at a
player's body or in the open), which shows where a fix is needed. CPU only: boxes in, counts out.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

VERSION = "ball-benchmark-1"
BALLS = {"ball", "sports ball"}
BINS = ("right", "wrong", "detected_not_chosen", "not_detected")
MATCH_SIZES = 1.5  # our box centre within this many ball sizes of the labelled centre
MATCH_MIN_PX = 8.
SIZE_BANDS = ((0, 8, "under 8 px"), (8, 12, "8 to 12 px"), (12, 18, "12 to 18 px"), (18, 10 ** 6, "18 px and over"))
SPEED_BANDS = ((0, 2, "still or slow, under 2 px a frame"), (2, 8, "2 to 8 px a frame"), (8, 20, "8 to 20 px a frame"), (20, 10 ** 6, "20 px a frame and over"))


def load_balls(doc):
    """{frame_idx: (x1, y1, x2, y2)} for the labelled ball, and {frame_idx: [person boxes]}. Frame 0 is file 000001."""
    frame_of = {image["image_id"]: int(Path(image["file_name"]).stem) - 1 for image in doc["images"]}
    balls, people = {}, defaultdict(list)
    for item in doc["annotations"]:
        role = (item.get("attributes") or {}).get("role")
        if item.get("supercategory") != "object" or not item.get("bbox_image"):
            continue
        box = item["bbox_image"]
        box = (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])
        if role == "ball":
            balls[frame_of[item["image_id"]]] = box
        elif role in {"player", "goalkeeper", "referee", "other"}:
            people[frame_of[item["image_id"]]].append(box)
    return balls, dict(people)


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _band(value, bands):
    return next(name for low, high, name in bands if low <= value < high)


def score(rows, balls, people, frames):
    """Counts for one clip. `rows` are tracking rows; `frames` is how many frames the clip has."""
    ours = defaultdict(list)
    for row in rows:
        if row.get("object_type") in BALLS and row.get("bbox"):
            ours[row["frame_idx"]].append(row)
    bins, by = Counter(), dict(size=defaultdict(Counter), speed=defaultdict(Counter), place=defaultdict(Counter))
    found_by, confidence, spare, detected = Counter(), [], 0, 0
    previous = None
    for frame in sorted(balls):
        box = balls[frame]
        cx, cy = _centre(box)
        size = max(box[2] - box[0], box[3] - box[1])
        reach = max(MATCH_MIN_PX, MATCH_SIZES * size)
        near = lambda row: ((_centre(row["bbox"])[0] - cx) ** 2 + (_centre(row["bbox"])[1] - cy) ** 2) ** .5 <= reach
        candidates = [row for row in ours.get(frame, []) if not row.get("ball_rejected")]
        matching = [row for row in candidates if near(row)]
        chosen = next((row for row in candidates if row.get("ball_selected")), None)
        kind = "right" if chosen is not None and near(chosen) else "wrong" if chosen is not None else "detected_not_chosen" if matching else "not_detected"
        bins[kind] += 1
        spare += len(candidates) - len(matching)
        detected += bool(matching)
        if matching:
            best = max(matching, key=lambda row: row["confidence"])
            found_by[best.get("model_role") or "unknown"] += 1
            confidence.append(best["confidence"])
        speed = None
        if previous is not None and frame - previous[0] == 1:
            speed = ((cx - previous[1]) ** 2 + (cy - previous[2]) ** 2) ** .5
        previous = (frame, cx, cy)
        at_player = any(b[0] - 2 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2 for b in people.get(frame, []))
        by["size"][_band(size, SIZE_BANDS)][kind] += 1
        by["place"]["at a player's body in the image" if at_player else "in the open"][kind] += 1
        if speed is not None:
            by["speed"][_band(speed, SPEED_BANDS)][kind] += 1
    unlabelled = sum(1 for frame, items in ours.items() if frame not in balls and any(r.get("ball_selected") for r in items))
    confidence.sort()
    labelled = sum(bins.values())
    return dict(schema_version=VERSION, frames=frames, labelled_frames=labelled, bins={kind: bins[kind] for kind in BINS},
                detected_frames=detected,
                chosen_frames=bins["right"] + bins["wrong"] + unlabelled, chosen_without_a_labelled_ball=unlabelled,
                other_candidates_in_labelled_frames=spare, found_by=dict(found_by),
                matching_confidence=dict(low=confidence[len(confidence) // 10], median=confidence[len(confidence) // 2]) if confidence else None,
                by={name: {band: {kind: counts[kind] for kind in BINS} for band, counts in groups.items()} for name, groups in by.items()})


def merge(reports):
    """One report from several clips' reports, by adding their counts, with the rates worked out."""
    reports = [r for r in reports if r]
    bins, by, found_by = Counter(), defaultdict(lambda: defaultdict(Counter)), Counter()
    for report in reports:
        bins.update(report["bins"])
        found_by.update(report["found_by"])
        for name, groups in report["by"].items():
            for band, counts in groups.items():
                by[name][band].update(counts)
    labelled = sum(bins.values())
    chosen = sum(r["chosen_frames"] for r in reports)
    share = lambda n, d: round(n / d, 3) if d else None
    rates = lambda counts: dict(labelled=sum(counts.values()), chosen_right=share(counts["right"], sum(counts.values())),
                                not_detected=share(counts["not_detected"], sum(counts.values())),
                                detected_not_chosen=share(counts["detected_not_chosen"], sum(counts.values())), wrong=share(counts["wrong"], sum(counts.values())))
    return dict(schema_version=VERSION, clips=len(reports), frames=sum(r["frames"] for r in reports), labelled_frames=labelled,
                bins={kind: bins[kind] for kind in BINS},
                recall_detected=share(sum(r["detected_frames"] for r in reports), labelled),
                recall_chosen=share(bins["right"], labelled), precision_chosen=share(bins["right"], chosen),
                chosen_without_a_labelled_ball=sum(r["chosen_without_a_labelled_ball"] for r in reports),
                other_candidates_in_labelled_frames=sum(r["other_candidates_in_labelled_frames"] for r in reports), found_by=dict(found_by),
                by={name: {band: rates(counts) for band, counts in groups.items()} for name, groups in by.items()})
