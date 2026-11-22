"""Why detections disagree with labelled people: every labelled box and every box of ours goes in one bin.

Precision and recall say how often a detector is wrong; they do not say how. Here a labelled
person is matched, split in pieces, only partly covered, covered by a box that is too big,
or missed. A box of ours is matched, a second box on someone already matched, a piece of a
person, too big, on a labelled person of another kind (a referee), or on nobody labelled.
The counts show which fault to fix. CPU only: boxes in, counts out.
"""
from __future__ import annotations

from collections import Counter, defaultdict

VERSION = "detection-diagnosis-1"
PLAYERS = {"player", "goalkeeper"}
PEOPLE = PLAYERS | {"referee"}
OURS = PLAYERS | {"referee", "person"}
MATCH_IOU = .5
MOSTLY_INSIDE = .7  # share of a box's area that lies inside another for it to count as a piece of it
TOUCHING = .2  # below this overlap two boxes are treated as unrelated
HEIGHT_BANDS = ((0, 60, "under 60 px"), (60, 120, "60 to 120 px"), (120, 240, "120 to 240 px"), (240, 10 ** 6, "240 px and over"))
LABEL_BINS = ("matched", "split", "piece_only", "oversized", "partial", "missed")
OUR_BINS = ("matched", "duplicate", "piece", "oversized", "other_kind", "off_label", "misplaced")


def _area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def _overlap(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    shared = _overlap(a, b)
    union = _area(a) + _area(b) - shared
    return shared / union if union > 0 else 0.


def inside(a, b):
    """Share of `a` that lies inside `b`."""
    area = _area(a)
    return _overlap(a, b) / area if area > 0 else 0.


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def height_band(box):
    height = box[3] - box[1]
    return next(name for low, high, name in HEIGHT_BANDS if low <= height < high)


def diagnose_frame(ours, labelled, target=PLAYERS, match_iou=MATCH_IOU):
    """Bin one frame. `ours` are dicts with bbox and object_type; `labelled` are dicts with box and role.

    Only labelled people whose role is in `target` must be found, and only our boxes of a kind that
    claims to be one of them are judged (a `person` box always is). Returns (label bins, our bins),
    each a list of (bin, item).
    """
    wanted = [p for p in labelled if p["role"] in target]
    others = [p for p in labelled if p["role"] not in target]
    mine = [r for r in ours if r["object_type"] in target or r["object_type"] == "person"]
    scored = sorted(((iou(r["bbox"], p["box"]), i, j) for i, r in enumerate(mine) for j, p in enumerate(wanted)), reverse=True)
    pair_of_ours, pair_of_label = {}, {}
    for overlap, i, j in scored:
        if overlap < match_iou:
            break
        if i not in pair_of_ours and j not in pair_of_label:
            pair_of_ours[i], pair_of_label[j] = j, i
    spare = [i for i in range(len(mine)) if i not in pair_of_ours]

    label_bins = []
    for j, person in enumerate(wanted):
        if j in pair_of_label:
            label_bins.append(("matched", person))
            continue
        pieces = [mine[i]["bbox"] for i in spare if inside(mine[i]["bbox"], person["box"]) >= MOSTLY_INSIDE]
        best = max((iou(mine[i]["bbox"], person["box"]) for i in range(len(mine))), default=0.)
        if len(pieces) >= 2 and iou(_union(pieces), person["box"]) >= match_iou:
            kind = "split"  # two or more of our boxes together cover the person
        elif pieces and best >= TOUCHING:
            kind = "piece_only"  # we boxed part of the person and nothing covers the rest
        elif any(inside(person["box"], mine[i]["bbox"]) >= MOSTLY_INSIDE and iou(mine[i]["bbox"], person["box"]) >= TOUCHING for i in spare):
            kind = "oversized"  # our box holds the person but is too big to match
        elif best >= TOUCHING:
            kind = "partial"
        else:
            kind = "missed"
        label_bins.append((kind, person))

    our_bins = []
    for i, row in enumerate(mine):
        box = row["bbox"]
        if i in pair_of_ours:
            our_bins.append(("matched", row))
        elif any(iou(box, person["box"]) >= match_iou for person in wanted):
            our_bins.append(("duplicate", row))  # a second box on a person who already has one
        elif any(iou(box, person["box"]) >= match_iou for person in others):
            our_bins.append(("other_kind", row))  # a referee boxed as a player, for example
        elif any(inside(box, person["box"]) >= MOSTLY_INSIDE for person in wanted):
            our_bins.append(("piece", row))
        elif any(inside(person["box"], box) >= MOSTLY_INSIDE and iou(box, person["box"]) >= TOUCHING for person in wanted):
            our_bins.append(("oversized", row))
        elif max((iou(box, person["box"]) for person in labelled), default=0.) < .1:
            our_bins.append(("off_label", row))  # nobody labelled is here: crowd, staff, or nothing
        else:
            our_bins.append(("misplaced", row))
    return label_bins, our_bins


def diagnose(ours_by_frame, labels_by_frame, target=PLAYERS, match_iou=MATCH_IOU):
    """Counts over many frames. Both arguments are {frame: [...]}; only frames present in the labels are scored."""
    labels, ours = Counter(), Counter()
    by_height = defaultdict(Counter)
    frames = 0
    for frame, labelled in labels_by_frame.items():
        frames += 1
        label_bins, our_bins = diagnose_frame(ours_by_frame.get(frame, []), labelled, target, match_iou)
        for kind, person in label_bins:
            labels[kind] += 1
            by_height[height_band(person["box"])][kind] += 1
        for kind, _ in our_bins:
            ours[kind] += 1
    wanted, mine = sum(labels.values()), sum(ours.values())
    share = lambda n, d: round(n / d, 3) if d else None
    return dict(schema_version=VERSION, frames=frames, labelled=wanted, ours=mine,
                recall=share(labels["matched"], wanted), precision=share(ours["matched"], mine),
                labelled_bins={kind: labels[kind] for kind in LABEL_BINS}, our_bins={kind: ours[kind] for kind in OUR_BINS},
                recall_by_height={name: dict(labelled=sum(by_height[name].values()), recall=share(by_height[name]["matched"], sum(by_height[name].values())),
                                             split=by_height[name]["split"], piece_only=by_height[name]["piece_only"], missed=by_height[name]["missed"])
                                  for _, _, name in HEIGHT_BANDS if by_height[name]})


def merge(reports):
    """One report from several clips' reports, by adding their counts."""
    reports = [r for r in reports if r]
    labels, ours, heights = Counter(), Counter(), defaultdict(Counter)
    for report in reports:
        labels.update(report["labelled_bins"])
        ours.update(report["our_bins"])
        for name, row in report["recall_by_height"].items():
            heights[name].update(labelled=row["labelled"], matched=round((row["recall"] or 0) * row["labelled"]), split=row["split"], piece_only=row["piece_only"], missed=row["missed"])
    wanted, mine = sum(labels.values()), sum(ours.values())
    share = lambda n, d: round(n / d, 3) if d else None
    return dict(schema_version=VERSION, clips=len(reports), frames=sum(r["frames"] for r in reports), labelled=wanted, ours=mine,
                recall=share(labels["matched"], wanted), precision=share(ours["matched"], mine),
                labelled_bins={kind: labels[kind] for kind in LABEL_BINS}, our_bins={kind: ours[kind] for kind in OUR_BINS},
                recall_by_height={name: dict(labelled=heights[name]["labelled"], recall=share(heights[name]["matched"], heights[name]["labelled"]),
                                             split=heights[name]["split"], piece_only=heights[name]["piece_only"], missed=heights[name]["missed"])
                                  for _, _, name in HEIGHT_BANDS if heights[name]["labelled"]})
