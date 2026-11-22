"""Tidying a detector's person boxes before tracking: one box per person, whatever class it was given.

Plain rows in, plain rows out (dicts with `bbox`, `confidence`, `object_type`), so the same steps
run in the pipeline and in offline scoring of saved detector output.
"""
from __future__ import annotations

from collections import Counter, defaultdict

NMS_IOU = .40
PERSON_KINDS = {"person", "player", "goalkeeper", "referee"}
STACK_MAX_PIECE_ASPECT = 1.9  # a piece of a person is squat; a whole standing person is taller than this
STACK_ASPECT = (1.6, 4.2)  # height over width a joined box must have to look like one person


def _area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def _overlap(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    shared = _overlap(a, b)
    union = _area(a) + _area(b) - shared
    return shared / union if union > 0 else 0.


def _aligned(a, b):
    """The two boxes share a column: centres close and widths similar."""
    width_a, width_b = max(1., a[2] - a[0]), max(1., b[2] - b[0])
    return abs((a[0] + a[2] - b[0] - b[2]) / 2) < .25 * max(width_a, width_b) and min(width_a, width_b) / max(width_a, width_b) > .5


def _fragment(a, b):
    """One box is a smaller piece lying inside the other, in the same column."""
    small, large = sorted((_area(a), _area(b)))
    return _aligned(a, b) and _overlap(a, b) / max(1., small) > .8 and small / max(1., large) < .65


def clean_people(rows, nms_iou=NMS_IOU):
    """Most confident first: a piece inside a larger box gives way to it, and boxes that overlap heavily are one person."""
    kept = []
    for row in sorted(rows, key=lambda r: -r["confidence"]):
        box = row["bbox"]
        contained = [other for other in kept if _fragment(box, other["bbox"])]
        if any(_area(other["bbox"]) > _area(box) for other in contained):
            continue
        if contained:
            kept = [other for other in kept if not any(other is c for c in contained)]
        if not any(_iou(box, other["bbox"]) > nms_iou for other in kept):
            kept.append(row)
    return kept


def join_stacked_pieces(rows):
    """Join an upper-body box and a legs box that sit one above the other into a single person box.

    Both pieces must be squat, share a column, and touch or nearly touch, and the joined box must be
    shaped like a standing person. Two whole players one behind the other are each too tall to qualify.
    """
    def aspect(box):
        return (box[3] - box[1]) / max(1., box[2] - box[0])

    candidates = []
    for i, upper in enumerate(rows):
        for j, lower in enumerate(rows):
            a, b = upper["bbox"], lower["bbox"]
            if i == j or (a[1] + a[3]) / 2 >= (b[1] + b[3]) / 2:
                continue
            if aspect(a) > STACK_MAX_PIECE_ASPECT or aspect(b) > STACK_MAX_PIECE_ASPECT or not _aligned(a, b) or _iou(a, b) > .3:
                continue
            shorter = min(a[3] - a[1], b[3] - b[1])
            gap = b[1] - a[3]  # negative when the pieces overlap
            if not -.35 * shorter <= gap <= .2 * shorter:
                continue
            joined = [min(a[0], b[0]), a[1], max(a[2], b[2]), b[3]]
            if STACK_ASPECT[0] <= aspect(joined) <= STACK_ASPECT[1]:
                candidates.append((abs(gap), i, j, joined))
    used, out = set(), []
    for _, i, j, joined in sorted(candidates):
        if i in used or j in used:
            continue
        used |= {i, j}
        best = max(rows[i], rows[j], key=lambda r: r["confidence"])
        out.append(dict(best, bbox=[round(v, 2) for v in joined], joined_from_pieces=True))
    return out + [row for index, row in enumerate(rows) if index not in used]


def adopt_roles(people, hinted, default="player"):
    """Give each person box the kind a football-specific detector saw there.

    `people` come from a general person detector, which finds people far more reliably but cannot say
    who is a goalkeeper or a referee. `hinted` are the football detector's own person boxes. A person
    takes the kind of a hinted box that lies mostly inside it; with none, the person is a player.
    """
    out = []
    for row in people:
        box = row["bbox"]
        inside = []
        for hint in hinted:
            area = _area(hint["bbox"])
            share = _overlap(hint["bbox"], box) / area if area > 0 else 0.
            if share >= .5:
                inside.append((_iou(box, hint["bbox"]) + share, hint["confidence"], hint["object_type"]))
        out.append(dict(row, object_type=max(inside)[2] if inside else default))
    return out


# One role per track. The football weights name a referee or goalkeeper in only some frames of a track, so the frames are
# pooled: on labelled SoccerNet clips, true referee tracks were called a referee in 17% of their frames and players in 0%.
ROLE_MIN_FRAMES = 5
REFEREE_SHARE = .15
REFEREE_SHARE_WITHOUT_KIT = .05  # enough when the track is also in neither kit group at least half the time
KITLESS_SHARE = .5
GOALKEEPER_SHARE = .1


def settle_roles(rows, default="player"):
    """{track_id: role} from each track's frame-by-frame role hints (`object_type`) and kit group (`team`).

    A track is a referee when enough of its frames were called one, or fewer were but it also matches neither
    kit group; a goalkeeper when enough frames were called one; otherwise `default`. On ten labelled clips this
    marked 36% of referee observations (49% on three held-out clips) and 0.15% of outfield players' as referees.
    """
    hints, kitless, total = defaultdict(Counter), Counter(), Counter()
    for row in rows:
        track = row.get("track_id")
        if track:
            hints[track][row["object_type"]] += 1
            kitless[track] += not row.get("team")
            total[track] += 1
    roles = {}
    for track, counts in hints.items():
        frames, referee, keeper = total[track], counts["referee"], counts["goalkeeper"]
        if referee >= ROLE_MIN_FRAMES and (referee / frames >= REFEREE_SHARE
                                           or (referee / frames >= REFEREE_SHARE_WITHOUT_KIT and kitless[track] / frames >= KITLESS_SHARE)):
            roles[track] = "referee"
        elif keeper >= ROLE_MIN_FRAMES and keeper / frames >= GOALKEEPER_SHARE:
            roles[track] = "goalkeeper"
        else:
            roles[track] = default
    return roles


def apply_roles(rows, roles):
    """Set each tracked person row to its track's settled role, keeping the frame's own hint as `role_hint`. Returns rows changed."""
    changed = 0
    for row in rows:
        role = roles.get(row.get("track_id"))
        if role and row.get("object_type") in PERSON_KINDS:
            row["role_hint"] = row["object_type"]
            changed += row["object_type"] != role
            row["object_type"] = role
    return changed
