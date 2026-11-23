"""Pairs of tracked identities that could be one player, offered to a person to check.

The automatic linker joins fragments only when the evidence is clear. This looks at what
it left apart and lists pairs where one identity ends and another begins soon after,
close enough that one person could have covered the ground. A suggestion is a question
for a reviewer, never a merge: nothing here changes an identity or any figure.

Where both ends fall in frames with a pitch calibration, closeness is judged in estimated
metres. Otherwise it is judged in the image, in player heights, without correcting for
camera movement, over a much shorter gap.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from workers.video import pitch_keypoints as pk

VERSION = "merge-suggestions-2"
PEOPLE = {"player", "goalkeeper"}
MIN_OBSERVATIONS = 5
END_SAMPLES = 3  # an identity's end and start positions are the median of this many samples
MAX_GAP_S = 10.
SLACK_M = 3.  # allowed miss at zero gap: position error at both ends
SPEED_M_S = 7.  # a player can cover this much per second of gap
MAX_DISTANCE_M = 25.
IMAGE_MAX_GAP_S = 3.
IMAGE_SLACK_HEIGHTS = .6
IMAGE_HEIGHTS_PER_S = 1.5
MAX_HEIGHT_RATIO = 1.6
MAX_SUGGESTIONS = 60
# A pair whose distance is at most this share of what one player could cover in the gap is a close fit.
# On ten SoccerNet test clips close fits were right about half the time (19 of 40 checked), all suggestions 43 of 224;
# how many other identities fit the same gap did not predict it. See docs/testing/SOCCERNET_GSR_2026-11-15.md.
CLOSE_FIT = .2
LIMITATIONS = ["Suggestions are questions for a person watching the footage, not merges. Nothing is joined automatically.",
               "They rest on timing and position only. Team-mates in the same kit who cross near each other look alike here.",
               "On ten labelled SoccerNet test clips, about half of the close-fit suggestions that could be checked joined the same player (19 of 40), and about one in five of all suggestions (43 of 224).",
               "Distances in metres are estimates from automatic pitch calibration, which was off by several metres on those test clips.",
               "Where no pitch calibration covers both ends, distance is judged in the image without correcting for camera movement.",
               "A player who is off screen for longer than the gap limit is not suggested."]


def _identities(rows):
    grouped = defaultdict(list)
    for row in rows:
        identity = row.get("linked_track_id") or row.get("track_id")
        if identity and row.get("object_type") in PEOPLE | {"referee", "person"}:
            grouped[identity].append(row)
    out = {}
    for identity, items in grouped.items():
        kind = Counter(row["object_type"] for row in items).most_common(1)[0][0]
        if kind not in PEOPLE or len(items) < MIN_OBSERVATIONS:
            continue
        items.sort(key=lambda row: row["frame_idx"])
        teams = Counter(row.get("linked_team") or row.get("team") for row in items if row.get("linked_team") or row.get("team"))
        team = teams.most_common(1)[0][0] if teams and teams.most_common(1)[0][1] / sum(teams.values()) >= .7 else None
        out[identity] = dict(track_id=identity, rows=items, frames={row["frame_idx"] for row in items}, first=items[0]["frame_idx"],
                             last=items[-1]["frame_idx"], scene=items[0].get("scene_id", 0), team=team)
    return out


def _place(items, pitch):
    """(pitch metres or None, image foot point, box height) as medians over a few samples at one end of an identity."""
    feet = np.array([((row["bbox"][0] + row["bbox"][2]) / 2, row["bbox"][3]) for row in items], dtype=float)
    height = float(np.median([row["bbox"][3] - row["bbox"][1] for row in items]))
    placed, tried = [], 0
    for row, foot in zip(items, feet):
        record = (pitch or {}).get(row["frame_idx"])
        if not record:
            continue  # calibration may be tried on every Nth frame only; an untried frame says nothing
        tried += 1
        if record.get("status") == "calibrated":
            xy, usable = pk.to_pitch(record, [foot])
            if usable[0]:
                placed.append(xy[0])
    # Metres are used only when every sample that was tried at this end was placed; a partly placed end is not trusted.
    metres = np.median(placed, axis=0) if tried and len(placed) == tried else None
    return metres, np.median(feet, axis=0), height


def suggest(rows, fps, pitch=None):
    """Suggested pairs for review, best first. `pitch` is {frame_idx: calibration record} or None."""
    base = dict(schema_version=VERSION, review_status="unreviewed", limitations=LIMITATIONS,
                parameters=dict(max_gap_seconds=MAX_GAP_S, slack_m=SLACK_M, speed_m_s=SPEED_M_S, max_distance_m=MAX_DISTANCE_M,
                                image_max_gap_seconds=IMAGE_MAX_GAP_S, image_slack_heights=IMAGE_SLACK_HEIGHTS,
                                image_heights_per_second=IMAGE_HEIGHTS_PER_S, max_height_ratio=MAX_HEIGHT_RATIO, min_observations=MIN_OBSERVATIONS, close_fit=CLOSE_FIT),
                evaluation=dict(dataset="SoccerNet Game State Reconstruction, 10 test clips", close_fit_right=19, close_fit_checked=40, all_right=43, all_checked=224))
    identities = _identities(rows)
    if not fps or len(identities) < 2:
        return dict(base, suggestions=[], suggestions_total=0)
    for identity in identities.values():
        identity["start"] = _place(identity["rows"][:END_SAMPLES], pitch)
        identity["end"] = _place(identity["rows"][-END_SAMPLES:], pitch)
    order = sorted(identities.values(), key=lambda identity: identity["first"])
    candidates = []
    for a in order:
        for b in order:
            gap = (b["first"] - a["last"]) / fps
            if gap <= 0 or gap > MAX_GAP_S or a["scene"] != b["scene"] or (a["team"] and b["team"] and a["team"] != b["team"]):
                continue
            if a["frames"] & b["frames"]:
                continue
            (end_m, end_px, end_h), (start_m, start_px, start_h) = a["end"], b["start"]
            if end_m is not None and start_m is not None:
                distance = float(np.linalg.norm(start_m - end_m))
                reach = SLACK_M + SPEED_M_S * gap
                if distance > min(reach, MAX_DISTANCE_M):
                    continue
                detail = dict(basis="pitch", distance_m=round(distance, 1), distance_heights=None)
            else:
                height = (end_h + start_h) / 2
                if gap > IMAGE_MAX_GAP_S or height <= 0 or max(end_h, start_h) / max(1e-6, min(end_h, start_h)) > MAX_HEIGHT_RATIO:
                    continue
                distance = float(np.linalg.norm(start_px - end_px)) / height
                reach = IMAGE_SLACK_HEIGHTS + IMAGE_HEIGHTS_PER_S * gap
                if distance > reach:
                    continue
                detail = dict(basis="image", distance_m=None, distance_heights=round(distance, 2))
            candidates.append(dict(from_track=a["track_id"], to_track=b["track_id"], gap_seconds=round(gap, 2),
                                   from_last_seen_seconds=round(a["last"] / fps, 2), to_first_seen_seconds=round(b["first"] / fps, 2),
                                   same_kit=(a["team"] == b["team"]) if a["team"] and b["team"] else None, fit=round(distance / reach, 3),
                                   close_fit=distance / reach <= CLOSE_FIT, **detail))
    ends, starts = Counter(c["from_track"] for c in candidates), Counter(c["to_track"] for c in candidates)
    for candidate in candidates:
        # Other identities that would fit the same end or the same start equally plausibly on these rules.
        candidate["rivals"] = ends[candidate["from_track"]] + starts[candidate["to_track"]] - 2
    # Closest fit first: on labelled clips that predicted a right suggestion; the number of rivals did not.
    candidates.sort(key=lambda c: (c["fit"], c["rivals"], c["basis"] != "pitch", c["from_track"], c["to_track"]))
    return dict(base, suggestions=candidates[:MAX_SUGGESTIONS], suggestions_total=len(candidates), close_fit_total=sum(c["close_fit"] for c in candidates))


def appearances(rows, fps, max_gap_seconds=1.):
    """{identity: [[start_s, end_s], ...]} for the stretches each tracked person is on screen; gaps up to a second are joined."""
    out = {}
    if not fps:
        return out
    for identity in _identities(rows).values():
        frames = sorted(identity["frames"])
        spans = [[frames[0], frames[0]]]
        for frame in frames[1:]:
            if (frame - spans[-1][1]) / fps <= max_gap_seconds:
                spans[-1][1] = frame
            else:
                spans.append([frame, frame])
        out[identity["track_id"]] = [[round(a / fps, 2), round(b / fps, 2)] for a, b in spans]
    return out


def suggest_for_run(directory, manifest):
    """Suggestions from a run's stored detections and calibration. CPU only; no model, no video."""
    import json
    from pathlib import Path
    directory = Path(directory)
    if not (directory / "tracking.json").is_file():
        return dict(schema_version=VERSION, review_status="unreviewed", limitations=LIMITATIONS, suggestions=[], suggestions_total=0,
                    reason="The detections for this clip are not stored, so nothing can be suggested.")
    pitch = None
    if (directory / "pitch-calibration.json").is_file():
        frames = json.loads((directory / "pitch-calibration.json").read_text(encoding="utf-8")).get("frames", {})
        pitch = {int(index): record for index, record in frames.items()}
    rows = json.loads((directory / "tracking.json").read_text(encoding="utf-8"))
    fps = manifest.get("metadata", {}).get("fps")
    out = dict(suggest(rows, fps, pitch), run_id=manifest.get("run_id"), appearances=appearances(rows, fps))
    # Shirt-number proposals exist only where an operator has run the experimental reader on this clip.
    if (directory / "shirt-numbers.json").is_file():
        numbers = json.loads((directory / "shirt-numbers.json").read_text(encoding="utf-8"))
        out["shirt_numbers"] = dict(reader=numbers.get("reader"), review_status="unreviewed", limitations=numbers.get("limitations", []),
                                    proposals=[{key: p.get(key) for key in ("track_id", "proposed_number", "agreement", "confident_reads", "crops_read")}
                                               for p in numbers.get("proposals", []) if p.get("proposed_number")],
                                    # Every identity with confident reads, for matching against a line-up; not shown as proposals.
                                    reads=[{key: p.get(key) for key in ("track_id", "proposed_number", "confident_reads", "crops_read", "reads")}
                                           for p in numbers.get("proposals", []) if p.get("reads")])
    return out
