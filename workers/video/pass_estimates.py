"""Passes estimated from the ball's path on the pitch map, for a person to check.

A pass is counted when one player had the ball, the ball was seen travelling, and a different player
then had it. Reaching a team-mate is "completed"; reaching the other kit group is "not completed".
A player "has" the ball while it stays beside them (a spell): it is close and slow enough to be at
their feet, and either moves with them, changes course at them, or matches a controlled touch
found in sequences.py.

This is a rule over tracked positions, not a trained event model, and it has not been checked
against labelled passes. It needs a pitch calibration, because the camera's own movement has to be
taken out of the ball's speed. Nothing here is counted as an event until a person approves it.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

VERSION = "pass-estimates-1"
REACH_M = 2.5  # the ball is at a player's feet, or being run with, inside this distance
CLOSE_M = 1.5  # a change of course counts as a touch only this close
MARGIN_M = .3  # the nearest player must be nearer than an opponent by this much
CARRY_MAX_SPEED = 9.  # m/s. A faster ball is in flight, whoever it is passing
SPEED_WINDOW_SECONDS = (.08, .24)  # the ball's speed is taken over a step of this length, to ride out detection jitter
MAX_SPEED = 45.  # a step faster than this is the path jumping between two candidates, not the ball
SPELL_GAP_SECONDS = .5
SPELL_MIN_SECONDS = .3
STEADY_M = 1.  # the ball's place relative to the player moves less than this while it is with them
TURN_DEGREES = 30.
TURN_SPEED_RATIO = .5
CARRY_GAP_SECONDS = 3.  # one player's spells this close together are one carry
MAX_PASS_SECONDS = 6.
MIN_PASS_M = 4.  # between the passer and the receiver. Closer than this is a tackle or a duel, listed elsewhere
TAKEN_PASSER_M = 5.  # the passer is still this close when an opponent has the ball
TAKEN_AFTER_SECONDS = .5
TAKEN_SECONDS = 1.5
FLIGHT_SPEED = 6.  # m/s
FLIGHT_MIN_SAMPLES = 2
LONG_SPEED = 12.  # a ball that leaves this fast, and is not seen to reach anyone, is listed apart
LONG_MIN_SAMPLES = 4
LONG_MIN_M = 10.
LONG_WINDOW_SECONDS = 1.5
LIMITATIONS = [
    "Rules over tracked positions on an estimated pitch map, not a trained event model. Not checked against labelled passes.",
    "A pass is counted only when the model saw both players with the ball and the ball travelling between them. Passes while the ball was not detected are missing, so the counts are low.",
    "A header or first-time touch between two players is usually not seen, so a pass can be credited to the wrong passer or receiver.",
    "Not completed means the next player seen with the ball was in the other kit group. A clearance, a blocked cross and a loose ball look the same as a misplaced pass.",
    "Teams are the model's kit-colour groups. When two players cross and the tracker swaps them, a pass is given to the wrong team.",
]


def _unavailable(reason):
    return dict(schema_version=VERSION, status="not_run", reason=reason, review_status="unreviewed", accuracy_evaluated=False,
                passes=[], played_long=[], counts=dict(completed=0, not_completed=0, outcome_not_seen=0), by_team=[], by_player=[], pairs=[],
                limitations=LIMITATIONS)


def _samples(evidence, pitch, resolved):
    """One record per frame whose ball can be placed on the pitch map, with every tracked player placed beside it."""
    from workers.video import pitch_keypoints as pk
    out = []
    for index, f in enumerate(evidence):
        ball, record = f["ball"], pitch.get(f.get("frame"))
        if not ball or not record or record.get("status") != "calibrated":
            continue
        people = [p for p in f["players"] if p["track"] and p["scene"] == ball["scene"] and p["confidence"] >= .25]
        xy, _ = pk.to_pitch(record, [(ball["x"], ball["y"])] + [(p["x"], p["y"]) for p in people])
        if np.isnan(xy[0]).any():
            continue
        players = {}
        for p, place in zip(people, xy[1:]):
            if not np.isnan(place).any():
                kit, team = resolved(p)
                players[p["track"]] = dict(xy=place, kit=kit, team=team, distance=float(np.linalg.norm(place - xy[0])))
        out.append(dict(index=index, time=f["time"], scene=ball["scene"], ball=xy[0], players=players, speed=None, velocity=None, owner=None))
    for i, s in enumerate(out):
        # Speed over a short step back (or forward, at the start of a run of detections).
        for step in (range(i - 1, -1, -1), range(i + 1, len(out))):
            other = next((out[j] for j in step if abs(out[j]["time"] - s["time"]) >= SPEED_WINDOW_SECONDS[0] - 1e-9), None)
            if other and abs(other["time"] - s["time"]) <= SPEED_WINDOW_SECONDS[1] + 1e-9 and other["scene"] == s["scene"]:
                velocity = (s["ball"] - other["ball"]) / (s["time"] - other["time"])
                if np.linalg.norm(velocity) <= MAX_SPEED:
                    s["velocity"], s["speed"] = velocity, float(np.linalg.norm(velocity))
                    break
    for s in out:
        ranked = sorted(s["players"].items(), key=lambda item: item[1]["distance"])
        if not ranked or ranked[0][1]["distance"] > REACH_M or (s["speed"] is not None and s["speed"] > CARRY_MAX_SPEED):
            continue
        track, near = ranked[0]
        rival = next((p for _, p in ranked[1:] if not (near["team"] and p["team"] == near["team"])), None)
        if rival is None or rival["distance"] - near["distance"] >= MARGIN_M:
            s["owner"] = track
    return out


def _mean_velocity(samples, start, end):
    """The ball's mean velocity over the samples with start <= time <= end, or None with too little to go on."""
    span = [s for s in samples if start - 1e-9 <= s["time"] <= end + 1e-9]
    if len(span) < 2 or span[-1]["time"] - span[0]["time"] < SPEED_WINDOW_SECONDS[0] - 1e-9:
        return None
    velocity = (span[-1]["ball"] - span[0]["ball"]) / (span[-1]["time"] - span[0]["time"])
    return velocity if np.linalg.norm(velocity) <= MAX_SPEED else None


def _spells(samples, anchors):
    """Stretches in which one player has the ball. Each is dict(track, first, last, samples, basis)."""
    runs, current = [], []
    for s in samples:
        if not s["owner"]:
            continue
        if current and (s["owner"] != current[-1]["owner"] or s["scene"] != current[-1]["scene"] or s["time"] - current[-1]["time"] > SPELL_GAP_SECONDS + 1e-9):
            runs.append(current)
            current = []
        current.append(s)
    if current:
        runs.append(current)
    by_index = {s["index"]: s for s in samples}
    touched = [(a["track_id"], set(a["indices"])) for a in anchors]
    spells = []
    for run in runs:
        track = run[0]["owner"]
        offsets = [s["ball"] - s["players"][track]["xy"] for s in run]
        duration = run[-1]["time"] - run[0]["time"]
        steady = any(run[j]["time"] - run[i]["time"] >= SPELL_MIN_SECONDS - 1e-9
                     and all(np.linalg.norm(offsets[k] - offsets[i]) <= STEADY_M for k in range(i, j + 1))
                     for i in range(len(run)) for j in range(i + 1, len(run)))
        turned = False
        if min(s["players"][track]["distance"] for s in run) <= CLOSE_M:
            before = _mean_velocity(samples, run[0]["time"] - .3, run[0]["time"])
            after = _mean_velocity(samples, run[-1]["time"], run[-1]["time"] + .3)
            if before is not None and after is not None:
                fast, slow = max(np.linalg.norm(before), np.linalg.norm(after)), min(np.linalg.norm(before), np.linalg.norm(after))
                if fast >= FLIGHT_SPEED:
                    cosine = float(before @ after) / max(1e-9, np.linalg.norm(before) * np.linalg.norm(after))
                    turned = cosine < math.cos(math.radians(TURN_DEGREES)) or slow / fast <= TURN_SPEED_RATIO
        anchored = any(owner == track and any(run[0]["index"] <= i <= run[-1]["index"] for i in indices) for owner, indices in touched)
        if steady or turned or anchored:
            spells.append(dict(track=track, samples=run, basis="controlled_touch" if anchored else "moves_with_player" if steady else "changes_course"))
    # A controlled touch found in the image that no spell covers still counts, where it can be placed on the pitch map.
    for owner, indices in touched:
        placed = [by_index[i] for i in sorted(indices) if i in by_index and owner in by_index[i]["players"]]
        if placed and not any(sp["track"] == owner and sp["samples"][0]["index"] <= placed[-1]["index"] and sp["samples"][-1]["index"] >= placed[0]["index"] for sp in spells):
            spells.append(dict(track=owner, samples=placed, basis="controlled_touch"))
    spells.sort(key=lambda sp: sp["samples"][0]["time"])
    merged = []
    for sp in spells:
        last = merged[-1] if merged else None
        if (last and last["track"] == sp["track"] and last["samples"][-1]["scene"] == sp["samples"][0]["scene"]
                and sp["samples"][0]["time"] - last["samples"][-1]["time"] <= CARRY_GAP_SECONDS):
            last["samples"] = last["samples"] + [s for s in sp["samples"] if s["time"] > last["samples"][-1]["time"]]
        else:
            merged.append(dict(sp))
    return merged


def _place(point):
    return dict(x=round(float(point[0]), 1), y=round(float(point[1]), 1))


def estimate_passes(evidence, anchors, resolved, metadata, context=None, pitch=None):
    """`evidence` and `anchors` are the per-frame records and controlled touches built in sequences.py; `pitch` is {frame: record}."""
    if not pitch:
        return _unavailable("No pitch calibration, so the ball's speed cannot be separated from the camera's movement.")
    samples = _samples(evidence, pitch, resolved)
    if not samples:
        return _unavailable("The ball could not be placed on the pitch map in any frame.")
    duration = metadata["duration_seconds"]
    stoppages = (context or {}).get("dead_ball_intervals", [])
    stopped = lambda start, end: any(x < end and y > start for x, y in stoppages)
    spells = _spells(samples, anchors)
    rejected = Counter()
    passes, long_balls, linked = [], [], set()

    def who(spell, sample):
        player = sample["players"][spell["track"]]
        return spell["track"], player["kit"], player["team"], player["xy"]

    for number, (a, b) in enumerate(zip(spells, spells[1:])):
        release, arrival = a["samples"][-1], b["samples"][0]
        if release["scene"] != arrival["scene"]:
            rejected["scene_cut"] += 1
            continue
        gap = arrival["time"] - release["time"]
        if gap <= 0 or gap > MAX_PASS_SECONDS:
            rejected["too_long_between_players"] += 1
            continue
        if stopped(release["time"], arrival["time"]):
            rejected["confirmed_stoppage"] += 1
            continue
        passer, from_kit, from_team, start = who(a, release)
        receiver, to_kit, to_team, end = who(b, arrival)
        if passer == receiver:
            rejected["same_player"] += 1
            continue
        if not from_team or not to_team:
            rejected["kit_group_unknown"] += 1
            continue
        length = float(np.linalg.norm(end - start))
        if length < MIN_PASS_M:
            rejected["players_too_close"] += 1
            continue
        flight = [s for s in samples if release["time"] < s["time"] < arrival["time"] and s["speed"] is not None and s["speed"] >= FLIGHT_SPEED]
        if len(flight) < FLIGHT_MIN_SAMPLES:
            rejected["ball_not_seen_travelling"] += 1
            continue
        completed = from_team == to_team
        # Half a second on, the ball is still beside the passer and an opponent is nearer to it: it was taken, not passed.
        # A pass struck past a pressing opponent is well clear by then. Tackles are listed elsewhere.
        if not completed and any(
                passer in s["players"] and s["players"][passer]["distance"] <= TAKEN_PASSER_M and any(
                    p["team"] and p["team"] != from_team and p["distance"] <= REACH_M and p["distance"] < s["players"][passer]["distance"]
                    for p in s["players"].values())
                for s in samples if release["time"] + TAKEN_AFTER_SECONDS <= s["time"] <= min(arrival["time"], release["time"] + TAKEN_SECONDS)):
            rejected["taken_by_an_opponent"] += 1
            linked.add(number)  # not a long ball either
            continue
        linked.add(number)
        passes.append(dict(
            type="possible_pass", timestamp_seconds=round(release["time"], 4), start_seconds=round(release["time"], 4), end_seconds=round(arrival["time"], 4),
            evidence_start_seconds=round(max(0, a["samples"][0]["time"] - .7, release["time"] - 2.5), 3), evidence_end_seconds=round(min(duration, arrival["time"] + 1.5), 3),
            scene_id=release["scene"], from_track=passer, to_track=receiver, from_kit=from_kit, to_kit=to_kit, from_team=from_team, to_team=to_team,
            outcome="completed" if completed else "not_completed", suggested_review_type="pass", suggested_outcome="completed" if completed else "incomplete",
            start_m=_place(start), end_m=_place(end), length_m=round(length, 1), seconds=round(gap, 2), flight_samples=len(flight),
            fastest_m_s=round(max(s["speed"] for s in flight), 1), passer_basis=a["basis"], receiver_basis=b["basis"],
            evidence_level="pitch_map_rule_review_only", possession_included=False, review_status="unreviewed", source=VERSION,
            evidence=("The first player had the ball, it was seen travelling, and a team-mate then had it." if completed
                      else "The first player had the ball, it was seen travelling, and a player in the other kit group then had it. It may be a misplaced pass, a clearance or a blocked cross.")))

    # A ball struck long that the model did not see reach anyone: listed, never counted as a pass.
    for number, a in enumerate(spells):
        if number in linked:
            continue
        release = a["samples"][-1]
        passer, from_kit, from_team, start = who(a, release)
        flight = [s for s in samples if 0 < s["time"] - release["time"] <= LONG_WINDOW_SECONDS and s["scene"] == release["scene"]
                  and s["speed"] is not None and s["speed"] >= LONG_SPEED]
        if len(flight) < LONG_MIN_SAMPLES or float(np.linalg.norm(flight[-1]["ball"] - start)) < LONG_MIN_M or stopped(release["time"], flight[-1]["time"]):
            continue
        long_balls.append(dict(timestamp_seconds=round(release["time"], 4), scene_id=release["scene"], from_track=passer, from_kit=from_kit, from_team=from_team,
                               start_m=_place(start), fastest_m_s=round(max(s["speed"] for s in flight), 1), outcome="outcome_not_seen",
                               evidence="The ball left this player fast and was not seen to reach anyone. It may be a long pass, a clearance or a shot."))

    for index, row in enumerate(passes):
        row["id"] = f"pass-{index + 1}"
    teams, players, pairs = {}, {}, Counter()
    for row in passes:
        team = teams.setdefault(row["from_team"], dict(team=row["from_team"], attempted=0, completed=0, not_completed=0, length_m=0.))
        maker = players.setdefault(row["from_track"], dict(track_id=row["from_track"], team=row["from_team"], attempted=0, completed=0, received=0))
        team["attempted"] += 1
        maker["attempted"] += 1
        team["length_m"] += row["length_m"]
        if row["outcome"] == "completed":
            team["completed"] += 1
            maker["completed"] += 1
            players.setdefault(row["to_track"], dict(track_id=row["to_track"], team=row["to_team"], attempted=0, completed=0, received=0))["received"] += 1
            pairs[row["from_team"], row["from_track"], row["to_track"]] += 1
        else:
            team["not_completed"] += 1
    by_team = [dict(team, completion_pct=round(100 * team["completed"] / team["attempted"], 1), mean_length_m=round(team.pop("length_m") / team["attempted"], 1))
               for _, team in sorted(teams.items())]
    return dict(
        schema_version=VERSION, status="estimated", review_status="unreviewed", accuracy_evaluated=False, passes=passes, played_long=long_balls,
        counts=dict(completed=sum(row["outcome"] == "completed" for row in passes), not_completed=sum(row["outcome"] == "not_completed" for row in passes),
                    outcome_not_seen=len(long_balls)),
        by_team=by_team, by_player=sorted(players.values(), key=lambda p: (-p["attempted"], -p["received"], p["track_id"])),
        pairs=[dict(team=team, from_track=a, to_track=b, passes=n) for (team, a, b), n in sorted(pairs.items(), key=lambda item: -item[1])],
        diagnostics=dict(ball_samples=len(samples), spells=len(spells), spell_basis=dict(Counter(sp["basis"] for sp in spells)), rejected=dict(rejected)),
        parameters=dict(reach_m=REACH_M, carry_max_speed_m_s=CARRY_MAX_SPEED, spell_min_seconds=SPELL_MIN_SECONDS, max_pass_seconds=MAX_PASS_SECONDS,
                        min_pass_m=MIN_PASS_M, flight_speed_m_s=FLIGHT_SPEED, long_ball_speed_m_s=LONG_SPEED),
        limitations=LIMITATIONS)
