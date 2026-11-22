"""The nearest-player shortlist for tagging: read from stored detections, never a decision."""
import json

import pytest

from workers.video.tagging_hints import load_frames, near_ball


def person(frame, track, x, kind="player"):
    return dict(frame_idx=frame, linked_track_id=track, track_id="raw", object_type=kind, bbox=[x - 10, 100, x + 10, 200], confidence=.9)


def ball(frame, x, **extra):
    return dict(dict(frame_idx=frame, object_type="ball", bbox=[x - 2, 196, x + 2, 200], confidence=.8, ball_selected=True), **extra)


LABELS = dict(players=[dict(id="P1", name="Ada Obi", number="8", tracks=["L1", "L4"], team="home", lineup_id="h8"),
                       dict(id="P2", name="Typed Name", number=None, tracks=["L2"], team=None, lineup_id=None)])


def frames(rows, tmp_path):
    path = tmp_path / "tracking.json"
    path.write_text(json.dumps(rows))
    return load_frames(path)


def test_named_and_unnamed_identities_are_listed_nearest_first(tmp_path):
    rows = [person(50, "L1", 100), person(50, "L2", 160), person(50, "L3", 230), person(50, "L9", 900), person(50, "R1", 105, kind="referee"), ball(50, 110)]
    out = near_ball(frames(rows, tmp_path), 25, 2.0, LABELS)
    assert out["status"] == "ok" and out["frame_seconds"] == 2.0
    assert [(c["track_id"], c["name"], c["team"], c["named"]) for c in out["candidates"]] == [("L1", "Ada Obi", "home", True), ("L2", "Typed Name", None, True), ("L3", None, None, False)]
    assert out["candidates"][0]["distance_heights"] == .1 and out["candidates"][0]["lineup_id"] == "h8"


def test_the_nearest_frame_with_a_ball_is_used_within_the_window(tmp_path):
    rows = [person(50, "L1", 100), person(56, "L1", 100), ball(56, 100), person(80, "L1", 100), ball(80, 100)]
    data = frames(rows, tmp_path)
    assert near_ball(data, 25, 2.0, LABELS)["frame_seconds"] == 2.24          # frame 50 has no ball; frame 56 is within 0.4 s
    late = near_ball(data, 25, 2.72, LABELS)
    assert late["status"] == "no_ball" and late["candidates"] == []           # frames 56 and 80 are both 0.48 s away


def test_no_tracking_no_player_and_ambiguous_ball_give_a_reason_not_a_guess(tmp_path):
    assert near_ball(None, 25, 1.0, LABELS)["status"] == "no_tracking"
    assert load_frames(tmp_path / "missing.json") is None
    alone = near_ball(frames([person(50, "L1", 900), ball(50, 100)], tmp_path), 25, 2.0, LABELS)
    assert alone["status"] == "no_player_near" and alone["candidates"] == []
    # Two selected balls, or one too unsure, is not a ball position.
    two = tmp_path / "two"; two.mkdir()
    assert near_ball(frames([person(50, "L1", 100), ball(50, 100), ball(50, 400)], two), 25, 2.0, LABELS)["status"] == "no_ball"
    weak = tmp_path / "weak"; weak.mkdir()
    assert near_ball(frames([person(50, "L1", 100), ball(50, 100, confidence=.1)], weak), 25, 2.0, LABELS)["status"] == "no_ball"
    with pytest.raises(ValueError):
        near_ball({}, 25, -1, LABELS)
