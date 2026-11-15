import numpy as np

from workers.video import pitch_keypoints as pk
from workers.video.merge_suggestions import appearances, suggest

WIDTH, HEIGHT, FPS = 1920, 1080, 10.
SCALE, OFFSET = 10., np.array([300., 200.])  # synthetic view: 10 px per metre


def record():
    xy = pk.landmarks() * SCALE + OFFSET
    seen = (xy[:, 0] >= 0) & (xy[:, 0] < WIDTH) & (xy[:, 1] >= 0) & (xy[:, 1] < HEIGHT)
    return pk.fit(xy, np.where(seen, .9, 0.), WIDTH, HEIGHT)


def track(identity, frames, x_m, y_m, team="kit-A", kind="player", speed=0.):
    rows = []
    for frame in frames:
        x, y = np.array([x_m + speed * frame / FPS, y_m]) * SCALE + OFFSET
        rows.append(dict(frame_idx=frame, linked_track_id=identity, track_id="raw-" + identity, object_type=kind, team=team, scene_id=0,
                         bbox=[float(x) - 10, float(y) - 40, float(x) + 10, float(y)]))
    return rows


def pairs(out):
    return [(s["from_track"], s["to_track"]) for s in out["suggestions"]]


def test_an_identity_that_ends_where_another_soon_begins_is_suggested_in_metres():
    rows = track("L1", range(0, 30), 40, 30) + track("L2", range(40, 70), 43, 31) + track("L3", range(40, 70), 90, 60)
    out = suggest(rows, FPS, {i: record() for i in range(80)})
    assert pairs(out) == [("L1", "L2")]
    only = out["suggestions"][0]
    assert only["basis"] == "pitch" and abs(only["distance_m"] - 3.2) < .2 and only["gap_seconds"] == 1.1
    assert only["rivals"] == 0 and only["same_kit"] is True and only["from_last_seen_seconds"] == 2.9
    assert out["review_status"] == "unreviewed"


def test_appearances_join_short_gaps_only():
    rows = track("L1", list(range(0, 10)) + list(range(15, 20)) + list(range(60, 70)), 40, 30)
    assert appearances(rows, FPS) == {"L1": [[0., 1.9], [6., 6.9]]}


def test_identities_on_screen_together_other_kits_and_referees_are_never_suggested():
    together = track("L1", range(0, 30), 40, 30) + track("L2", range(29, 60), 41, 30)
    assert pairs(suggest(together, FPS, {i: record() for i in range(80)})) == []
    kits = track("L1", range(0, 30), 40, 30) + track("L2", range(35, 60), 41, 30, team="kit-B")
    assert pairs(suggest(kits, FPS, {i: record() for i in range(80)})) == []
    referee = track("L1", range(0, 30), 40, 30) + track("L2", range(35, 60), 41, 30, kind="referee")
    assert pairs(suggest(referee, FPS, {i: record() for i in range(80)})) == []
    brief = track("L1", range(0, 30), 40, 30) + track("L2", range(35, 38), 41, 30)
    assert pairs(suggest(brief, FPS, {i: record() for i in range(80)})) == []


def test_too_far_for_the_gap_or_too_long_a_gap_is_not_suggested():
    far = track("L1", range(0, 30), 40, 30) + track("L2", range(35, 60), 60, 30)  # 20 m in 0.6 s
    assert pairs(suggest(far, FPS, {i: record() for i in range(80)})) == []
    late = track("L1", range(0, 30), 40, 30) + track("L2", range(140, 170), 41, 30)  # 11 s later
    assert pairs(suggest(late, FPS, {i: record() for i in range(200)})) == []


def test_rivals_are_counted_and_the_clearest_pair_comes_first():
    rows = (track("L1", range(0, 30), 40, 30) + track("L2", range(40, 70), 41, 30) + track("L3", range(40, 70), 44, 32)
            + track("L4", range(0, 30), 80, 50) + track("L5", range(40, 70), 81, 50))
    out = suggest(rows, FPS, {i: record() for i in range(80)})
    assert pairs(out)[0] == ("L4", "L5") and out["suggestions"][0]["rivals"] == 0
    rivals = {(s["from_track"], s["to_track"]): s["rivals"] for s in out["suggestions"]}
    assert rivals["L1", "L2"] == 1 and rivals["L1", "L3"] == 1


def test_without_calibration_the_image_is_used_over_a_shorter_gap_and_says_so():
    rows = track("L1", range(0, 30), 40, 30) + track("L2", range(40, 70), 42, 30)  # 20 px = half a player height
    out = suggest(rows, FPS, None)
    assert pairs(out) == [("L1", "L2")] and out["suggestions"][0]["basis"] == "image" and out["suggestions"][0]["distance_m"] is None
    later = track("L1", range(0, 30), 40, 30) + track("L2", range(70, 99), 42, 30)  # 4.1 s: too long without metres
    assert pairs(suggest(later, FPS, None)) == []
    # One end uncalibrated: metres are not trusted for the pair.
    half = suggest(rows, FPS, {i: record() for i in range(30)})
    assert half["suggestions"][0]["basis"] == "image"


def test_calibration_tried_on_every_other_frame_still_gives_metres_but_a_failed_frame_does_not():
    rows = track("L1", range(0, 30), 40, 30) + track("L2", range(40, 70), 43, 31)
    every_other = {i: record() for i in range(0, 80, 2)}
    assert suggest(rows, FPS, every_other)["suggestions"][0]["basis"] == "pitch"
    failed = dict(every_other)
    failed[28] = dict(status="invalid", reason="No pitch found in the frame", landmarks_seen=0)
    assert suggest(rows, FPS, failed)["suggestions"][0]["basis"] == "image"
