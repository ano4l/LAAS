import math

from workers.video import gsr_benchmark as gb

SCALE = .05  # metres per pixel in the synthetic camera: 105 x 68 m is 2100 x 1360 px
RECORD = dict(status="calibrated", matrix=[[SCALE, 0, 0], [0, SCALE, 0], [0, 0, 1]])


def _labels(players, frames=75, flip_x=False):
    """Labelled clip: players move along x at their own speed; positions centred on the halfway spot like SoccerNet's."""
    images, annotations = [], []
    for f in range(frames):
        image_id = f"9{f:06d}"
        images.append(dict(image_id=image_id, file_name=f"{f + 1:06d}.jpg", is_labeled=True))
        for track, (x0, y, speed, team, jersey) in players.items():
            x = x0 + speed * f / 25
            foot = (x / SCALE, y / SCALE)
            px = -(x - 52.5) if flip_x else x - 52.5
            annotations.append(dict(image_id=image_id, track_id=track, supercategory="object",
                                    attributes=dict(role="player", team=team, jersey=jersey if f % 2 else None),
                                    bbox_image=dict(x=foot[0] - 20, y=foot[1] - 80, w=40, h=80),
                                    bbox_pitch=dict(x_bottom_middle=px, y_bottom_middle=y - 34)))
    return gb.load_labels(dict(info=dict(frame_rate=25, name="SNGS-test"), images=images, annotations=annotations))


def _rows(labels, identities):
    rows = []
    for frame, people in labels["frames"].items():
        for p in people:
            ident = identities(p["track"], frame)
            if ident is None:
                continue
            rows.append(dict(frame_idx=frame, object_type="player", bbox=list(p["box"]), confidence=.9, track_id=ident,
                             linked_track_id=ident, linked_team="scene-0/kit-A" if p["team"] == "left" else "scene-0/kit-B", scene_id=0))
    return rows


def test_exact_positions_score_zero_error_and_equal_distance():
    labels = _labels({1: (30., 20., 3., "left", "10"), 2: (60., 40., 6., "right", "7")})
    rows = _rows(labels, lambda track, frame: f"L{track}")
    report = gb.score(rows, labels, {f: RECORD for f in range(75)})
    assert report["detection"]["recall"] == 1 and report["detection"]["precision"] == 1
    assert report["position_error"]["median_m"] == 0 and report["calibration_only_error"]["median_m"] == 0
    assert report["calibration"]["orientation"] == [1, 1] and report["calibration"]["frames_with_swapped_orientation"] == 0
    motion = report["motion"]
    assert motion["ours"]["distance_m"] == motion["labelled"]["distance_m"] > 0
    assert math.isclose(motion["labelled"]["mean_speed_m_s"], 4.5, abs_tol=.05)  # two players, 3 and 6 m/s, measured equally long
    assert motion["distance_ratio_median"] == 1
    assert report["identities"]["following_a_player"] == 2 and report["identities"]["identities_per_followed_player"] == 1
    assert report["kits"]["share"] == 1


def test_mirrored_labels_are_aligned_once_per_clip():
    labels = _labels({1: (30., 20., 3., "left", None)}, flip_x=True)
    report = gb.score(_rows(labels, lambda track, frame: "L1"), labels, {f: RECORD for f in range(75)})
    assert report["calibration"]["orientation"] == [-1, 1]
    assert report["position_error"]["median_m"] == 0


def test_uncalibrated_frames_are_not_placed():
    labels = _labels({1: (30., 20., 3., "left", None)})
    pitch = {f: (RECORD if f < 40 else dict(status="invalid", reason="too few landmarks")) for f in range(75)}
    report = gb.score(_rows(labels, lambda track, frame: "L1"), labels, pitch)
    assert report["calibration"]["share"] == round(40 / 75, 3)
    assert report["position_error"]["count"] == 40


def test_suggestions_are_judged_by_the_labelled_player_both_identities_follow():
    labels = _labels({1: (30., 20., 3., "left", "10"), 2: (60., 40., 6., "right", "7")})
    # Player 1 is split at frame 40 (and missing 30-39); player 2 is one identity throughout.
    rows = _rows(labels, lambda track, frame: ("L2" if track == 2 else None if 30 <= frame < 40 else "L1a" if frame < 30 else "L1b"))
    suggestions = dict(suggestions=[dict(from_track="L1a", to_track="L1b", gap_seconds=.4, basis="pitch", distance_m=1.2, fit=.2, rivals=0),
                                    dict(from_track="L1a", to_track="L2", gap_seconds=.4, basis="pitch", distance_m=20., fit=.9, rivals=1),
                                    dict(from_track="L1a", to_track="L9", gap_seconds=.4, basis="pitch", distance_m=2., fit=.5, rivals=1)])
    report = gb.score(rows, labels, {f: RECORD for f in range(75)}, suggestions)
    judged = report["suggestions"]
    assert (judged["same"], judged["different"], judged["unknown"]) == (1, 1, 1)
    assert judged["true_pairs"] == 1 and judged["true_pairs_suggested"] == 1
    assert report["identities"]["identities_per_followed_player"] == 1.5


def test_shirt_number_proposals_are_judged_against_the_labelled_number():
    labels = _labels({1: (30., 20., 3., "left", "10"), 2: (60., 40., 6., "right", "7")})
    numbers = dict(proposals=[dict(track_id="L1", proposed_number="10", confident_reads=9, agreement=.9),
                              dict(track_id="L2", proposed_number="1", confident_reads=5, agreement=.8),
                              dict(track_id="L3", proposed_number=None)])
    report = gb.score(_rows(labels, lambda track, frame: f"L{track}"), labels, {f: RECORD for f in range(75)}, None, numbers)
    shirts = report["shirt_numbers"]
    assert (shirts["proposals"], shirts["right"], shirts["wrong"]) == (2, 1, 1)
    assert {s["track_id"]: s["digits"] for s in shirts["items"]} == {"L1": 2, "L2": 1}
