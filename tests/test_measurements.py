import pytest

from workers.video.measurements import VideoMeasurements, from_tracking


META = dict(width=1000, height=600, fps=2, frame_count=4, duration_seconds=2)


def player(x=100, identity="T1"):
    return dict(object_type="player", track_id=identity, bbox=[x-10, 100, x+10, 200], confidence=.9)


def ball(x=100):
    return dict(object_type="ball", track_id=None, bbox=[x-2, 198, x+2, 202], confidence=.7)


def test_known_geometry_visibility_gaps_and_proximity():
    calc = VideoMeasurements(META, 2)
    frames = [[player(), ball()], [player(110)], [], [player(130), ball(800)]]
    rows = []
    for i, frame in enumerate(frames):
        calc.observe(i, frame)
        rows.extend(dict(row, frame_idx=i) for row in frame)
    result = calc.finish()
    assert result["mean_visible_players"] == .75
    assert result["ball_coverage_pct"] == 50
    assert result["longest_ball_detection_gap_seconds"] == 1
    assert result["ball_proximity_frames"] == 1
    track = result["tracks"][0]
    assert track["observed_seconds"] == 1.5
    assert track["visibility_pct"] == 75
    assert track["image_displacement_px"] == 30
    assert sum(map(sum, result["image_occupancy_grid"])) == 3
    assert from_tracking(rows, META, dict(sample_fps=2, decoded_frames=4)) == result
    assert "possession" not in result


def test_ambiguous_ball_and_nearby_players_do_not_get_associated():
    calc = VideoMeasurements(META, 2)
    calc.observe(0, [player(), ball(), ball(110)])
    calc.observe(1, [player(), player(102, "T2"), ball(101)])
    assert calc.finish()["ball_proximity_frames"] == 0
    assert calc.finish()["ambiguous_ball_frames"] == 1


def test_large_jump_and_tracking_gap_excluded():
    calc = VideoMeasurements(dict(META, duration_seconds=10, frame_count=20), 2)
    calc.observe(0, [player()])
    calc.observe(1, [player(900)])
    calc.observe(10, [player(910)])
    track = calc.finish()["tracks"][0]
    assert track["image_displacement_px"] == 0
    assert track["excluded_movement_steps"] == 2


def test_empty_observations_keep_unknown_confidence_and_zero_coverage():
    calc = VideoMeasurements(META, 2)
    for i in range(4):
        calc.observe(i, [])
    result = calc.finish()
    assert result["ball_coverage_pct"] == 0
    assert result["longest_ball_detection_gap_seconds"] == 2
    assert result["mean_ball_confidence"] is None
    assert result["tracks"] == []


def test_same_player_returning_after_ball_gap_is_not_a_transfer():
    meta = dict(META, fps=25, frame_count=25, duration_seconds=1)
    calc = VideoMeasurements(meta, 25)
    for i in range(5):
        calc.observe(i, [dict(player(), team='kit-A'), ball()])
    for i in range(5, 10):
        calc.observe(i, [])
    calc.observe(10, [dict(player(160), team='kit-A'), ball(160)])
    assert calc.finish()['event_candidates'] == []
    # A distinct nearby track is only a review candidate, never an approved pass.
    other = VideoMeasurements(meta, 25)
    for i in range(5):
        other.observe(i, [dict(player(), team='kit-A'), ball()])
    other.observe(6, [dict(player(160, 'T2'), team='kit-A'), ball(160)])
    candidate = other.finish()['event_candidates'][0]
    assert candidate['from_track'] == 'T1' and candidate['to_track'] == 'T2'
    assert candidate['review_status'] == 'unreviewed'
