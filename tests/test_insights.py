import copy
import json

import pytest

from workers.video.insights import VideoInsights, ensure_insights, from_tracking

META = dict(width=1000, height=600, fps=10, frame_count=10, duration_seconds=1)


def player(x, track="T1", kit="kit-A", scene=0):
    return dict(object_type="player", track_id=track, bbox=[x-10, 100, x+10, 200],
                confidence=.9, team=kit, scene_id=scene, team_assignment_confidence=.8)


def ball(x, scene=0, **extra):
    return dict(object_type="ball", bbox=[x-2, 198, x+2, 202], confidence=.8,
                scene_id=scene, ball_selected=True, **extra)


def calculate(frames, meta=None):
    meta = meta or dict(META, frame_count=len(frames), duration_seconds=len(frames)/10)
    calc = VideoInsights(meta, meta["fps"])
    for i, rows in enumerate(frames):
        calc.observe(i, rows)
    return calc.finish()


def test_empty_clip_is_unknown_not_even_possession_or_zero_confidence():
    out = calculate([[] for _ in range(10)])
    assert out["control"]["unknown_seconds"] == 1
    assert out["control"]["coverage_pct"] == 0
    assert out["control"]["by_kit"] == []
    assert out["ball"]["image_thirds_pct"] == [None, None, None]
    assert out["quality"]["kit_assignment_coverage_pct"] is None
    assert out["candidates"] == out["kits"] == out["bookmarks"] == []
    assert out["quality"]["accuracy_evaluated"] is False


def test_sustained_control_denominator_retains_unknown_and_anonymous_kit():
    out = calculate([[player(100), ball(100)]]*4+[[]]*6)
    c = out["control"]
    assert c["estimated_control_seconds"] == .4
    assert c["unknown_seconds"] == .6
    assert c["coverage_pct"] == 40
    assert c["by_kit"] == [dict(kit="scene-0/kit-A", estimated_seconds=.4, share_of_known_control_pct=100, clip_share_pct=40)]
    assert c["spells"][0]["review_status"] == "unreviewed"
    assert not any(k in out for k in ["home", "away", "approved_events", "xg"])


@pytest.mark.parametrize("last_kit,expected", [("kit-A", "possible_pass"), ("kit-B", "possible_turnover")])
def test_candidate_needs_sustained_distinct_controllers_and_observed_bridge(last_kit, expected):
    ps = [player(100), player(400, "T2", last_kit)]
    out = calculate([ps+[ball(x)] for x in [100]*4+[200, 300]+[400]*4])
    assert len(out["control"]["spells"]) == 2
    assert len(out["candidates"]) == 1
    candidate = out["candidates"][0]
    assert candidate["type"] == expected
    assert candidate["observed_bridge_frames"] == 2
    assert candidate["from_track"] == "T1" and candidate["to_track"] == "T2"
    assert candidate["review_status"] == "unreviewed"


def test_missing_ball_in_bridge_and_same_track_return_cannot_create_pass():
    ps = [player(100), player(400, "T2")]
    out = calculate([ps+[ball(100)]]*4+[ps]*2+[ps+[ball(400)]]*4)
    assert len(out["control"]["spells"]) == 2
    assert out["candidates"] == []
    again = calculate([[player(100), ball(100)]]*4+[[]]*2+[[player(400), ball(400)]]*4)
    assert len(again["control"]["spells"]) == 2
    assert again["candidates"] == []


def test_cut_and_group_label_changes_do_not_merge_control_or_kit_identity():
    out = calculate([[player(100), ball(100)]]*4+[[player(300, scene=1), ball(300, scene=1)]]*4)
    assert [s["scene_id"] for s in out["control"]["spells"]] == [0, 1]
    assert {g["kit"] for g in out["kits"]} == {"scene-0/kit-A", "scene-1/kit-A"}
    assert out["candidates"] == []
    assert out["ball"]["path"][3]["segment_id"] != out["ball"]["path"][4]["segment_id"]


def test_flybys_ambiguous_players_and_outside_ground_never_become_control():
    fleeting = calculate([[player(100), ball(100)]]*2+[[]]*8)
    ambiguous = calculate([[player(100), player(102, "T2"), ball(101)]]*10)
    aerial = calculate([[player(100), ball(100, field_plausibility="outside_ground_region")]]*10)
    for out in [fleeting, ambiguous, aerial]:
        assert out["control"]["spells"] == []
        assert out["candidates"] == []
    assert aerial["ball"]["observed_seconds"] == 1  # observed ball still useful, without invented ground control


def test_selection_ambiguity_and_confidence_gate_do_not_promote_raw_candidates():
    low = dict(ball(100), confidence=.2)
    for rows in [[player(100), dict(ball(100), ball_selected=False)], [player(100), ball(100), ball(110)], [player(100), low]]:
        out = calculate([rows]*10)
        assert out["ball"]["path"] == []
        assert out["control"]["coverage_pct"] == 0


def test_identity_split_at_same_position_and_jump_do_not_create_candidate():
    out = calculate([[player(100), ball(100)]]*4+[[player(102, "T2"), ball(102)]]*4)
    assert out["candidates"] == []
    jump = calculate([[player(100), ball(100)]]*4+[[player(900), ball(900)]]*4)
    assert len(jump["control"]["spells"]) == 2
    assert jump["candidates"] == []
    assert jump["ball"]["excluded_motion_steps"] == 1


def test_full_tracking_backfill_counts_empty_frames_and_last_partial_interval():
    meta = dict(META, fps=25, frame_count=26, duration_seconds=26/25)
    rows = [dict(r, frame_idx=i) for i in [0, 5, 10] for r in [player(100), ball(100)]]
    out = from_tracking(list(reversed(rows)), meta, dict(sample_fps=5, decoded_frames=26))
    assert out["analyzed_frames"] == 6
    assert out["analyzed_seconds"] == 1.04
    assert out["control"]["estimated_control_seconds"] == .6
    assert out["control"]["unknown_seconds"] == .44
    assert out["quality"]["selected_ball_coverage_pct"] == 57.69


def test_shape_requires_three_players_and_preserves_spatial_denominators():
    ps = [player(100), player(300, "T2"), player(600, "T3")]
    out = calculate([ps]*10)
    shape = out["kits"][0]["shape_timeline"][0]
    assert shape["centroid_x_pct"] == pytest.approx(33.33)
    assert shape["width_pct"] == 50
    assert shape["mean_visible_players"] == 3
    assert sum(map(sum, out["kits"][0]["image_occupancy_grid"])) == 30
    assert out["kits"][0]["image_thirds_pct"] == [66.67, 33.33, 0]
    assert calculate([ps[:2]]*10)["kits"][0]["shape_timeline"] == []


def test_path_bounded_and_gap_segments_never_connected():
    frames = [[ball(100)] if i % 4 else [] for i in range(4000)]
    out = calculate(frames)
    assert out["ball"]["path_observations"] == 3000
    assert len(out["ball"]["path"]) <= 600
    assert out["ball"]["path_display_stride"] == 5
    assert out["ball"]["path"][0]["segment_id"] != out["ball"]["path"][1]["segment_id"]


def test_backfill_preserves_other_metrics_reviews_and_is_idempotent(tmp_path):
    manifest = dict(engine="yolo_bytetrack", metadata=META, detection_summary=dict(sample_fps=10, decoded_frames=10),
                    metrics=dict(video_measurements={"existing": True}, possession={"home_pct": None}))
    original = copy.deepcopy(manifest["metrics"])
    (tmp_path/"tracking.json").write_text("[]")
    (tmp_path/"review.json").write_text('{"revision": 7}')
    assert ensure_insights(manifest, tmp_path)
    assert manifest["metrics"]["video_measurements"] == original["video_measurements"]
    assert manifest["metrics"]["possession"] == original["possession"]
    assert json.loads((tmp_path/"review.json").read_text()) == {"revision": 7}
    assert not ensure_insights(manifest, tmp_path)
    assert not ensure_insights(dict(engine="synthetic"), tmp_path)
    assert not ensure_insights(dict(engine="yolo_bytetrack", metrics={}), tmp_path)
    (tmp_path/"tracking.json").unlink()
    assert not ensure_insights(dict(engine="yolo_bytetrack", metrics={}, metadata=META, detection_summary=manifest["detection_summary"]), tmp_path)
