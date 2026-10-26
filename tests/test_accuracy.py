import pytest
from workers.video.accuracy import analysis_settings, tile_windows, merge_ball_candidates
from scripts.evaluate_ball import evaluate
from workers.video.accuracy import ball_search_window


def test_ball_roi_is_bounded_stale_positions_do_not_become_detections():
    box=ball_search_window(1920,1080,(10,1900,1060),(0,0),11,25)
    a,b,c,d=box
    assert 0<=a<c<=1920 and 0<=b<d<=1080
    assert ball_search_window(1920,1080,(10,100,100),(0,0),30,25) is None
    assert ball_search_window(1920,1080,None,(0,0),1,25) is None


def test_accurate_profile_preserves_all_source_frames(monkeypatch):
    for name in ["ANALYSIS_PROFILE", "ANALYSIS_FPS", "YOLO_IMAGE_SIZE", "BALL_TILE_SIZE"]:
        monkeypatch.delenv(name, raising=False)
    assert analysis_settings(29.97) == dict(profile="accurate", stride=1, image_size=1280, tile_size=0)
    monkeypatch.setenv("ANALYSIS_FPS", "10")
    assert analysis_settings(25)["stride"] == 3
    monkeypatch.setenv("ANALYSIS_FPS", "nan")
    with pytest.raises(ValueError):
        analysis_settings(25)


def test_tiles_cover_edges_and_nms_keeps_distinct_balls():
    windows = list(tile_windows(1920, 1080, 640))
    assert windows[0] == (0, 0, 640, 640)
    assert windows[-1] == (1280, 440, 1920, 1080)
    for y in range(0, 1080, 10):
        for x in range(0, 1920, 10):
            assert any(a <= x < c and b <= y < d for a, b, c, d in windows)
    rows = [dict(bbox=[10, 10, 20, 20], confidence=0.9),
            dict(bbox=[11, 10, 21, 20], confidence=0.8),
            dict(bbox=[30, 30, 40, 40], confidence=0.7)]
    assert merge_ball_candidates(rows) == [rows[0], rows[2]]


def test_metrics_penalize_duplicates_misses_and_empty_frame_false_alarms():
    labels = dict(width=100, height=100, frames=[
        dict(frame_idx=0, reviewed=True, boxes=[[10, 10, 20, 20]]),
        dict(frame_idx=1, reviewed=True, boxes=[[10, 10, 20, 20]]),
        dict(frame_idx=2, reviewed=True, boxes=[]),
        dict(frame_idx=3, reviewed=False, boxes=[])])
    predictions = [dict(frame_idx=index, object_type="ball", confidence=0.9, bbox=[10, 10, 20, 20])
                   for index in [0, 0, 2, 3]]
    result = evaluate(labels, predictions)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["excluded_unreviewed_frames"] == 1
    with pytest.raises(ValueError, match="No manually reviewed"):
        evaluate(dict(frames=[dict(reviewed=False)]), [])
