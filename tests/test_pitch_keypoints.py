import numpy as np
import pytest

from workers.video import pitch_keypoints as pk
from workers.video.collective_dynamics import collective_dynamics

WIDTH, HEIGHT = 1920, 1080
# A synthetic view: 10 px per metre, pitch origin at image (300, 200). Not a real camera.
SCALE, OFFSET = 10., np.array([300., 200.])


def view(indices, noise=0.):
    """Image positions and confidences for the chosen landmark numbers (1-based); all others unseen."""
    xy = pk.landmarks() * SCALE + OFFSET
    if noise:
        xy = xy + np.random.default_rng(1).normal(0, noise, xy.shape)
    confidence = np.zeros(len(xy))
    confidence[[i - 1 for i in indices]] = .9
    return xy, confidence


def test_landmarks_follow_the_laws_of_the_game():
    world = pk.landmarks()
    assert world.shape == (32, 2)
    assert world[0].tolist() == [0, 0] and world[29].tolist() == [105, 68]
    assert world[8].tolist() == [11, 34] and world[21].tolist() == [94, 34]  # penalty spots
    assert world[12][1] - world[9][1] == pytest.approx(40.32) and world[9][0] == 16.5  # penalty area
    assert world[31][0] - world[30][0] == pytest.approx(18.3)  # centre circle diameter
    # Landmarks 11 and 12 lie on the penalty arc, 9.15 m from the spot.
    assert np.linalg.norm(world[10] - world[8]) == pytest.approx(9.15)


def test_fit_recovers_a_known_mapping_and_projects_foot_points():
    xy, confidence = view([14, 15, 16, 17, 18, 19, 20, 21, 31, 32])
    record = pk.fit(xy, confidence, WIDTH, HEIGHT)
    assert record["status"] == "calibrated" and record["landmarks_used"] == 10
    assert record["mean_error_m"] < .01 and record["mean_reprojection_px"] < .1
    points, usable = pk.to_pitch(record, [OFFSET + SCALE * np.array([52.5, 34.]), OFFSET + SCALE * np.array([120., 34.])])
    assert usable.tolist() == [True, False]  # the second lies 15 m beyond the goal line
    assert points[0] == pytest.approx([52.5, 34.], abs=.01)
    area = np.array(pk.visible_area(record, WIDTH, HEIGHT))
    assert area[:, 0].min() == 0 and area[:, 0].max() == 105 and area[:, 1].max() == 68  # the whole pitch is in this view


def test_fit_refuses_weak_or_inconsistent_evidence():
    xy, confidence = view([14, 15, 16, 17])
    assert pk.fit(xy, confidence, WIDTH, HEIGHT)["status"] == "invalid"  # four landmarks cannot be cross-checked
    xy, confidence = view([1, 2, 3, 4, 5, 6])  # all on one goal line
    assert pk.fit(xy, confidence, WIDTH, HEIGHT)["status"] == "invalid"
    xy, confidence = view([14, 15, 16, 17, 18, 19, 20, 21, 31, 32], noise=25)  # 2.5 m scatter
    assert pk.fit(xy, confidence, WIDTH, HEIGHT)["status"] == "invalid"
    xy, confidence = view([14, 15, 16, 17, 18, 19, 20, 21, 31, 32])
    xy[:, 0] = WIDTH - xy[:, 0]  # mirrored image
    assert pk.fit(xy, confidence, WIDTH, HEIGHT)["status"] == "invalid"


def test_one_bad_landmark_is_left_out_not_averaged_in():
    xy, confidence = view([14, 15, 16, 17, 18, 19, 20, 21, 31, 32])
    xy[17] += 80  # landmark 18 misplaced by 8 m
    record = pk.fit(xy, confidence, WIDTH, HEIGHT)
    assert record["status"] == "calibrated" and record["landmarks_seen"] == 10 and record["landmarks_used"] == 9
    assert record["mean_error_m"] < .01


def test_shape_snapshot_gets_pitch_positions_only_from_a_calibrated_frame():
    meta = dict(width=WIDTH, height=HEIGHT, fps=10, duration_seconds=1, frame_count=10)
    spots = [(40., 20.), (50., 30.), (45., 50.), (130., 30.)]  # the last is off the pitch

    def frame(i):
        return dict(frame=i, time=i / 10, duration=.1, players=[
            dict(track=f"T{n}", scene=0, kit="scene-0/kit-A", x=OFFSET[0] + SCALE * x, y=OFFSET[1] + SCALE * y, height=100, confidence=.9)
            for n, (x, y) in enumerate(spots)])

    frames = [frame(i) for i in range(10)]
    xy, confidence = view([14, 15, 16, 17, 18, 19, 20, 21, 31, 32])
    record = pk.fit(xy, confidence, WIDTH, HEIGHT)
    bad = dict(status="invalid", reason="Only 2 pitch landmarks seen; 5 are needed", landmarks_seen=2)
    # Frame 5 is nearest the middle of the second but failed calibration; frame 3 passed.
    out = collective_dynamics(frames, meta, 10, {3: record, 5: bad})
    snapshot = out["teams"][0]["timeline"][0]["snapshot"]
    assert out["pitch"]["status"] == "estimated" and out["pitch"]["frames_calibrated"] == 1 and out["pitch"]["accuracy_evaluated"] is False
    assert snapshot["timestamp_seconds"] == .3
    pitch = snapshot["pitch"]
    assert [p["track_id"] for p in pitch["players"]] == ["T0", "T1", "T2"] and pitch["players_not_placed"] == 1
    assert (pitch["players"][0]["x_m"], pitch["players"][0]["y_m"]) == pytest.approx((40, 20), abs=.02)
    assert pitch["length_extent_m"] == pytest.approx(10, abs=.1) and pitch["width_extent_m"] == pytest.approx(30, abs=.1)
    assert pitch["hull_area_m2"] == pytest.approx(125, abs=1)
    # Without calibration nothing about the pitch is claimed, and the snapshot is the middle frame as before.
    plain = collective_dynamics(frames, meta, 10)
    assert plain["pitch"]["status"] == "unavailable" and "pitch" not in plain["teams"][0]["timeline"][0]["snapshot"]
    assert plain["teams"][0]["timeline"][0]["snapshot"]["timestamp_seconds"] == .5
    none_passed = collective_dynamics(frames, meta, 10, {5: bad})
    assert none_passed["pitch"]["status"] == "unavailable" and "pitch" not in none_passed["teams"][0]["timeline"][0]["snapshot"]


def test_calibrate_run_adds_calibration_to_an_existing_run_or_writes_nothing(tmp_path, monkeypatch):
    import cv2
    import hashlib
    import json

    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
    for _ in range(6):
        writer.write(np.zeros((180, 320, 3), np.uint8))
    writer.release()
    weights = tmp_path / "pitch.pt"
    weights.write_bytes(b"weights")
    run = tmp_path / "run"
    run.mkdir()
    manifest = dict(engine="yolo_bytetrack", warnings=[], metrics={},
                    metadata=dict(width=320, height=180, fps=10., duration_seconds=.6, sha256=hashlib.sha256(video.read_bytes()).hexdigest()),
                    detection_summary=dict(sample_fps=5., decoded_frames=6))
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "tracking.json").write_text("[]", encoding="utf-8")

    class Model:
        def __init__(self, weights, device):
            pass

        def calibrate(self, frame):
            return dict(status="invalid", reason="No pitch found in the frame", landmarks_seen=0)

    monkeypatch.setattr(pk, "PitchKeypointModel", Model)
    result = pk.calibrate_run(video, run, weights)
    saved = json.loads((run / "pitch-calibration.json").read_text(encoding="utf-8"))
    assert sorted(saved["frames"]) == ["0", "2", "4"] and saved["summary"]["frames_calibrated"] == 0
    assert result["detection_summary"]["pitch_calibration"]["summary"]["frames_tried"] == 3
    assert result["warnings"] == [pk.CALIBRATION_WARNING]
    assert json.loads((run / "manifest.json").read_text(encoding="utf-8")) == result
    assert result["metrics"]["video_insights"]["collective_dynamics"]["pitch"]["status"] != "estimated"

    # A different video is refused before anything is written.
    (run / "pitch-calibration.json").unlink()
    video.write_bytes(video.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        pk.calibrate_run(video, run, weights)
    assert not (run / "pitch-calibration.json").exists()
