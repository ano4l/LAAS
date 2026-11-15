import numpy as np
import pytest

from workers.video import pitch_keypoints as pk
from workers.video.pitch_motion import pitch_motion

WIDTH, HEIGHT, FPS = 1920, 1080, 10.
SCALE, OFFSET = 10., np.array([300., 200.])  # synthetic view: 10 px per metre
META = dict(width=WIDTH, height=HEIGHT, duration_seconds=8.)


def record():
    xy = pk.landmarks() * SCALE + OFFSET
    seen = (xy[:, 0] >= 0) & (xy[:, 0] < WIDTH) & (xy[:, 1] >= 0) & (xy[:, 1] < HEIGHT)
    out = pk.fit(xy, np.where(seen, .9, 0.), WIDTH, HEIGHT)
    assert out["status"] == "calibrated"
    return out


def player(track, x_m, y_m, kit="kit-A"):
    x, y = np.array([x_m, y_m]) * SCALE + OFFSET
    return dict(scene="scene-0", track=track, kit=kit, x=float(x), y=float(y), height=40., confidence=.9)


def clip(count=80, noise=0., skip=(), teleport=None):
    random, frames = np.random.default_rng(5), []
    for i in range(count):
        t = i / FPS
        jitter = random.normal(0, noise, 4)
        players = [player("runner", 20 + 4 * t + jitter[0], 30 + jitter[1]), player("stander", 60 + jitter[2], 40 + jitter[3], "kit-B")]
        if teleport is not None and i >= teleport:
            players[1] = player("stander", 20 + jitter[2], 10 + jitter[3], "kit-B")
        frames.append(dict(frame=i, time=t, duration=1 / FPS, players=[] if i in skip else players))
    return frames


def test_a_steady_run_is_measured_and_a_standing_player_barely_moves():
    frames = clip(noise=.15)
    out = pitch_motion(frames, META, FPS, {i: record() for i in range(len(frames))})
    assert out["status"] == "estimated" and out["calibrated_seconds"] == pytest.approx(8.)
    runner, stander = out["teams"][0], out["teams"][1]
    assert runner["kit"] == "kit-A" and runner["tracks"][0]["track_id"] == "runner"
    assert runner["mean_speed_m_s"] == pytest.approx(4., abs=.25)
    assert runner["distance_m"] == pytest.approx(4 * runner["measured_player_seconds"], rel=.06)
    assert runner["measured_player_seconds"] < 8  # the ends of a stretch, where a full average does not fit, are not measured
    assert stander["mean_speed_m_s"] < .35
    assert np.sum(runner["heatmap_player_seconds"]) == pytest.approx(8.) == runner["placed_player_seconds"]


def test_uncalibrated_frames_and_missed_samples_are_not_bridged():
    frames = clip()
    pitch = {i: record() if not 30 <= i < 50 else dict(status="invalid", reason="No pitch found in the frame", landmarks_seen=0) for i in range(len(frames))}
    out = pitch_motion(frames, META, FPS, pitch)
    runner = out["teams"][0]
    assert out["calibrated_seconds"] == pytest.approx(6.)
    assert runner["placed_player_seconds"] == pytest.approx(6.)
    # Two separate stretches; the 8 m covered while uncalibrated is not counted.
    assert runner["measured_player_seconds"] <= 5 and runner["distance_m"] == pytest.approx(4 * runner["measured_player_seconds"], rel=.01)
    one = pitch_motion(clip(skip={40}), META, FPS, {i: record() for i in range(80)})["teams"][0]
    assert one["measured_player_seconds"] < pitch_motion(clip(), META, FPS, {i: record() for i in range(80)})["teams"][0]["measured_player_seconds"]


def test_an_impossible_jump_is_rejected_not_counted_as_a_sprint():
    out = pitch_motion(clip(teleport=40), META, FPS, {i: record() for i in range(80)})
    stander = out["teams"][1]
    assert stander["tracks"][0]["rejected_hops"] >= 1
    assert stander["fastest_hop_m_s"] <= 11 and stander["distance_m"] < 60


def test_cells_the_camera_never_saw_have_no_coverage_and_no_calibration_means_unavailable():
    out = pitch_motion(clip(), META, FPS, {i: record() for i in range(80)})
    coverage = np.array(out["coverage_seconds"])
    assert coverage.shape == (14, 21)
    # The synthetic image shows pitch x up to (1920 - 300) / 10 = 162 m, y up to 88 m: the whole pitch.
    assert coverage.min() == pytest.approx(8.)
    assert pitch_motion(clip(), META, FPS, None)["status"] == "unavailable"
    nothing = pitch_motion(clip(), META, FPS, {i: dict(status="invalid", reason="x", landmarks_seen=0) for i in range(80)})
    assert nothing["status"] == "unavailable" and nothing["teams"] == []
