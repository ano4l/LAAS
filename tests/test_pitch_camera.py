import numpy as np

from workers.video import pitch_camera as pc
from workers.video import pitch_keypoints as pk

WIDTH, HEIGHT, FPS = 1920, 1080, 25.
# A synthetic main camera: beyond the near touchline, 18 m above the pitch (up is -z here).
POSITION = np.array([52.5, 68. + 45., -18.])


def camera(target, f):
    forward = np.array([*target, 0.]) - POSITION
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 0., 1.], forward)
    right /= np.linalg.norm(right)
    return f, np.array([right, np.cross(forward, right), forward])


def frames(count, noise, seed=3, cut_at=None):
    random, out, truth = np.random.default_rng(seed), {}, {}
    for index in range(count):
        position = POSITION if cut_at is None or index < cut_at else POSITION + np.array([40., 30., -10.])
        f, rotation = camera((35. + .5 * index, 34.), 1900. + 6 * index)
        k = np.array([[f, 0, WIDTH / 2], [0, f, HEIGHT / 2], [0, 0, 1.]])
        g = k @ np.column_stack([rotation[:, 0], rotation[:, 1], -rotation @ position])
        truth[index] = g
        xy = pk._project(g, pk.landmarks())
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < WIDTH) & (xy[:, 1] >= 0) & (xy[:, 1] < HEIGHT)
        out[index] = pk.fit(xy + random.normal(0, noise, xy.shape), np.where(inside, .9, 0.), WIDTH, HEIGHT)
    return out, truth


def position_error(record, truth, key="matrix"):
    """Mean error (m) of image points on the pitch, away from most landmarks: the near touchline side."""
    world = np.array([(x, 66.) for x in np.linspace(20, 85, 14)])
    image = pk._project(truth, world)
    seen = (image[:, 0] >= 0) & (image[:, 0] < WIDTH) & (image[:, 1] >= 0) & (image[:, 1] < HEIGHT)
    xy, usable = pk.to_pitch(dict(matrix=record[key]), image[seen])
    return float(np.linalg.norm(xy[usable] - world[seen][usable], axis=1).mean())


def test_a_steady_camera_is_found_and_makes_positions_steadier_and_closer_to_the_truth():
    records, truth = frames(40, noise=2.5)
    assert all(record["status"] == "calibrated" for record in records.values())
    summary = pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert summary["status"] == "stabilised" and summary["frames_stabilised"] == 40
    assert np.linalg.norm(np.array(summary["camera_position_m"]) - POSITION) < 6
    assert summary["jitter_after_m"] < .5 * summary["jitter_before_m"]
    before = np.mean([position_error(record, truth[index], "frame_matrix") for index, record in records.items()])
    after = np.mean([position_error(record, truth[index]) for index, record in records.items()])
    assert after < .6 * before
    assert all(record["stabilised"] and record["stable_error_m"] <= pc.MAX_MEAN_ERROR_M for record in records.values())


def test_steadying_twice_gives_the_same_result():
    records, _ = frames(20, noise=2.)
    pc.stabilise(records, WIDTH, HEIGHT, FPS)
    first = {index: list(record["matrix"]) for index, record in records.items()}
    pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert all(np.allclose(first[index], record["matrix"], rtol=1e-6) for index, record in records.items())


def test_too_few_frames_or_mixed_cameras_keep_their_own_fits():
    records, _ = frames(5, noise=1.)
    own = {index: list(record["matrix"]) for index, record in records.items()}
    summary = pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert summary["status"] == "not_stabilised" and "needed" in summary["reason"]
    assert all(record["matrix"] == own[index] and record["stabilised"] is False for index, record in records.items())

    # Half the clip from another camera: either refused outright, or frames the clip camera cannot explain keep their own fit.
    records, _ = frames(40, noise=1., cut_at=20)
    own = {index: list(record["matrix"]) for index, record in records.items() if record["status"] == "calibrated"}
    summary = pc.stabilise(records, WIDTH, HEIGHT, FPS)
    for index, matrix in own.items():
        record = records[index]
        assert record.get("frame_matrix", record["matrix"]) == matrix
        assert not record["stabilised"] or record["stable_error_m"] <= pc.MAX_MEAN_ERROR_M
    assert len(own) == 40 and summary["frames_stabilised"] < 40


def test_uncalibrated_frames_are_never_given_a_mapping():
    records, _ = frames(20, noise=1.)
    records[7] = dict(status="invalid", reason="Only 2 pitch landmarks seen; 5 are needed", landmarks_seen=2)
    pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert records[7] == dict(status="invalid", reason="Only 2 pitch landmarks seen; 5 are needed", landmarks_seen=2)
