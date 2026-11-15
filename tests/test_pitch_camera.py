import numpy as np

from workers.video import pitch_camera as pc
from workers.video import pitch_keypoints as pk

WIDTH, HEIGHT, FPS = 1920, 1080, 25.
# A synthetic main camera: beyond the near touchline, 18 m above the pitch (up is -z here).
POSITION = np.array([52.5, 68. + 45., -18.])


def camera(position, target, f):
    forward = np.array([*target, 0.]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 0., 1.], forward)
    right /= np.linalg.norm(right)
    return f, np.array([right, np.cross(forward, right), forward])


def frames(count, noise, seed=3, cut_at=None):
    random, out, truth = np.random.default_rng(seed), {}, {}
    for index in range(count):
        cut = cut_at is not None and index >= cut_at
        position = POSITION + np.array([40., 30., -10.]) if cut else POSITION
        f, rotation = camera(position, (70. if cut else 35. + .5 * index, 34.), 1900. + 6 * index)
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
    xy, usable = pk.to_pitch(dict(matrix=record.get(key, record["matrix"])), image[seen])
    return float(np.linalg.norm(xy[usable] - world[seen][usable], axis=1).mean())


def test_smoothing_makes_positions_steadier_and_closer_to_the_truth_and_keeps_the_landmark_checks():
    records, truth = frames(40, noise=2.5)
    assert all(record["status"] == "calibrated" for record in records.values())
    summary = pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert summary["status"] == "stabilised" and summary["frames_stabilised"] >= 34 and summary["stretches"] == 1
    assert summary["jitter_after_m"] < .5 * summary["jitter_before_m"]
    before = np.mean([position_error(record, truth[index], "frame_matrix") for index, record in records.items()])
    after = np.mean([position_error(record, truth[index]) for index, record in records.items()])
    assert after < .7 * before
    # A steadied frame passes the same landmark check a frame's own fit must pass.
    assert all(record["stable_error_m"] <= pk.MAX_MEAN_ERROR_M for record in records.values() if record["stabilised"])


def test_steadying_twice_gives_the_same_result():
    records, _ = frames(20, noise=2.)
    pc.stabilise(records, WIDTH, HEIGHT, FPS)
    first = {index: list(record["matrix"]) for index, record in records.items()}
    pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert all(np.allclose(first[index], record["matrix"], rtol=1e-6) for index, record in records.items())


def test_too_few_neighbours_keep_their_own_fits():
    records, _ = frames(3, noise=1.)
    own = {index: list(record["matrix"]) for index, record in records.items()}
    summary = pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert summary["status"] == "not_stabilised" and summary["frames_stabilised"] == 0 and summary["reason"]
    assert all(record["matrix"] == own[index] and record["stabilised"] is False for index, record in records.items())


def test_smoothing_does_not_cross_a_cut_or_a_gap():
    records, truth = frames(40, noise=0., cut_at=20)
    assert all(record["status"] == "calibrated" for record in records.values())
    summary = pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert summary["stretches"] == 2
    # Without noise, smoothing within each side changes nothing; smoothing across the cut would drag both sides off.
    for index in (18, 19, 20, 21):
        assert position_error(records[index], truth[index]) < .05

    records, _ = frames(40, noise=1.)
    for index in range(10, 30):  # 0.8 s without calibration
        records[index] = dict(status="invalid", reason="No pitch found in the frame", landmarks_seen=0)
    assert pc.stabilise(records, WIDTH, HEIGHT, FPS)["stretches"] == 2


def test_uncalibrated_frames_are_never_given_a_mapping():
    records, _ = frames(20, noise=1.)
    records[7] = dict(status="invalid", reason="Only 2 pitch landmarks seen; 5 are needed", landmarks_seen=2)
    pc.stabilise(records, WIDTH, HEIGHT, FPS)
    assert records[7] == dict(status="invalid", reason="Only 2 pitch landmarks seen; 5 are needed", landmarks_seen=2)


def test_a_stored_calibration_from_an_older_version_is_steadied_again_without_a_model():
    records, _ = frames(20, noise=2.)
    doc = pk.document(records, "pitch.pt", "abc", 1, (WIDTH, HEIGHT, FPS))
    assert doc["schema_version"] == pk.VERSION and doc["summary"]["steadying"]["frames_stabilised"] > 0
    assert not pk.upgrade(doc, (WIDTH, HEIGHT, FPS))  # already current
    expected = {index: list(record["matrix"]) for index, record in doc["frames"].items()}
    # As an older version stored it: its own fits only, under the old version name.
    for record in doc["frames"].values():
        record["matrix"] = record.pop("frame_matrix", record["matrix"])
        record["stabilised"] = False
    doc["schema_version"] = "pitch-keypoints-2"
    assert pk.upgrade(doc, (WIDTH, HEIGHT, FPS))
    assert doc["schema_version"] == pk.VERSION and doc["model"] == "pitch.pt" and doc["model_sha256"] == "abc"
    assert all(np.allclose(expected[index], record["matrix"], rtol=1e-6) for index, record in doc["frames"].items())
