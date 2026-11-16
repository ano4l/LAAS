import cv2
import numpy as np

from workers.video import pitch_keypoints as pk
from workers.video import pitch_lines as pl

WIDTH, HEIGHT = 1920, 1080


def camera(target_x=35., focal=1900., position=(52.5, 68. + 45., -18.)):
    """Pitch -> image for a synthetic main camera beyond the near touchline (up is -z)."""
    position = np.array(position)
    forward = np.array([target_x, 34., 0.]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 0., 1.], forward)
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(forward, right), forward])
    k = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.]])
    return k @ np.column_stack([rotation[:, 0], rotation[:, 1], -rotation @ position])


def frame(to_image, players=0, seed=1):
    """Striped grass with the pitch's lines painted through `to_image`, and optionally some white-shirted players."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for band in range(0, WIDTH, 160):
        image[:, band:band + 160] = (40, 150, 45) if band // 160 % 2 else (35, 125, 40)
    image[:180] = (90, 80, 70)  # stands
    points = pk._project(to_image, np.concatenate([part for _, part in pl.markings(step=.05)]))
    for x, y in points:
        if 0 <= x < WIDTH and 180 <= y < HEIGHT:
            cv2.circle(image, (int(x), int(y)), 2, (235, 235, 235), -1)
    random = np.random.default_rng(seed)
    for _ in range(players):
        x, y = int(random.uniform(100, WIDTH - 100)), int(random.uniform(300, HEIGHT - 50))
        cv2.rectangle(image, (x - 12, y - 60), (x + 12, y), (240, 240, 240), -1)
    return image


def test_the_true_mapping_is_well_supported_and_a_shifted_one_leaves_a_marking_on_bare_grass():
    to_image = camera()
    image = frame(to_image, players=15)
    truth = pl.support(image, np.linalg.inv(to_image))
    assert truth["points"] >= pl.MIN_POINTS and truth["share"] > .9
    assert truth["markings_seen"] >= 4 and truth["markings_on_lines"] == truth["markings_seen"]
    shift = np.array([[1, 0, 3.], [0, 1, 0], [0, 0, 1]])  # the whole pitch 3 m along: lines along the pitch still match
    wrong = pl.support(image, shift @ np.linalg.inv(to_image))
    assert wrong["share"] < .7 and wrong["weakest_share"] < .2 and wrong["markings_on_lines"] < wrong["markings_seen"]


def test_a_consistent_but_invented_penalty_area_in_midfield_is_not_supported():
    # The view looks at midfield; a mapping that puts the right penalty area there, as hallucinated landmarks would.
    to_image = camera(target_x=55.)
    image = frame(to_image)
    slide = np.array([[1, 0, 35.], [0, 1, 0], [0, 0, 1]])
    wrong = pl.support(image, slide @ np.linalg.inv(to_image))
    assert wrong["points"] >= pl.MIN_POINTS and wrong["markings_on_lines"] <= wrong["markings_seen"] / 2
    assert wrong["weakest"].startswith(("right penalty", "right goal")) and wrong["weakest_share"] < .2
    truth = pl.support(image, np.linalg.inv(to_image))
    assert truth["share"] > .9 and truth["markings_on_lines"] == truth["markings_seen"]


def test_too_few_markings_in_view_cannot_be_judged():
    # Zoomed right in on open grass between the lines.
    to_image = camera(target_x=30., focal=30000.)
    out = pl.support(frame(to_image), np.linalg.inv(to_image))
    assert out["share"] is None and out["points"] < pl.MIN_POINTS


def test_line_pixels_need_grass_around_them():
    image = frame(camera())
    mask = pl.line_mask(image)
    assert not mask[:150].any()  # the stands are not grass
    assert mask[200:].mean() < .1  # lines are thin


def calibrated(to_image):
    xy = pk._project(to_image, pk.landmarks())
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < WIDTH) & (xy[:, 1] >= 0) & (xy[:, 1] < HEIGHT)
    record = pk.fit(xy, np.where(inside, .9, 0.), WIDTH, HEIGHT)
    assert record["status"] == "calibrated"
    return record


def invented(record):
    """The same record with its mapping slid 35 m along the pitch, as a fit to invented landmarks would be."""
    slide = np.array([[1, 0, 35.], [0, 1, 0], [0, 0, 1]])
    return dict(record, matrix=list((slide @ np.asarray(record["matrix"]).reshape(3, 3)).ravel()))


def test_the_line_check_measures_a_frame_and_never_rejects_it():
    to_image = camera(target_x=55.)
    image, record = frame(to_image), calibrated(to_image)
    good = pk.check_lines(record, image)
    assert good["status"] == "calibrated" and good["line_support"] >= pk.MIN_LINE_SUPPORT and good["line_markings"][0] == good["line_markings"][1]
    bad = pk.check_lines(invented(record), image)
    assert bad["status"] == "calibrated" and bad["matrix"] == invented(record)["matrix"] and bad["line_support"] < pk.MIN_LINE_SUPPORT
    assert pk.check_lines(dict(status="invalid", reason="x"), image) == dict(status="invalid", reason="x")


def test_a_clip_where_most_checked_frames_miss_the_lines_keeps_its_frames_and_carries_a_warning():
    to_image = camera(target_x=55.)
    image, record = frame(to_image), calibrated(to_image)
    good, bad = pk.check_lines(record, image), pk.check_lines(invented(record), image)
    frames = {index: dict(bad if index % 3 else good) for index in range(12)}  # a third match
    frames[12] = dict(record)  # not checked: counts towards neither side
    doc = pk.document(frames, "pitch.pt", "abc", 1, (WIDTH, HEIGHT, 25.))
    check = doc["summary"]["line_check"]
    assert check["status"] == "lines_do_not_match" and check["frames_checked"] == 12 and check["frames_matching"] == 4
    assert "only 33% of checked frames match" in check["warning"] and "several metres off" in check["warning"]
    assert doc["summary"]["frames_calibrated"] == 13 and all(r["status"] == "calibrated" for r in doc["frames"].values())
    # The insights carry it to the Shape tab and the profile.
    from workers.video.collective_dynamics import pitch_status
    status = pitch_status({int(k): v for k, v in doc["frames"].items()})
    assert status["status"] == "estimated" and status["line_check"]["warning"] == check["warning"]


def test_a_clip_where_most_frames_match_has_no_warning():
    to_image = camera(target_x=55.)
    image, record = frame(to_image), calibrated(to_image)
    good, bad = pk.check_lines(record, image), pk.check_lines(invented(record), image)
    doc = pk.document({index: dict(bad if index == 5 else good) for index in range(12)}, "pitch.pt", "abc", 1, (WIDTH, HEIGHT, 25.))
    check = doc["summary"]["line_check"]
    assert check["status"] == "lines_match" and check["frames_matching"] == 11 and check["warning"] is None


def test_a_calibration_made_before_the_line_check_says_it_was_not_run():
    record = calibrated(camera(target_x=55.))
    doc = pk.document({index: dict(record) for index in range(6)}, "pitch.pt", "abc", 1, (WIDTH, HEIGHT, 25.))
    assert doc["summary"]["line_check"]["status"] == "not_run" and doc["summary"]["line_check"]["warning"] is None


def test_a_stored_run_is_checked_from_its_video_without_the_model(tmp_path):
    import hashlib
    import json
    to_image = camera(target_x=55.)
    image, record = frame(to_image), calibrated(to_image)
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25., (WIDTH, HEIGHT))
    for _ in range(12):
        writer.write(image)
    writer.release()
    run = tmp_path / "run"
    run.mkdir()
    frames = {index: dict(record if index < 3 else invented(record)) for index in range(12)}  # most frames' mappings are wrong
    (run / "pitch-calibration.json").write_text(json.dumps(pk.document(frames, "pitch.pt", "abc", 1)), encoding="utf-8")
    (run / "tracking.json").write_text("[]", encoding="utf-8")
    metadata = dict(width=WIDTH, height=HEIGHT, fps=25., duration_seconds=12 / 25, frame_count=12, sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    (run / "manifest.json").write_text(json.dumps(dict(engine="yolo_bytetrack", metadata=metadata, detection_summary=dict(sample_fps=25., decoded_frames=12), metrics={})), encoding="utf-8")
    check = pk.check_run_lines(video, run, every=1)
    assert check["status"] == "lines_do_not_match" and check["frames_checked"] == 12 and check["frames_matching"] == 3
    stored = json.loads((run / "pitch-calibration.json").read_text(encoding="utf-8"))
    assert stored["schema_version"] == pk.VERSION and stored["summary"]["frames_calibrated"] == 12
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["detection_summary"]["pitch_calibration"]["summary"]["line_check"]["status"] == "lines_do_not_match"
    assert manifest["metrics"]["video_insights"]["collective_dynamics"]["pitch"]["line_check"]["warning"]
