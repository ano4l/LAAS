import numpy as np

from workers.video.ball_path import select_ball_path
from workers.video.camera_motion import CameraMotion

META = dict(fps=25, width=1920, height=1080)


def ball(frame, x, y, confidence, **extra):
    return dict(frame_idx=frame, object_type="ball", bbox=[x - 5, y - 5, x + 5, y + 5], confidence=confidence, scene_id=0, **extra)


def selected(rows):
    return [(r["frame_idx"], round((r["bbox"][0] + r["bbox"][2]) / 2)) for r in rows if r["ball_selected"]]


def test_blurred_ball_between_confident_detections_is_kept_and_isolated_weak_ones_are_not():
    rows = [ball(0, 100, 500, .8), ball(1, 110, 500, .12), ball(2, 120, 500, .15), ball(3, 130, 500, .7),
            ball(40, 900, 300, .2), ball(41, 905, 300, .2)]
    summary = select_ball_path(rows, META)
    assert selected(rows) == [(0, 100), (1, 110), (2, 120), (3, 130)]
    assert [r["ball_path_support"] for r in rows[:4]] == ["confident", "weak_between_confident", "weak_between_confident", "confident"]
    assert summary["selected"] == 4  # the isolated weak pair is never worth starting a path for
    assert all(r["ball_path_segment"] == 0 for r in rows[:4]) and rows[4]["ball_path_segment"] is None


def test_weak_run_after_the_last_confident_detection_is_not_selected():
    rows = [ball(0, 100, 500, .8), ball(1, 105, 500, .8), ball(2, 110, 500, .2), ball(3, 115, 500, .2)]
    summary = select_ball_path(rows, META)
    assert selected(rows) == [(0, 100), (1, 105)] and summary["weak_candidates_not_bridged"] == 2


def test_path_prefers_the_candidate_that_continues_the_trajectory_and_keeps_the_online_choice_for_reference():
    rows = [ball(0, 100, 500, .6, ball_selected=True), ball(1, 110, 500, .6, ball_selected=True),
            ball(2, 120, 500, .5, ball_selected=False), ball(2, 1500, 200, .55, ball_selected=True), ball(3, 130, 500, .6)]
    select_ball_path(rows, META)
    assert selected(rows) == [(0, 100), (1, 110), (2, 120), (3, 130)]
    assert rows[3]["ball_selected_online"] is True and rows[3]["ball_selected"] is False


def test_weak_object_fixed_in_the_scene_is_rejected_even_while_the_camera_pans():
    # The camera pans 8 px per frame, so a sideline object drifts across the image while staying put in the scene.
    warps = {f: [[1, 0, 8], [0, 1, 0]] for f in range(60)}
    camera = CameraMotion.from_warps(warps)
    decoy = [ball(f, 300 + 8 * f, 200, .45 if f == 30 else .2) for f in range(0, 60, 2)]
    real = [ball(f, 900 + 20 * f, 600, .7) for f in range(60)]
    rows = decoy + real
    summary = select_ball_path(rows, META, camera=camera)
    assert summary["static_candidates_rejected"] == len(decoy) and summary["camera_motion"] == "image_features"
    assert all(not r["ball_selected"] for r in decoy) and all(r["ball_selected"] for r in real)


def test_confidently_detected_resting_ball_is_not_treated_as_a_fixed_object():
    rows = [ball(f, 500, 500, .8 if f % 5 == 0 else .3) for f in range(61)]
    summary = select_ball_path(rows, META, camera=CameraMotion.from_warps({f: [[1, 0, 0], [0, 1, 0]] for f in range(60)}))
    assert summary["static_candidates_rejected"] == 0 and summary["selected"] == 61


def test_no_candidates_and_other_objects_are_left_alone():
    rows = [dict(frame_idx=0, object_type="player", track_id="T1", bbox=[0, 0, 10, 30], confidence=.9)]
    assert select_ball_path(rows, META)["selected"] == 0 and "ball_selected" not in rows[0]


def test_camera_motion_stabilises_pan_and_zoom_and_restarts_at_scene_cuts():
    camera = CameraMotion.from_warps({0: [[1, 0, 0], [0, 1, 0]], 1: [[1, 0, 10], [0, 1, 0]], 2: [[2, 0, 0], [0, 2, 0]],
                                      3: [[1, 0, 500], [0, 1, 0]]}, scenes={0: 0, 1: 0, 2: 0, 3: 1})
    assert np.allclose(camera.stabilise(1, (110, 50)), (100, 50))
    assert np.allclose(camera.stabilise(2, (220, 100)), (100, 50)) and np.isclose(camera.scale(2), .5)
    assert np.allclose(camera.stabilise(3, (220, 100)), (220, 100))  # new scene: its own reference frame
    people = [dict(frame_idx=f, track_id=f"T{k}", object_type="player", bbox=[100 * k + 5 * f, 0, 100 * k + 5 * f + 10, 30])
              for f in range(3) for k in range(5)]
    fallback = CameraMotion.from_people(people)
    assert fallback.source == "tracked_people_median" and np.allclose(fallback.stabilise(2, (60, 30)), (50, 30))
