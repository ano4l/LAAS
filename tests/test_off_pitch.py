"""People who stay outside the pitch outline are set aside; players who step out briefly are not."""
import numpy as np

from workers.video import off_pitch
from workers.video.people_boxes import apply_roles

# Ten image pixels to a metre, so the 105 x 68 m pitch is the image area 0..1050 by 0..680.
RECORD = dict(status="calibrated", matrix=[[.1, 0, 0], [0, .1, 0], [0, 0, 1]])


def box(x_m, y_m):
    """A person box whose foot point is at this place on the pitch, in metres."""
    return [10 * x_m - 10, 10 * y_m - 60, 10 * x_m + 10, 10 * y_m]


def track(name, places, kind="player"):
    return [dict(frame_idx=frame, track_id=name, object_type=kind, bbox=box(*place)) for frame, place in enumerate(places)]


def test_distance_outside_the_outline_is_zero_on_the_pitch_and_metres_beyond_it():
    distances = off_pitch.outside_m(RECORD, [box(50, 30), box(-2, 30), box(50, 71), box(108, 72)])
    assert np.allclose(distances, [0, 2, 3, 5])


def test_a_point_beyond_the_horizon_counts_as_outside():
    tilted = dict(status="calibrated", matrix=[[-.1, 0, 100], [0, .1, 0], [0, .01, -1]])   # horizon across the image at y = 100
    near, far = off_pitch.outside_m(tilted, [[40, 300, 60, 400], [40, 20, 60, 50]])
    assert np.isfinite(near) and np.isinf(far)


def test_only_tracks_that_stay_off_the_pitch_are_set_aside():
    frames = {frame: RECORD for frame in range(40)}
    coach = track("coach", [(40, 71)] * 40)                      # three metres beyond the touchline throughout
    winger = track("winger", [(60, 67.5)] * 40)                  # on the touchline
    thrower = track("thrower", [(30, 70)] * 12 + [(30, 60)] * 28)  # out to fetch the ball, then back on
    linesman = track("linesman", [(70, 68.8)] * 40)              # within the margin of the outline
    glimpse = track("glimpse", [(40, 75)] * 5)                   # too few frames to judge
    official = track("official", [(52, 72)] * 40, "referee")
    rows = coach + winger + thrower + linesman + glimpse + official
    assert off_pitch.settle(rows, frames, keep={"official"}) == {"coach": off_pitch.ROLE}
    assert set(off_pitch.settle(rows, frames)) == {"coach", "official"}
    apply_roles(rows, off_pitch.settle(rows, frames, keep={"official"}))
    assert {r["object_type"] for r in coach} == {off_pitch.ROLE} and coach[0]["role_hint"] == "player"
    assert {r["object_type"] for r in winger + thrower} == {"player"}


def test_frames_without_a_calibration_are_not_counted():
    frames = {frame: RECORD if frame < 6 else dict(status="invalid") for frame in range(40)}
    assert off_pitch.settle(track("coach", [(40, 71)] * 40), frames) == {}   # only six placed frames
    assert off_pitch.measure(track("coach", [(40, 71)] * 40), frames)["coach"] == dict(placed=6, beyond=[6, 6, 6, 0, 0])
    assert off_pitch.settle(track("coach", [(40, 71)] * 40), {}) == {}
