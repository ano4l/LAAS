"""Passes estimated from the ball's path on the pitch map. Scenarios are synthetic: they check the rules, not accuracy."""
from workers.video.insights import VideoInsights

META = dict(width=1000, height=600, fps=10, sha256='a'*64)
# Ten image pixels to a metre: the image is a 100 x 60 m stretch of the pitch seen from above.
RECORD = dict(status='calibrated', matrix=[[.1, 0, 0], [0, .1, 0], [0, 0, 1]])


def person(x, track, kit, foot=200):
    return dict(object_type='player', track_id=track, bbox=[x-10, foot-100, x+10, foot], confidence=.9, team=kit, scene_id=0)


def ball(x, y=200):
    return dict(object_type='ball', bbox=[x-2, y-2, x+2, y+2], confidence=.8, ball_selected=True, scene_id=0)


def run(frames, calibrated=True, context=None):
    meta = dict(META, frame_count=len(frames), duration_seconds=len(frames)/10)
    calc = VideoInsights(meta, 10, context, {i: RECORD for i in range(len(frames))} if calibrated else None)
    for i, rows in enumerate(frames):
        calc.observe(i, rows)
    return calc.finish()


def played(people, held=5, start=105, end=395, steps=15, hidden=()):
    """The ball rests at `start` px, travels to `end` px at 2 m a frame (20 m/s), and rests there."""
    xs = [start]*held + [start + (end-start)*(i+1)/steps for i in range(steps)] + [end]*held
    return [people + ([] if i in hidden else [ball(x)]) for i, x in enumerate(xs)]


def test_ball_seen_travelling_to_a_team_mate_is_a_completed_pass():
    out = run(played([person(100, 'T1', 'kit-A'), person(400, 'T2', 'kit-A'), person(700, 'T3', 'kit-B')]))
    passes = out['sequences']['passes']
    assert passes['status'] == 'estimated' and passes['accuracy_evaluated'] is False
    assert passes['counts'] == dict(completed=1, not_completed=0, outcome_not_seen=0)
    row = passes['passes'][0]
    assert (row['from_track'], row['to_track'], row['outcome'], row['suggested_outcome']) == ('T1', 'T2', 'completed', 'completed')
    assert row['length_m'] == 30. and row['start_m'] == dict(x=10., y=20.) and row['end_m'] == dict(x=40., y=20.)
    assert row['review_status'] == 'unreviewed' and row['possession_included'] is False
    assert passes['by_team'] == [dict(team='scene-0/kit-A', attempted=1, completed=1, not_completed=0, completion_pct=100., mean_length_m=30.)]
    assert passes['pairs'] == [dict(team='scene-0/kit-A', from_track='T1', to_track='T2', passes=1)]
    assert [(p['track_id'], p['attempted'], p['completed'], p['received']) for p in passes['by_player']] == [('T1', 1, 1, 0), ('T2', 0, 0, 1)]
    # One entry in the review queue for the moment, carrying the place and outcome for the review form.
    queued = [e for e in out['candidates'] if e['type'] == 'possible_pass']
    assert len(queued) == 1 and queued[0]['suggested_outcome'] == 'completed' and queued[0]['end_m'] == dict(x=40., y=20.)


def test_ball_reaching_the_other_kit_group_is_a_pass_that_was_not_completed():
    # An opponent pressing the passer does not turn a struck pass into a tackle.
    out = run(played([person(100, 'T1', 'kit-A'), person(118, 'T4', 'kit-B', foot=215), person(400, 'T2', 'kit-B')]))
    passes = out['sequences']['passes']
    assert passes['counts'] == dict(completed=0, not_completed=1, outcome_not_seen=0)
    assert passes['passes'][0]['outcome'] == 'not_completed' and passes['passes'][0]['suggested_outcome'] == 'incomplete'
    assert passes['by_team'][0]['completion_pct'] == 0. and passes['pairs'] == []


def test_a_ball_taken_beside_the_passer_is_not_a_pass():
    # The ball is out of sight for 0.7 s, then is 4 m from the passer at an opponent's feet before going on to another opponent.
    frames = played([person(100, 'T1', 'kit-A'), person(150, 'T4', 'kit-B'), person(400, 'T2', 'kit-B')], start=105, held=5)
    frames = frames[:5] + [[r for r in rows if r['object_type'] != 'ball'] for rows in frames[5:12]] + \
        played([person(100, 'T1', 'kit-A'), person(150, 'T4', 'kit-B'), person(400, 'T2', 'kit-B')], start=140, held=1, steps=13)
    passes = run(frames)['sequences']['passes']
    assert passes['passes'] == [] and passes['diagnostics']['rejected'] == dict(taken_by_an_opponent=1) and passes['played_long'] == []


def test_players_side_by_side_or_a_ball_not_seen_travelling_give_no_pass():
    close = run(played([person(100, 'T1', 'kit-A'), person(130, 'T2', 'kit-A')], end=125, steps=2))['sequences']['passes']
    assert close['passes'] == [] and close['diagnostics']['rejected'] == dict(players_too_close=1)
    hidden = run(played([person(100, 'T1', 'kit-A'), person(400, 'T2', 'kit-A')], hidden=range(5, 20)))['sequences']['passes']
    assert hidden['passes'] == [] and hidden['diagnostics']['rejected'] == dict(ball_not_seen_travelling=1)


def test_a_ball_flying_past_a_player_does_not_give_that_player_the_ball():
    # T5 stands 1.5 m off the ball's line; the ball passes at 20 m/s on its way from T1 to T2.
    out = run(played([person(100, 'T1', 'kit-A'), person(250, 'T5', 'kit-B', foot=215), person(400, 'T2', 'kit-A')]))
    passes = out['sequences']['passes']
    assert [(r['from_track'], r['to_track'], r['outcome']) for r in passes['passes']] == [('T1', 'T2', 'completed')]


def test_a_ball_struck_long_and_not_seen_to_arrive_is_listed_apart_and_never_counted():
    frames = played([person(100, 'T1', 'kit-A')], end=700, steps=30, held=5)[:25]   # the ball is lost in flight
    passes = run(frames)['sequences']['passes']
    assert passes['passes'] == [] and passes['counts'] == dict(completed=0, not_completed=0, outcome_not_seen=1)
    assert passes['played_long'][0]['from_track'] == 'T1' and passes['played_long'][0]['outcome'] == 'outcome_not_seen'


def test_without_a_pitch_calibration_nothing_is_estimated_and_a_confirmed_stoppage_stops_a_pass():
    people = [person(100, 'T1', 'kit-A'), person(400, 'T2', 'kit-A')]
    out = run(played(people), calibrated=False)
    assert out['sequences']['passes']['status'] == 'not_run' and out['sequences']['passes']['passes'] == []
    stopped = run(played(people), context=dict(revision=1, dead_ball_intervals=[[.8, 1.2]]))['sequences']['passes']
    assert stopped['passes'] == [] and stopped['diagnostics']['rejected'] == dict(confirmed_stoppage=1)
