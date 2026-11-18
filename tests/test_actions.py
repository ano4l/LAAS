"""Rule-based shot, tackle, duel and dribble candidates. Scenarios are synthetic: they check the rules, not accuracy."""
from workers.video.insights import VideoInsights

META = dict(width=1000, height=600, fps=10, sha256='a'*64)


def person(x, track, kit, foot=200, role='player'):
    return dict(object_type=role, track_id=track, bbox=[x-10, foot-100, x+10, foot], confidence=.9, team=kit, scene_id=0)


def ball(x, y=200):
    return dict(object_type='ball', bbox=[x-2, y-2, x+2, y+2], confidence=.8, ball_selected=True, scene_id=0)


def run(frames, pitch=None, context=None):
    meta = dict(META, frame_count=len(frames), duration_seconds=len(frames)/10)
    calc = VideoInsights(meta, 10, context, pitch)
    for i, rows in enumerate(frames):
        calc.observe(i, rows)
    return calc.finish()


def kinds(out):
    return [e['type'] for e in out['sequences']['actions']['candidates']]


def test_ball_changing_kit_group_at_close_range_is_a_possible_tackle_not_a_turnover():
    pair = [person(100, 'T1', 'kit-A'), person(150, 'T2', 'kit-B')]
    out = run([pair+[ball(100)]]*3+[pair+[ball(150)]]*3)
    assert kinds(out) == ['possible_tackle']
    tackle = out['sequences']['actions']['candidates'][0]
    assert (tackle['from_track'], tackle['to_track']) == ('T1', 'T2')
    assert tackle['review_status'] == 'unreviewed' and tackle['possession_included'] is False
    assert tackle['suggested_review_type'] == 'tackle' and tackle['players_closest_heights'] == .5
    # The pass and turnover list is unchanged; the merged queue holds both kinds.
    assert out['sequences']['candidates'] == [] and [e['id'] for e in out['candidates']] == ['tackle-1']


def test_same_kit_handover_and_distant_opponent_are_not_tackles():
    mates = [person(100, 'T1', 'kit-A'), person(150, 'T2', 'kit-A')]
    assert kinds(run([mates+[ball(100)]]*3+[mates+[ball(150)]]*3)) == []
    apart = [person(100, 'T1', 'kit-A'), person(600, 'T2', 'kit-B')]
    travel = [apart+[ball(x)] for x in range(150, 600, 50)]
    out = run([apart+[ball(100)]]*3+travel+[apart+[ball(600)]]*3)
    assert kinds(out) == [] and [e['type'] for e in out['sequences']['candidates']] == ['possible_turnover']
    assert out['sequences']['candidates'][0]['suggested_review_type'] == 'interception'


def test_two_opponents_equally_close_is_a_possible_duel_with_the_next_touch_reported():
    pair = [person(100, 'T1', 'kit-A'), person(150, 'T2', 'kit-B')]
    out = run([pair+[ball(125)]]*4+[pair+[ball(150)]]*3)
    assert kinds(out) == ['possible_duel']
    duel = out['sequences']['actions']['candidates'][0]
    assert {duel['from_track'], duel['to_track']} == {'T1', 'T2'}
    assert duel['next_touch_track'] == 'T2' and duel['contested_frames'] == 4
    lone = run([pair+[ball(125)]]*4)
    assert lone['sequences']['actions']['candidates'][0]['next_touch_track'] is None
    # One contested frame is not enough.
    assert kinds(run([pair+[ball(125)]]+[pair]*3)) == []


def test_keeping_the_ball_past_a_close_opponent_is_a_possible_dribble():
    frames = [[person(x, 'T1', 'kit-A'), person(300, 'T2', 'kit-B', foot=260), ball(x+5)] for x in range(100, 520, 20)]
    out = run(frames)
    assert kinds(out) == ['possible_dribble']
    dribble = out['sequences']['actions']['candidates'][0]
    assert (dribble['from_track'], dribble['to_track']) == ('T1', 'T2') and dribble['opponent_closest_heights'] == .6
    # No opponent nearby, or the opponent never falls away: nothing is proposed.
    assert kinds(run([[person(x, 'T1', 'kit-A'), person(300, 'T2', 'kit-B', foot=560), ball(x+5)] for x in range(100, 520, 20)])) == []
    assert kinds(run([[person(x, 'T1', 'kit-A'), person(x+20, 'T2', 'kit-B', foot=260), ball(x+5)] for x in range(100, 520, 20)])) == []


def shot_frames(extra, speed=100):
    people = [person(100, 'T1', 'kit-A')]+extra
    return [people+[ball(100)]]*3+[people+[ball(100+speed*i)] for i in range(1, 6)]


def test_fast_ball_at_a_goalkeeper_is_a_possible_shot():
    keeper = person(900, 'G1', 'kit-B', role='goalkeeper')
    out = run(shot_frames([keeper]))
    assert kinds(out) == ['possible_shot']
    shot = out['sequences']['actions']['candidates'][0]
    assert shot['from_track'] == 'T1' and shot['to_track'] is None
    assert shot['basis'] == 'towards_goalkeeper' and shot['goalkeeper_track'] == 'G1' and shot['ball_speed_heights_per_second'] == 10
    # No goalkeeper and no pitch map, a slow ball, or a keeper off to the side: not proposed.
    assert kinds(run(shot_frames([]))) == []
    assert kinds(run(shot_frames([keeper], speed=40))) == []
    assert kinds(run(shot_frames([person(300, 'G1', 'kit-B', foot=590, role='goalkeeper')]))) == []


def test_fast_ball_taken_by_a_team_mate_is_a_pass_not_a_shot():
    people = [person(100, 'T1', 'kit-A'), person(600, 'T3', 'kit-A'), person(900, 'G1', 'kit-B', role='goalkeeper')]
    frames = [people+[ball(100)]]*3+[people+[ball(100+100*i)] for i in range(1, 6)]+[people+[ball(600)]]*3
    out = run(frames)
    assert kinds(out) == [] and [e['type'] for e in out['sequences']['candidates']] == ['possible_pass']


def test_fast_ball_at_a_goal_on_the_pitch_map_is_a_possible_shot():
    # A plain scaling stands in for a calibration: 10 px per metre, so x = 800 px is 80 m and the far goal is at 1050 px.
    record = dict(status='calibrated', matrix=[.1, 0, 0, 0, .1, 0, 0, 0, 1])
    people = [person(800, 'T1', 'kit-A', foot=340)]
    frames = [people+[ball(800, 340)]]*3+[people+[ball(800+80*i, 340)] for i in range(1, 4)]
    out = run(frames, pitch={i: record for i in range(len(frames))})
    assert kinds(out) == ['possible_shot']
    shot = out['sequences']['actions']['candidates'][0]
    assert shot['basis'] == 'towards_goal_on_pitch_map' and shot['estimated_distance_m'] == 25
    # Heading straight across the pitch, or with no pitch map: not proposed.
    across = [people+[ball(800, 340)]]*3+[people+[ball(800, 340-80*i)] for i in range(1, 4)]
    assert kinds(run(across, pitch={i: record for i in range(len(across))})) == []
    assert kinds(run(frames)) == []


def test_confirmed_stoppage_blocks_candidates_and_counts_are_reported():
    pair = [person(100, 'T1', 'kit-A'), person(150, 'T2', 'kit-B')]
    frames = [pair+[ball(100)]]*3+[pair+[ball(150)]]*3
    out = run(frames)
    assert out['sequences']['actions']['counts'] == dict(shot=0, tackle=1, duel=0, dribble=0)
    assert any(note['kind'] == 'actions' for note in out['findings'])
    stopped = run(frames, context=dict(dead_ball_intervals=[[0., .6]], kit_mappings={}, revision=1))
    assert kinds(stopped) == []
