"""Possible shots, tackles, duels and dribbles, proposed for a person to review.

These are rules over the tracked ball and players, not a trained event model, and none
has been checked against labelled events. Each one is a prompt to watch a moment; nothing
here is counted as an event until a person records it under Review.

Distances are in player heights (the height of the player's box in the image), so the
rules work without a pitch calibration. One height is roughly 1.8 m near that player.
"""
from collections import Counter
import math

VERSION = "event-candidates-1"

DUEL_MIN_FRAMES = 2
DUEL_MIN_SECONDS = .15
DUEL_REACH = .9  # both players within this many heights of the ball
DUEL_OUTCOME_SECONDS = 2.  # the next controlled touch this soon after decides who came away with it
TACKLE_MAX_GAP_SECONDS = 1.5
TACKLE_MAX_PLAYER_DISTANCE = 1.5  # the two players' feet came this close, in heights
TACKLE_MAX_BALL_TRAVEL = 3.  # further than this the ball was played, not taken
DRIBBLE_MIN_SECONDS = .8
DRIBBLE_CHAIN_GAP_SECONDS = 1.
DRIBBLE_CLOSE = 1.  # an opponent came this close to the carrier
DRIBBLE_CLEAR = 1.5  # and was this far away again while the carrier still had the ball
DRIBBLE_KEEP_SECONDS = .4
SHOT_MIN_SPEED = 7.  # heights per second, about 12 m/s
SHOT_WINDOW_SECONDS = (.15, .5)  # the ball's speed is taken this long after it leaves the foot
SHOT_KEEPER_ANGLE = 20.  # degrees between the ball's direction and the goalkeeper
SHOT_KEEPER_RANGE = (1.5, 10.)  # heights. Image distance understates depth, so this is further than 18 m on the pitch
SHOT_MAX_DISTANCE_M = 35.
SHOT_VETO_DISTANCE_M = 45.  # a pitch map that puts the player this far from the goal ahead rules a shot out
SHOT_GOAL_MARGIN_M = 5.  # either side of the 7.32 m goal, for calibration error
LIMITATIONS = [
    "Rules over tracked positions, not a trained event model. None has been checked against labelled events.",
    "They need the ball to be seen and both players to be tracked and sorted into kit groups, so most real events are missed. A count here is not a match total.",
    "A shot needs a fast ball heading at a detected goalkeeper, or at a goal on the pitch map. A fast pass towards the goal looks the same.",
    "A tackle is the ball changing kit group at close range; a duel is two opponents equally close to the ball. Fouls, blocks and loose balls can look the same.",
    "A dribble is a player keeping the ball while an opponent comes close and falls away again. It does not know whether the opponent tried to win it.",
]


def _heights(ax, ay, bx, by, height):
    return math.hypot(ax-bx, ay-by)/max(height, 1.)


def _player(frame, track, scene):
    return next((p for p in frame['players'] if p['track'] == track and p['scene'] == scene), None)


def detect_actions(evidence, anchors, resolved, metadata, sample_fps, context=None, pitch=None):
    """`evidence` and `anchors` are the per-frame records and controlled touches built in sequences.py."""
    context = context or {}
    duration = metadata['duration_seconds']
    step = 1/sample_fps
    stoppages = context.get('dead_ball_intervals', [])
    stopped = lambda start, end: any(x < end and y > start for x, y in stoppages)
    found = []

    def add(kind, t, start, end, scene, a, b, text, **extra):
        # a and b are (track, kit, team) or None. For a shot or dribble, a is the player on the ball.
        a, b = a or (None, None, None), b or (None, None, None)
        found.append(dict(type='possible_'+kind, timestamp_seconds=round(t, 4), start_seconds=round(start, 4), end_seconds=round(end, 4),
            evidence_start_seconds=round(max(0, start-1.5), 3), evidence_end_seconds=round(min(duration, end+1.5), 3), scene_id=scene,
            from_track=a[0], to_track=b[0], from_kit=a[1], to_kit=b[1], from_team=a[2], to_team=b[2],
            evidence_level='rule_based_review_only', possession_included=False, suggested_review_type=kind,
            review_status='unreviewed', source=VERSION, evidence=text, **extra))

    def who(p):
        kit, team = resolved(p)
        return p['track'], kit, team

    # Tackles: the ball goes from one kit group's controlled touch to the other's at close range.
    tackles = []
    for a, b in zip(anchors, anchors[1:]):
        gap = b['start_seconds']-a['end_seconds']
        if (a['scene_id'] != b['scene_id'] or not a['team'] or not b['team'] or a['team'] == b['team']
                or gap < -1e-7 or gap > TACKLE_MAX_GAP_SECONDS or stopped(a['end_seconds'], b['start_seconds'])):
            continue
        first = max(a['indices'][0], a['indices'][-1]-max(1, round(.5/step)))
        closest = None
        for f in evidence[first:b['indices'][0]+1]:
            loser, winner = _player(f, a['track_id'], a['scene_id']), _player(f, b['track_id'], a['scene_id'])
            if loser and winner:
                d = _heights(loser['x'], loser['y'], winner['x'], winner['y'], loser['height'])
                closest = d if closest is None else min(closest, d)
        if closest is None or closest > TACKLE_MAX_PLAYER_DISTANCE:
            continue
        travel = _heights(a['last']['ball']['x'], a['last']['ball']['y'], b['first']['ball']['x'], b['first']['ball']['y'], a['height'])
        if travel > TACKLE_MAX_BALL_TRAVEL:
            continue
        tackles.append((a['last']['time'], b['start_seconds']))
        add('tackle', b['start_seconds'], a['last']['time'], b['start_seconds'], a['scene_id'],
            (a['track_id'], a['kit'], a['team']), (b['track_id'], b['kit'], b['team']),
            'The ball went from one kit group to the other while the two players were close together. The second player is the one who came away with it.',
            players_closest_heights=round(closest, 2), ball_travel_heights=round(travel, 2), gap_seconds=round(max(0, gap), 3))

    # Duels: a run of frames where two opponents are about equally close to the ball.
    run = []
    def close_duel():
        nonlocal run
        frames, run = run, []
        if not frames:
            return
        start, end = frames[0][0]['time'], frames[-1][0]['time']+frames[-1][0]['duration']
        if len(frames) < DUEL_MIN_FRAMES or end-start < DUEL_MIN_SECONDS-1e-8 or stopped(start, end):
            return
        if any(start <= b+.1 and end >= a-.5 for a, b in tackles):
            return  # already listed as a tackle
        (first_track, second_track), _ = Counter(pair for _, pair in frames).most_common(1)[0]
        sample = next(f for f, pair in frames if pair == (first_track, second_track))
        one, two = who(_player(sample, first_track, sample['scene'])), who(_player(sample, second_track, sample['scene']))
        winner = next((a for a in anchors if a['scene_id'] == sample['scene'] and a['team']
                       and end-1e-7 <= a['start_seconds'] <= end+DUEL_OUTCOME_SECONDS), None)
        add('duel', start, start, end, sample['scene'], one, two,
            'Two players from different kit groups were about equally close to the ball.'
            + (' The next controlled touch after it is given as who came away with it.' if winner else ' No controlled touch followed soon enough to say who came away with it.'),
            contested_seconds=round(end-start, 3), contested_frames=len(frames),
            next_touch_team=winner['team'] if winner else None, next_touch_track=winner['track_id'] if winner else None)
    for f in evidence:
        pair = None
        if f['contested']:
            ball = f['ball']
            ranked = sorted((_heights(p['x'], p['y'], ball['x'], ball['y'], p['height']), i, p) for i, p in enumerate(f['players'])
                            if p['track'] and p['confidence'] >= .25 and p['scene'] == ball['scene'])
            near = ranked[0][2]
            rival = next((p for d, _, p in ranked[1:] if d <= DUEL_REACH and resolved(p)[1] and resolved(near)[1] and resolved(p)[1] != resolved(near)[1]), None)
            if rival:
                pair = tuple(sorted((near['track'], rival['track'])))
        if run and (not pair or f['scene'] != run[-1][0]['scene'] or f['time']-run[-1][0]['time'] > 1.5*step+1e-8):
            close_duel()
        if pair:
            run.append((f, pair))
    close_duel()

    # Dribbles: one player's touches in a row, with an opponent coming close and falling away while the player keeps the ball.
    chains = []
    for a in anchors:
        last = chains[-1][-1] if chains else None
        if (last and last['track_id'] == a['track_id'] and last['scene_id'] == a['scene_id']
                and a['start_seconds']-last['end_seconds'] <= DRIBBLE_CHAIN_GAP_SECONDS):
            chains[-1].append(a)
        else:
            chains.append([a])
    for chain in chains:
        head, tail = chain[0], chain[-1]
        start, end = head['start_seconds'], tail['end_seconds']
        if not head['team'] or end-start < DRIBBLE_MIN_SECONDS-1e-8 or stopped(start, end):
            continue
        nearest = {}  # opponent track -> (distance, time, player) at its closest
        seen = []
        for f in evidence[head['indices'][0]:tail['indices'][-1]+1]:
            carrier = _player(f, head['track_id'], head['scene_id'])
            if not carrier:
                continue
            for p in f['players']:
                if not p['track'] or p['scene'] != head['scene_id'] or p['confidence'] < .25:
                    continue
                team = resolved(p)[1]
                if not team or team == head['team']:
                    continue
                d = _heights(p['x'], p['y'], carrier['x'], carrier['y'], carrier['height'])
                seen.append((f['time'], p['track'], d))
                if p['track'] not in nearest or d < nearest[p['track']][0]:
                    nearest[p['track']] = (d, f['time'], p)
        beaten = [(t, track, d, p) for track, (d, t, p) in nearest.items()
                  if d <= DRIBBLE_CLOSE and end-t >= DRIBBLE_KEEP_SECONDS-1e-8
                  and any(track == other and later > t and far >= DRIBBLE_CLEAR for later, other, far in seen)]
        if not beaten:
            continue
        t, track, d, p = min(beaten)
        add('dribble', t, start, end, head['scene_id'], (head['track_id'], head['kit'], head['team']), who(p),
            'One player kept the ball while an opponent came close and fell away again. The first player is the one on the ball.',
            opponent_closest_heights=round(d, 2), carried_seconds=round(end-start, 3), opponents_passed=len(beaten))

    # Shots: the ball leaves a controlled touch fast, heading at a goalkeeper or at a goal on the pitch map.
    for index, a in enumerate(anchors):
        origin, shooter = a['last']['ball'], a['last']['actor']
        t0, height = a['last']['time'], a['height']
        following = anchors[index+1] if index+1 < len(anchors) else None
        flight, previous = [], a['last']
        for f in evidence[a['indices'][-1]+1:]:
            if f['time']-t0 > SHOT_WINDOW_SECONDS[1]+1e-8 or (f['scene'] is not None and f['scene'] != a['scene_id']):
                break
            if f['actor'] and f['actor']['track'] != a['track_id']:
                break  # someone else has it
            if not f['ball']:
                continue
            if f['time']-previous['time'] > 2.5*step+1e-8:
                break
            flight.append(f)
            previous = f
        flight = [f for f in flight if f['time']-t0 >= SHOT_WINDOW_SECONDS[0]-1e-8]
        if not flight or stopped(t0, flight[-1]['time']):
            continue
        end = flight[-1]
        elapsed = end['time']-t0
        vx, vy = (end['ball']['x']-origin['x'])/elapsed, (end['ball']['y']-origin['y'])/elapsed
        speed = math.hypot(vx, vy)/height
        if speed < SHOT_MIN_SPEED:
            continue
        # A team-mate taking the ball next makes it a pass.
        if (following and following['scene_id'] == a['scene_id'] and a['team'] and following['team'] == a['team']
                and following['track_id'] != a['track_id'] and following['start_seconds']-t0 <= 4):
            continue
        basis, extra = None, {}
        for p in a['last']['players']:
            if p.get('role') != 'goalkeeper' or p['track'] == a['track_id'] or p['scene'] != a['scene_id']:
                continue
            dx, dy = p['x']-origin['x'], p['y']-p['height']/2-origin['y']  # aim at the keeper's middle, not the feet
            away = math.hypot(dx, dy)/height
            if not SHOT_KEEPER_RANGE[0] <= away <= SHOT_KEEPER_RANGE[1]:
                continue
            cosine = (vx*dx+vy*dy)/(math.hypot(vx, vy)*math.hypot(dx, dy))
            if cosine >= math.cos(math.radians(SHOT_KEEPER_ANGLE)):
                basis, extra = 'towards_goalkeeper', dict(goalkeeper_track=p['track'], goalkeeper_distance_heights=round(away, 1))
                break
        too_far = False
        record = (pitch or {}).get(a['last'].get('frame'))
        if record and record.get('status') == 'calibrated':
            from workers.video import pitch_keypoints as pk
            # The shooter's feet are on the ground; the ball in flight is not, so its mapped position is only a direction.
            xy, usable = pk.to_pitch(record, [(shooter['x'], shooter['y']), (end['ball']['x'], end['ball']['y'])])
            if usable.all():
                (x0, y0), (x1, y1) = xy
                for goal_x in (0., pk.LENGTH_M):
                    if (goal_x-x0)*(x1-x0) <= 1e-6:
                        continue
                    crossing = y0+(y1-y0)*(goal_x-x0)/(x1-x0)
                    distance = math.hypot(goal_x-x0, pk.WIDTH_M/2-y0)
                    too_far = distance > SHOT_VETO_DISTANCE_M
                    if abs(crossing-pk.WIDTH_M/2) <= 3.66+SHOT_GOAL_MARGIN_M and distance <= SHOT_MAX_DISTANCE_M:
                        basis = basis or 'towards_goal_on_pitch_map'
                        extra.update(estimated_distance_m=round(float(distance), 1))
        if not basis or too_far:
            continue
        receiver = (following if following and following['scene_id'] == a['scene_id'] and following['track_id'] != a['track_id']
                    and following['start_seconds']-t0 <= 3 else None)
        add('shot', t0, t0, end['time'], a['scene_id'], (a['track_id'], a['kit'], a['team']),
            (receiver['track_id'], receiver['kit'], receiver['team']) if receiver else None,
            'The ball left this player fast, ' + ('heading at a detected goalkeeper.' if basis == 'towards_goalkeeper' else 'heading at a goal on the estimated pitch map.')
            + (' The second player is the next one seen with the ball.' if receiver else ''),
            basis=basis, ball_speed_heights_per_second=round(speed, 1), **extra)

    found.sort(key=lambda row: (row['timestamp_seconds'], row['type']))
    numbers = Counter()
    for row in found:
        kind = row['suggested_review_type']
        numbers[kind] += 1
        row['id'] = f'{kind}-{numbers[kind]}'
    return dict(schema_version=VERSION, review_status='unreviewed', accuracy_evaluated=False, candidates=found,
                counts={kind: numbers[kind] for kind in ('shot', 'tackle', 'duel', 'dribble')},
                parameters=dict(duel_min_seconds=DUEL_MIN_SECONDS, duel_reach_heights=DUEL_REACH, tackle_max_gap_seconds=TACKLE_MAX_GAP_SECONDS,
                                tackle_max_player_distance_heights=TACKLE_MAX_PLAYER_DISTANCE, tackle_max_ball_travel_heights=TACKLE_MAX_BALL_TRAVEL,
                                dribble_min_seconds=DRIBBLE_MIN_SECONDS, dribble_close_heights=DRIBBLE_CLOSE, dribble_clear_heights=DRIBBLE_CLEAR,
                                shot_min_speed_heights_per_second=SHOT_MIN_SPEED, shot_goalkeeper_angle_degrees=SHOT_KEEPER_ANGLE,
                                shot_goalkeeper_range_heights=list(SHOT_KEEPER_RANGE), shot_max_distance_m=SHOT_MAX_DISTANCE_M, shot_veto_distance_m=SHOT_VETO_DISTANCE_M, shot_goal_margin_m=SHOT_GOAL_MARGIN_M),
                limitations=LIMITATIONS)
