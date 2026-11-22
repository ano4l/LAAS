"""Temporal control/flight reasoning; model estimates never become approvals.

All geometry is in the camera image. Brief missing-ball gaps may support an
estimated interval, but no ball coordinates are generated for those gaps.
"""
from collections import Counter, defaultdict
import math

VERSION = "temporal-possession-2"
RETENTION_MAX_GAP_SECONDS = 5.0


def percentage(n, d):
    return round(100*n/d, 2) if d else None


def infer_sequences(frames, metadata, sample_fps, context=None, pitch=None):
    context = context or {}
    duration = metadata['duration_seconds']
    diagonal = math.hypot(metadata['width'], metadata['height'])
    step = 1/sample_fps
    # A stable anonymous group can be recovered retrospectively for one track,
    # but identities are never joined across tracks/scenes by colour alone.
    votes = defaultdict(Counter)
    for f in frames:
        for p in f['players']:
            if p['track'] and p['kit']:
                votes[p['scene'], p['track']][p['kit']] += 1
    groups = {}
    for key, counts in votes.items():
        kit, count = counts.most_common(1)[0]
        if count >= 3 and count/sum(counts.values()) >= .75:
            groups[key] = kit
    mapping = context.get('kit_mappings', {})

    def resolved(p):
        kit = groups.get((p['scene'], p['track']), p['kit'])
        # Unknown mapping stays scene-scoped. A user-confirmed team mapping
        # may aggregate totals across scenes; scene boundaries still stop events.
        return kit, mapping.get(kit, kit)

    evidence = []
    for f in frames:
        ball = f['ball']
        ranked = []
        if ball and ball['ground']:
            for p in f['players']:
                if p['track'] and p['confidence'] >= .25 and p['scene'] == ball['scene']:
                    distance = math.hypot(p['x']-ball['x'], p['y']-ball['y'])/p['height']
                    ranked.append((distance, p))
            ranked.sort(key=lambda item: item[0])
        actor, contested = None, False
        if ranked and ranked[0][0] <= .75:
            near, p = ranked[0]
            margin = ranked[1][0]-near if len(ranked)>1 else 1.
            opponent_near = any(d <= .9 and resolved(other)[1] and resolved(p)[1]
                                and resolved(other)[1] != resolved(p)[1] for d, other in ranked[1:])
            contested = opponent_near and margin < .2
            if not contested and margin >= .12:
                kit, team = resolved(p)
                actor = dict(p, kit=kit, team=team, distance=near, margin=margin)
        evidence.append(dict(f, actor=actor, contested=contested))

    def touch_cue(index):
        """One-touch support needs observed motion before/after near a foot."""
        f = evidence[index]
        if not f['actor'] or f['actor']['distance'] > .5 or f['actor']['margin'] < .2:
            return False
        before = next((evidence[j] for j in range(index-1, max(-1,index-5), -1)
                       if evidence[j]['ball'] and 0 < f['time']-evidence[j]['time'] <= .18), None)
        after = next((evidence[j] for j in range(index+1, min(len(evidence),index+5))
                      if evidence[j]['ball'] and 0 < evidence[j]['time']-f['time'] <= .18), None)
        if not before or not after or before['scene'] != f['scene'] or after['scene'] != f['scene']:
            return False
        a, b, c = before['ball'], f['ball'], after['ball']
        incoming = (b['x']-a['x'], b['y']-a['y'])
        outgoing = (c['x']-b['x'], c['y']-b['y'])
        ni, no = math.hypot(*incoming), math.hypot(*outgoing)
        minimum = max(2., .08*f['actor']['height'])
        if max(ni, no) < minimum or min(ni, no) < 1:
            return False
        cosine = sum(x*y for x,y in zip(incoming,outgoing))/(ni*no)
        vi, vo = ni/(f['time']-before['time']), no/(after['time']-f['time'])
        return cosine < math.cos(math.radians(35)) or min(vi,vo)/max(vi,vo) <= .45

    anchors, pending = [], []
    def close():
        nonlocal pending
        if pending:
            seconds = sum(evidence[i]['duration'] for i in pending)
            quick = any(touch_cue(i) for i in pending)
            if (len(pending) >= 2 and seconds >= .12-1e-8) or quick:
                first, last = evidence[pending[0]], evidence[pending[-1]]
                p = first['actor']
                relative_shift = math.hypot((last['ball']['x']-last['actor']['x'])-(first['ball']['x']-p['x']),
                                            (last['ball']['y']-last['actor']['y'])-(first['ball']['y']-p['y']))
                held = (relative_shift <= .3*p['height'] and relative_shift/max(seconds,1e-9) <= .8*p['height']
                        and min(evidence[i]['actor']['distance'] for i in pending) <= .55)
                if not (quick or held):
                    pending=[]
                    return
                anchors.append(dict(id=f'anchor-{len(anchors)+1}', indices=list(pending), scene_id=p['scene'],
                    track_id=p['track'], kit=p['kit'], team=p['team'],
                    start_seconds=first['time'], end_seconds=last['time']+last['duration'],
                    observed_seconds=seconds, observed_frames=len(pending),
                    support_type='motion_supported_touch' if quick else 'sustained_proximity',
                    first=first, last=last, height=p['height']))
            pending = []
    for i, f in enumerate(evidence):
        p = f['actor']
        if pending:
            previous = evidence[pending[-1]]
            a = previous['actor']
            same = (p and (p['scene'],p['track'],p['team']) == (a['scene'],a['track'],a['team'])
                    and 0 < f['time']-previous['time'] <= 1.5*step)
            if same:
                distance = math.hypot(f['ball']['x']-previous['ball']['x'],f['ball']['y']-previous['ball']['y'])
                same = distance <= diagonal*(.015+f['time']-previous['time'])
                first=evidence[pending[0]]
                relative_shift=math.hypot((f['ball']['x']-p['x'])-(first['ball']['x']-first['actor']['x']),
                                           (f['ball']['y']-p['y'])-(first['ball']['y']-first['actor']['y']))
                same = same and relative_shift <= .25*p['height']
            if not same:
                close()
        if p:
            pending.append(i)
    close()

    # States are non-overlapping source-time intervals, including all unknowns.
    states = [dict(state='contested' if f['contested'] else 'unknown', team=None, kit=None,
                   track_id=None, support='none') for f in evidence]
    for a in anchors:
        for i in a['indices']:
            states[i] = dict(state='controlled', team=a['team'],kit=a['kit'],track_id=a['track_id'],support=a['support_type'])
    rejections = Counter()
    candidates = []
    for a,b in zip(anchors,anchors[1:]):
        if a['scene_id'] != b['scene_id']:
            rejections['scene_cut'] += 1; continue
        gap = b['start_seconds']-a['end_seconds']
        same_actor = a['track_id'] == b['track_id']
        if gap < -1e-7 or gap > (.8 if same_actor else 4.):
            rejections['transition_too_long'] += 1; continue
        start_index, end_index = a['indices'][-1], b['indices'][0]
        if any(x < b['end_seconds'] and y > a['start_seconds'] for x,y in context.get('dead_ball_intervals',[])):
            rejections['confirmed_stoppage'] += 1; continue
        span = evidence[start_index:end_index+1]
        if any(f['scene'] is not None and f['scene'] != a['scene_id'] for f in span):
            rejections['scene_cut'] += 1; continue
        if any(f['contested'] for f in span):
            rejections['contested_transition'] += 1; continue
        if any(f['actor'] and f['actor']['track'] not in {a['track_id'],b['track_id']} for f in span):
            rejections['other_player_interaction'] += 1; continue
        observed = [f for f in span if f['ball']]
        total = b['first']['time']-a['last']['time']+b['first']['duration']
        support = sum(f['duration'] for f in observed)
        missing = max(0,total-support)
        longest_gap = 0.; run = 0.
        for f in span:
            run = 0. if f['ball'] else run+f['duration']
            longest_gap = max(longest_gap,run)
        strong = longest_gap <= .25+1e-8 and support/max(total,1e-9) >= .65
        review_only = (not strong and longest_gap <= .75+1e-8 and support/max(total,1e-9) >= .35 and len(observed)>=3)
        if not strong and not review_only:
            rejections['insufficient_ball_evidence'] += 1; continue
        if any(f['time']-old['time'] > (.75 if review_only else .25)+step+1e-8
               or math.hypot(f['ball']['x']-old['ball']['x'],f['ball']['y']-old['ball']['y']) > diagonal*(.015+f['time']-old['time'])
               for old,f in zip(observed,observed[1:])):
            rejections['implausible_motion'] += 1; continue
        first,last = a['last']['ball'],b['first']['ball']
        separation = math.hypot(last['x']-first['x'],last['y']-first['y'])
        if not same_actor and separation < max(.012*diagonal,.6*min(a['height'],b['height'])):
            rejections['possible_identity_split'] += 1; continue
        if not a['team'] or not b['team']:
            rejections['unknown_kit'] += 1; continue
        # Travelling ball must actually leave the origin. Nearby static-ball
        # identity changes do not become passes. A looping path stays unknown.
        length = sum(math.hypot(f['ball']['x']-old['ball']['x'],f['ball']['y']-old['ball']['y']) for old,f in zip(observed,observed[1:]))
        if not same_actor and (length < separation*.8 or length > separation*3):
            rejections['unsupported_flight_path'] += 1; continue
        same_team = a['team'] == b['team']
        if same_team and strong:
            for index in range(start_index+1,end_index):
                if states[index]['state'] == 'unknown':
                    states[index] = dict(state='in_transit' if not same_actor else 'control_gap',team=a['team'],kit=a['kit'],
                                         track_id=a['track_id'] if same_actor else None,
                                         support='bounded_missing_ball' if not evidence[index]['ball'] else 'observed_ball_flight')
        if same_actor:
            continue
        candidates.append(dict(id=f'sequence-{len(candidates)+1}',type='possible_pass' if same_team else 'possible_turnover',
            timestamp_seconds=round(a['last']['time'],4),start_seconds=round(a['last']['time'],4),end_seconds=round(b['start_seconds'],4),
            evidence_start_seconds=round(max(0,a['start_seconds']-.7),3),evidence_end_seconds=round(min(duration,b['end_seconds']+.7),3),
            scene_id=a['scene_id'],from_track=a['track_id'],to_track=b['track_id'],from_kit=a['kit'],to_kit=b['kit'],
            from_team=a['team'],to_team=b['team'],image_separation_px=round(separation,2),
            supported_control_seconds=round(a['observed_seconds']+b['observed_seconds'],3),observed_bridge_frames=max(0,len(observed)-2),
            ball_support_pct=percentage(support,total),missing_ball_seconds=round(missing,3),
            max_missing_ball_gap_seconds=round(longest_gap,3),
            receiver_support=b['support_type'],evidence_level='partial_tracking_review_only' if review_only else 'supported_transition',
            possession_included=bool(same_team and strong),suggested_review_type='pass' if same_team else 'interception',
            quality_flags=(['partial_tracking_not_used_for_possession'] if review_only else [])+(['intermittent_ball_gaps'] if missing>1e-7 else [])+(['aerial_or_outside_ground_bridge'] if any(not f['ball']['ground'] for f in observed) else []),
            review_status='unreviewed',source=VERSION,
            evidence='Observed departure and supported receiver; bounded continuity checks. Ball gaps remain unobserved; event and team possession are estimates.'))

    # Team retention: a team keeps possession from one of its controlled touches to its
    # next one, unless something in between says otherwise. This is the usual football
    # definition, but here it is an inference: an opponent touch the detector missed
    # would be wrong, so it is bounded in time, kept as its own state and totalled apart.
    retained_pairs = 0
    for a, b in zip(anchors, anchors[1:]):
        gap = b['start_seconds'] - a['end_seconds']
        if (a['scene_id'] != b['scene_id'] or not a['team'] or a['team'] != b['team']
                or gap <= 1e-7 or gap > RETENTION_MAX_GAP_SECONDS):
            continue
        if any(x < b['start_seconds'] and y > a['end_seconds'] for x, y in context.get('dead_ball_intervals', [])):
            continue
        span = range(a['indices'][-1] + 1, b['indices'][0])
        if any(evidence[i]['scene'] is not None and evidence[i]['scene'] != a['scene_id'] for i in span):
            continue
        filled = False
        for index in span:
            if states[index]['state'] == 'unknown':
                states[index] = dict(state='team_retained', team=a['team'], kit=a['kit'], track_id=None,
                                     support='same_team_touch_before_and_after')
                filled = True
        retained_pairs += filled

    # Shots, tackles, duels and dribbles are proposed from the same touches, kept apart from the pass and turnover candidates.
    from workers.video.actions import detect_actions
    actions = detect_actions(evidence, anchors, resolved, metadata, sample_fps, context, pitch)

    # Explicit user-confirmed stoppages split the state partition precisely;
    # absence of a detected ball is never treated as a stoppage.
    stoppages=context.get('dead_ball_intervals',[])
    intervals=[]
    cursor=0.
    def append(start,end,state):
        if end<=start+1e-9:return
        row=dict(start_seconds=start,end_seconds=end,**state)
        if intervals and all(intervals[-1][k]==row[k] for k in ['state','team','kit','track_id','support']) and abs(intervals[-1]['end_seconds']-start)<1e-7:
            intervals[-1]['end_seconds']=end
        else:intervals.append(row)
    unknown=dict(state='unknown',team=None,kit=None,track_id=None,support='none')
    for f,state in zip(evidence,states):
        t,end=f['time'],min(duration,f['time']+f['duration'])
        if t>cursor:append(cursor,t,unknown)
        cuts=sorted({t,end,*[x for interval in stoppages for x in interval if t<x<end]})
        for x,y in zip(cuts,cuts[1:]):
            dead=any(a<y and b>x for a,b in stoppages)
            append(x,y,dict(state='dead_ball',team=None,kit=None,track_id=None,support='confirmed_context') if dead else state)
        cursor=end
    append(cursor,duration,unknown)
    # Also apply confirmed stoppages to decoded gaps/tails, not just frames.
    raw_intervals=intervals;intervals=[]
    for row in raw_intervals:
        t,end=row['start_seconds'],row['end_seconds']
        cuts=sorted({t,end,*[x for interval in stoppages for x in interval if t<x<end]})
        state={k:v for k,v in row.items() if k not in {'start_seconds','end_seconds'}}
        for x,y in zip(cuts,cuts[1:]):
            dead=any(a<y and b>x for a,b in stoppages)
            append(x,y,dict(state='dead_ball',team=None,kit=None,track_id=None,support='confirmed_context') if dead else state)
    totals=Counter(); team_totals=Counter(); observed_control=Counter(); flights=Counter(); inferred=Counter(); retained=Counter()
    for r in intervals:
        dt=r['end_seconds']-r['start_seconds'];totals[r['state']]+=dt
        if r['state']=='team_retained':retained[r['team']]+=dt
        if r['team'] and r['state'] in {'controlled','in_transit','control_gap'}:
            team_totals[r['team']]+=dt
            (observed_control if r['state']=='controlled' else flights)[r['team']]+=dt
            if r['support']=='bounded_missing_ball':inferred[r['team']]+=dt
    known=sum(team_totals.values()); dead=totals['dead_ball']; eligible=duration-dead
    # known_seconds, coverage_pct and unknown_seconds keep their meaning (supported control,
    # flight and short gaps; everything else unknown). Retention is reported beside them,
    # never folded in, so retained time still counts inside unknown_seconds.
    with_retention=known+sum(retained.values())
    retention=dict(max_gap_seconds=RETENTION_MAX_GAP_SECONDS,retained_seconds=round(sum(retained.values()),4),
        touch_pairs=retained_pairs,known_with_retention_seconds=round(with_retention,4),
        coverage_with_retention_pct=percentage(with_retention,eligible),
        by_team=[dict(team=t,retained_seconds=round(retained[t],4),with_retention_seconds=round(team_totals[t]+retained[t],4),
                      share_with_retention_pct=percentage(team_totals[t]+retained[t],with_retention),
                      share_of_non_dead_with_retention_pct=percentage(team_totals[t]+retained[t],eligible))
                 for t in sorted(set(team_totals)|set(retained))],
        definition='Time between two consecutive controlled touches by the same team, at most max_gap_seconds apart, in one scene, with no confirmed stoppage. Inferred, not observed: a missed opponent touch in between would make it wrong.')
    return dict(retention=retention,schema_version=VERSION,review_status='unreviewed',context_revision=context.get('revision',0),
        mapping_status='user_confirmed' if mapping else 'anonymous_scene_groups',
        source_duration_seconds=duration,classified_non_dead_seconds=round(eligible,4),
        known_seconds=round(known,4),unknown_seconds=round(totals['unknown']+totals['team_retained']+sum(r['end_seconds']-r['start_seconds'] for r in intervals if r['state']=='controlled' and not r['team']),4),
        contested_seconds=round(totals['contested'],4),dead_ball_seconds=round(dead,4),coverage_pct=percentage(known,eligible),
        by_team=[dict(team=t,estimated_seconds=round(s,4),share_of_known_pct=percentage(s,known),share_of_non_dead_pct=percentage(s,eligible),
                      observed_control_seconds=round(observed_control[t],4),flight_and_gap_seconds=round(flights[t],4),bounded_missing_ball_seconds=round(inferred[t],4)) for t,s in sorted(team_totals.items())],
        intervals=[{k:round(v,4) if isinstance(v,float) else v for k,v in r.items()} for r in intervals],
        anchors=[{k:round(v,4) if isinstance(v,float) else v for k,v in a.items() if k not in {'indices','first','last','height'}} for a in anchors],
        candidates=candidates,actions=actions,diagnostics=dict(anchors=len(anchors),motion_supported_touches=sum(a['support_type']=='motion_supported_touch' for a in anchors),review_only_candidates=sum(e['evidence_level']=='partial_tracking_review_only' for e in candidates),rejected_transitions=dict(rejections)),
        parameters=dict(min_sustained_seconds=.12,min_sustained_frames=2,max_missing_ball_gap_seconds=.25,min_transition_ball_support_pct=65,max_transition_seconds=4,
                        review_only_max_gap_seconds=.75,review_only_min_ball_support_pct=35),
        limitations='Image-based, unvalidated estimates. Team shares use known classified non-dead time only; unknown/contested time is disclosed. Only explicitly confirmed stoppages are excluded. No named identities or automatic approvals; no coordinates are invented for missing balls.')
