"""Human annotation records and explicitly scoped football-event aggregation."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from uuid import uuid4

OUTCOMES={
    'pass':{'completed','incomplete','unknown'},
    'shot':{'goal','on_target','off_target','blocked','unknown'},
    'tackle':{'won','lost','unknown'},
    'interception':{'successful','unknown'},
    'dribble':{'successful','unsuccessful','unknown'},
    'duel':{'won','lost','unknown'},
}

# Where an event happened is a click on a drawn pitch, so these are the assumed dimensions of that drawing,
# not a measurement of the real pitch. x runs left to right and y top to bottom, as the reviewer sees it.
LENGTH_M,WIDTH_M=105.,68.
PERIODS=(1,2,3,4)  # the two halves, then the two halves of extra time
BODY_PARTS={'foot','head','other'}
SITUATIONS={'open_play','free_kick','corner','penalty','throw_in','goal_kick','kick_off'}
WITH_END_POINT={'pass','shot','dribble'}
WITH_BODY_PART={'pass','shot'}
EXPORT_VERSION='1.0.0'


class ReviewConflict(ValueError):pass


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def text(value,label,maximum,allow_empty=False):
    if not isinstance(value,str) or len(value)>maximum or (not allow_empty and not value.strip()):
        raise ValueError(f'Invalid {label}')
    return value.strip()


def interval(start,end,duration):
    if not finite(start) or not finite(end) or not 0<=start<end<=duration:
        raise ValueError('Intervals must be finite, ordered and inside the source video')
    return float(start),float(end)


def point(value,label):
    if value is None:return None
    if (not isinstance(value,dict) or not finite(value.get('x')) or not finite(value.get('y'))
            or not 0<=value['x']<=LENGTH_M or not 0<=value['y']<=WIDTH_M):
        raise ValueError(f'{label} must be a point on the 105 x 68 m pitch')
    return dict(x=round(float(value['x']),2),y=round(float(value['y']),2))


def choice(value,allowed,label):
    if value is None or value=='':return None
    if not isinstance(value,str) or value not in allowed:raise ValueError(f'Invalid {label}')
    return value


def default_periods():
    return {str(p):dict(home_attacks=None,reference_video_seconds=None,reference_clock_seconds=None) for p in PERIODS}


def clean_periods(payload,duration):
    """Per period: which goal the home team attacks, and one moment where the match clock is known."""
    periods=default_periods()
    if not isinstance(payload,dict) or not set(payload)<=set(periods):raise ValueError('Invalid periods')
    for key,row in payload.items():
        if not isinstance(row,dict):raise ValueError('Invalid period')
        video,clock=row.get('reference_video_seconds'),row.get('reference_clock_seconds')
        if (video is None)!=(clock is None):raise ValueError('A match-clock reference needs both a video time and a clock time')
        if video is not None and (not finite(video) or not finite(clock) or not 0<=video<=duration or not 0<=clock<=4*3600):
            raise ValueError('Invalid match-clock reference')
        periods[key]=dict(home_attacks=choice(row.get('home_attacks'),{'left','right'},'direction of play'),
            reference_video_seconds=None if video is None else float(video),reference_clock_seconds=None if clock is None else float(clock))
    return periods


def attacking(location,team,period,periods):
    """The point turned so the event's team attacks towards x = LENGTH_M. None when the direction of play is not recorded."""
    direction=periods[str(period)]['home_attacks']
    if location is None or direction is None:return None
    if (direction=='right')==(team=='home'):return location
    return dict(x=round(LENGTH_M-location['x'],2),y=round(WIDTH_M-location['y'],2))


def progressive(start,end):
    """Wyscout's published rule for a progressive pass, on points where the team attacks towards x = LENGTH_M."""
    goal=(LENGTH_M,WIDTH_M/2)
    gain=math.dist((start['x'],start['y']),goal)-math.dist((end['x'],end['y']),goal)
    own_start,own_end=start['x']<LENGTH_M/2,end['x']<LENGTH_M/2
    return gain>=(30 if own_start and own_end else 10 if not own_start and not own_end else 15)


def match_clock(event,periods):
    reference=periods[str(event.get('period',1))]
    if reference['reference_video_seconds'] is None:return None
    return round(reference['reference_clock_seconds']+event['timestamp_seconds']-reference['reference_video_seconds'],3)


def merge_intervals(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return merged


def summarize(state,duration):
    events=[e for e in state['events'] if e['review_status']=='approved']
    periods=state.get('periods') or default_periods()
    reviewed=sum(b-a for a,b in merge_intervals(state['coverage']))
    def percentage(a,b):return round(100*a/b,2) if b else None
    teams={}
    players={}
    player_outcomes={}
    for side in ['home','away']:
        rows=[e for e in events if e['team']==side]
        if not rows:
            teams[side]=None
            continue
        counts=Counter((e['type'],e['outcome']) for e in rows)
        total=lambda kind:sum(n for (k,_),n in counts.items() if k==kind)
        # Rates divide by records with a known outcome only; an unknown outcome is never counted as a failure.
        def decided(kind,good,bad,good_key,bad_key,rate_key):
            return {'recorded':total(kind),good_key:counts[kind,good],bad_key:counts[kind,bad],
                    'known_outcomes':counts[kind,good]+counts[kind,bad],rate_key:percentage(counts[kind,good],counts[kind,good]+counts[kind,bad])}
        on_target=counts['shot','goal']+counts['shot','on_target']
        shots_known=total('shot')-counts['shot','unknown']
        teams[side]=dict(name=state['teams'][side],event_count=len(rows),
            passes=dict(recorded=total('pass'),completed=counts['pass','completed'],
                known_outcomes=counts['pass','completed']+counts['pass','incomplete'],
                outcome_coverage_pct=percentage(counts['pass','completed']+counts['pass','incomplete'],total('pass')),
                completion_pct=percentage(counts['pass','completed'],counts['pass','completed']+counts['pass','incomplete'])),
            shots=dict(recorded=total('shot'),on_target=on_target,goals=counts['shot','goal'],off_target=counts['shot','off_target'],
                blocked=counts['shot','blocked'],known_outcomes=shots_known,
                on_target_pct=percentage(on_target,shots_known),conversion_pct=percentage(counts['shot','goal'],shots_known)),
            tackles=decided('tackle','won','lost','won','lost','win_pct'),
            interceptions=total('interception'),
            dribbles=decided('dribble','successful','unsuccessful','successful','unsuccessful','success_pct'),
            duels=decided('duel','won','lost','won','lost','win_pct'),
            turnovers=dict(conceded=counts['pass','incomplete']+counts['dribble','unsuccessful'],
                incomplete_passes=counts['pass','incomplete'],unsuccessful_dribbles=counts['dribble','unsuccessful'],
                won=total('interception')+counts['tackle','won'],tackles_won=counts['tackle','won']))
        # A pass counts towards progression only with a start, an end and a recorded direction of play.
        passes=[e for e in rows if e['type']=='pass']
        turned=[(attacking(e.get('location'),side,e.get('period',1),periods),attacking(e.get('end_location'),side,e.get('period',1),periods)) for e in passes]
        turned=[(a,b) for a,b in turned if a and b]
        teams[side]['progression']=dict(
            shots_located=sum(1 for e in rows if e['type']=='shot' and e.get('location')),
            passes_located=sum(1 for e in passes if e.get('location') and e.get('end_location')),
            passes_with_direction=len(turned),
            progressive_passes=sum(progressive(a,b) for a,b in turned) if turned else None)
        for e in rows:
            if e.get('player'):
                key=(side,e['player']);counts_player=players.setdefault(key,Counter());counts_player[e['type']]+=1
                player_outcomes.setdefault(key,{}).setdefault(e['type'],Counter())[e['outcome']]+=1
    possession=Counter()
    for row in state['possession_intervals']:
        if row['review_status']=='approved':possession[row['team']]+=row['end_seconds']-row['start_seconds']
    known=possession['home']+possession['away']
    return dict(scope='approved_annotations_only',summary_version=2,approved_events=len(events),draft_events=len(state['events'])-len(events),
        reviewed_seconds=round(reviewed,3),reviewed_video_pct=percentage(reviewed,duration),teams=teams,
        definitions=dict(
            scope='Counts of saved, approved records for this video only. They cover what a person recorded, not the whole match.',
            rates='Each rate divides by records with a known outcome. Records with an unknown outcome are counted as recorded but left out of rates.',
            shots='On target includes goals. Conversion is goals divided by shots with a known outcome.',
            turnovers='Possession lost is incomplete passes plus unsuccessful dribbles. Possession won is interceptions plus tackles won. Loose balls, clearances and fouls are not recorded, so neither is a full turnover count.',
            locations='A location is where the reviewer clicked on a drawn 105 x 68 m pitch. It is a judgement by eye, not a measurement.',
            progressive_passes='A pass is progressive when it ends closer to the opponent goal by at least 30 m with both ends in the own half, 15 m when it crosses halfway, or 10 m with both ends in the opponent half (the rule Wyscout publishes). It is counted only for passes with a start, an end and a recorded direction of play, whether or not the pass was completed.'),
        players=[dict(team=side,player=player,event_count=sum(counts.values()),counts=dict(counts),
                      outcomes={kind:dict(outcomes) for kind,outcomes in sorted(player_outcomes[side,player].items())}) for (side,player),counts in sorted(players.items())],
        possession=dict(home_seconds=round(possession['home'],3),away_seconds=round(possession['away'],3),
            unknown_seconds=round(max(0,duration-known),3),home_share_pct=percentage(possession['home'],known),
            away_share_pct=percentage(possession['away'],known),known_coverage_pct=percentage(known,duration)))


def empty_review(manifest):
    state=dict(schema_version='event-review-2',run_id=manifest['run_id'],revision=0,teams=dict(home='Home',away='Away'),
               source_sha256=manifest['metadata'].get('sha256'),periods=default_periods(),coverage=[],events=[],possession_intervals=[])
    state['summary']=summarize(state,manifest['metadata']['duration_seconds'])
    return state


def load_review(path,manifest):
    if not path.is_file():return empty_review(manifest)
    state=json.loads(path.read_text(encoding='utf-8'))
    if state['run_id']!=manifest['run_id'] or state.get('source_sha256')!=manifest['metadata'].get('sha256'):
        raise ValueError('Review source does not match this video')
    # A review saved before locations and periods existed reads as one with none recorded.
    state.setdefault('periods',default_periods())
    for event in state['events']:
        event.setdefault('period',1)
        for key in ('location','end_location','body_part','situation','receiver'):event.setdefault(key,None)
    state['summary']=summarize(state,manifest['metadata']['duration_seconds'])
    return state


def save_review(path,manifest,payload):
    current=load_review(path,manifest)
    revision=payload.get('revision')
    if not isinstance(revision,int) or isinstance(revision,bool):raise ValueError('Review revision is required')
    if revision!=current['revision']:raise ReviewConflict('Review changed in another session. Reload before saving.')
    duration=manifest['metadata']['duration_seconds']
    teams=payload.get('teams',current['teams'])
    if not isinstance(teams,dict):raise ValueError('Invalid teams')
    state=empty_review(manifest)
    state.update(revision=revision+1,teams={side:text(teams.get(side),'team name',80) for side in ['home','away']},
                 updated_at=datetime.now(timezone.utc).isoformat(),source='human_annotation',
                 periods=clean_periods(payload.get('periods',current['periods']),duration))
    coverage=payload.get('coverage',[])
    events=payload.get('events',[])
    possession=payload.get('possession_intervals',[])
    if not all(isinstance(x,list) for x in [coverage,events,possession]) or len(coverage)>200 or len(events)>1000 or len(possession)>1000:
        raise ValueError('Invalid annotation collection or too many records')
    state['coverage']=[]
    for pair in coverage:
        if not isinstance(pair,list) or len(pair)!=2:raise ValueError('Coverage requires start/end pairs')
        state['coverage'].append(list(interval(*pair,duration)))
    identifiers=set()
    def identity(row):
        value=row.get('id')
        if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value) or value in identifiers:
            raise ValueError('Annotation IDs must be valid and unique')
        identifiers.add(value)
        if row.get('review_status') not in {'draft','approved'}:raise ValueError('Review status must be draft or approved')
        return value
    for row in events:
        if not isinstance(row,dict):raise ValueError('Invalid event')
        key=identity(row);kind=row.get('type');outcome=row.get('outcome')
        if kind not in OUTCOMES or outcome not in OUTCOMES[kind]:raise ValueError('Invalid event type or outcome')
        timestamp=row.get('timestamp_seconds')
        if not finite(timestamp) or not 0<=timestamp<duration:raise ValueError('Event timestamp must be inside the source video')
        if row.get('team') not in {'home','away'}:raise ValueError('Event team is required')
        player=row.get('player');receiver=row.get('receiver');period=row.get('period',1)
        if isinstance(period,bool) or period not in PERIODS:raise ValueError('Invalid period')
        end=point(row.get('end_location'),'End point')
        if end and kind not in WITH_END_POINT:raise ValueError('Only passes, shots and dribbles have an end point')
        if receiver and kind!='pass':raise ValueError('Only a pass has a receiver')
        if (row.get('body_part') or row.get('situation')) and kind not in WITH_BODY_PART:raise ValueError('Only passes and shots have a body part or situation')
        state['events'].append(dict(id=key,timestamp_seconds=float(timestamp),type=kind,outcome=outcome,team=row['team'],
            player=text(player,'player',100,True) or None if player is not None else None,
            review_status=row['review_status'],notes=text(row.get('notes',''),'notes',1000,True),source='human_annotation',
            period=period,location=point(row.get('location'),'Location'),end_location=end,
            body_part=choice(row.get('body_part'),BODY_PARTS,'body part'),situation=choice(row.get('situation'),SITUATIONS,'situation'),
            receiver=text(receiver,'receiver',100,True) or None if receiver is not None else None))
    approved=[]
    for row in possession:
        if not isinstance(row,dict):raise ValueError('Invalid possession interval')
        key=identity(row);start,end=interval(row.get('start_seconds'),row.get('end_seconds'),duration)
        if row.get('team') not in {'home','away','unknown'}:raise ValueError('Invalid possession team')
        if row['review_status']=='approved':approved.append((start,end))
        state['possession_intervals'].append(dict(id=key,start_seconds=start,end_seconds=end,team=row['team'],
            review_status=row['review_status'],source='human_annotation'))
    for previous,current_interval in zip(sorted(approved),sorted(approved)[1:]):
        if current_interval[0]<previous[1]:raise ValueError('Approved possession intervals cannot overlap')
    state['summary']=summarize(state,duration)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)
    return state


def approved_export(state,manifest):
    """Saved, approved records in a versioned form for other tools. Drafts and model predictions are left out."""
    duration=manifest['metadata']['duration_seconds']
    periods=state.get('periods') or default_periods()
    events=sorted((e for e in state['events'] if e['review_status']=='approved'),key=lambda e:(e.get('period',1),e['timestamp_seconds'],e['id']))
    used={e.get('period',1) for e in events}
    def turned(e,key):return attacking(e.get(key),e['team'],e.get('period',1),periods)
    return dict(schema='laas.approved-events',schema_version=EXPORT_VERSION,generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Saved, approved human annotations for one video. Drafts and model predictions are not included. Totals cover what was recorded, not the whole match.',
        source=dict(run_id=state['run_id'],filename=manifest['metadata'].get('filename'),sha256=manifest['metadata'].get('sha256'),duration_seconds=duration),
        review=dict(revision=state['revision'],updated_at=state.get('updated_at')),
        teams=[dict(team_id=side,name=state['teams'][side]) for side in ['home','away']],
        coordinate_system=dict(unit='metre',pitch_length=LENGTH_M,pitch_width=WIDTH_M,pitch_dimensions='assumed, not measured',
            origin='top-left corner of the pitch as drawn in the review tool',x='left to right',y='top to bottom',
            location='where the reviewer clicked for the event; a judgement by eye',
            location_attacking="the same point turned so the event's team attacks towards x = pitch_length; null when the direction of play is not recorded for that period"),
        periods=[dict(period_id=p,**periods[str(p)]) for p in PERIODS
                 if p in used or periods[str(p)]['home_attacks'] or periods[str(p)]['reference_video_seconds'] is not None],
        events=[dict(event_id=e['id'],period_id=e.get('period',1),video_timestamp_seconds=e['timestamp_seconds'],match_clock_seconds=match_clock(e,periods),
            team_id=e['team'],team_name=state['teams'][e['team']],player_name=e.get('player'),receiver_name=e.get('receiver'),
            type=e['type'],result=e['outcome'],body_part=e.get('body_part'),situation=e.get('situation'),
            location=e.get('location'),end_location=e.get('end_location'),
            location_attacking=turned(e,'location'),end_location_attacking=turned(e,'end_location'),
            notes=e.get('notes',''),review=dict(status='approved',source='human_annotation')) for e in events],
        possession_intervals=[dict(interval_id=r['id'],start_seconds=r['start_seconds'],end_seconds=r['end_seconds'],team_id=r['team'])
                              for r in state['possession_intervals'] if r['review_status']=='approved'],
        reviewed_coverage=[dict(start_seconds=a,end_seconds=b) for a,b in merge_intervals(state['coverage'])],
        summary=summarize(state,duration))
