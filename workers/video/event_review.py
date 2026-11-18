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


def merge_intervals(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return merged


def summarize(state,duration):
    events=[e for e in state['events'] if e['review_status']=='approved']
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
            turnovers='Possession lost is incomplete passes plus unsuccessful dribbles. Possession won is interceptions plus tackles won. Loose balls, clearances and fouls are not recorded, so neither is a full turnover count.'),
        players=[dict(team=side,player=player,event_count=sum(counts.values()),counts=dict(counts),
                      outcomes={kind:dict(outcomes) for kind,outcomes in sorted(player_outcomes[side,player].items())}) for (side,player),counts in sorted(players.items())],
        possession=dict(home_seconds=round(possession['home'],3),away_seconds=round(possession['away'],3),
            unknown_seconds=round(max(0,duration-known),3),home_share_pct=percentage(possession['home'],known),
            away_share_pct=percentage(possession['away'],known),known_coverage_pct=percentage(known,duration)))


def empty_review(manifest):
    state=dict(schema_version='event-review-1',run_id=manifest['run_id'],revision=0,teams=dict(home='Home',away='Away'),
               source_sha256=manifest['metadata'].get('sha256'),coverage=[],events=[],possession_intervals=[])
    state['summary']=summarize(state,manifest['metadata']['duration_seconds'])
    return state


def load_review(path,manifest):
    if not path.is_file():return empty_review(manifest)
    state=json.loads(path.read_text(encoding='utf-8'))
    if state['run_id']!=manifest['run_id'] or state.get('source_sha256')!=manifest['metadata'].get('sha256'):
        raise ValueError('Review source does not match this video')
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
                 updated_at=datetime.now(timezone.utc).isoformat(),source='human_annotation')
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
        player=row.get('player')
        state['events'].append(dict(id=key,timestamp_seconds=float(timestamp),type=kind,outcome=outcome,team=row['team'],
            player=text(player,'player',100,True) or None if player is not None else None,
            review_status=row['review_status'],notes=text(row.get('notes',''),'notes',1000,True),source='human_annotation'))
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
