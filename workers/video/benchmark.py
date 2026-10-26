"""Source-bound evaluation against explicit, reviewed sequence annotations."""
import math


def validate_labels(labels, metadata, allow_visual_pilot=False):
    if labels.get('schema_version')!='sequence-benchmark-1':raise ValueError('Unsupported benchmark schema')
    if not metadata.get('sha256') or labels.get('source_sha256')!=metadata['sha256']:
        raise ValueError('Labels do not match this source video SHA256')
    duration=metadata['duration_seconds']
    if isinstance(labels.get('duration_seconds'),bool) or not isinstance(labels.get('duration_seconds'),(int,float)) or not math.isfinite(labels['duration_seconds']) or abs(labels['duration_seconds']-duration)>.001:
        raise ValueError('Label duration mismatch')
    source=labels.get('annotation_source')
    if source!='human' and not (allow_visual_pilot and source=='codex_visual_pilot'):
        raise ValueError('Independent human-reviewed labels required. Unreviewed/predicted annotations are not ground truth.')
    if not isinstance(labels.get('reviewer'),str) or not labels['reviewer'].strip():raise ValueError('Reviewer/provenance required')
    if labels.get('split') not in {'development','validation','test'}:raise ValueError('Invalid dataset split')
    if not isinstance(labels.get('kit_mappings',{}),dict) or any(not isinstance(k,str) or not isinstance(v,str) or not k.strip() or not v.strip() for k,v in labels.get('kit_mappings',{}).items()):
        raise ValueError('Invalid explicit team mappings')
    def intervals(rows):
        result=[]
        for row in rows:
            if len(row)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in row) or not 0<=row[0]<row[1]<=duration:
                raise ValueError('Invalid interval')
            result.append(row)
        result.sort()
        if any(b[0]<a[1] for a,b in zip(result,result[1:])):raise ValueError('Intervals overlap')
        return result
    windows=intervals(labels.get('reviewed_windows',[]))
    if not windows:raise ValueError('No completely reviewed windows')
    events=labels.get('events',[])
    identifiers=set()
    for event in events:
        t=event.get('timestamp_seconds')
        if not isinstance(t,(int,float)) or isinstance(t,bool) or not math.isfinite(t) or not 0<=t<duration:
            raise ValueError('Invalid event time')
        if event.get('type') not in {'pass','turnover'}:raise ValueError('Unsupported event type')
        if not event.get('id') or event['id'] in identifiers:raise ValueError('Duplicate/missing event ID')
        identifiers.add(event['id'])
        if not any(a<=t<b for a,b in windows):raise ValueError('Events must be inside a completely reviewed window')
    states=labels.get('possession_intervals',[])
    intervals([[r['start_seconds'],r['end_seconds']] for r in states])
    if any(not isinstance(r.get('team'),str) or not r['team'].strip() for r in states):raise ValueError('Invalid possession state/team')
    return windows


def score_sequences(labels, insights, metadata, tolerance=.5, allow_visual_pilot=False):
    if not 0<tolerance<=2:raise ValueError('Invalid event tolerance')
    windows=validate_labels(labels,metadata,allow_visual_pilot)
    predictions=insights.get('sequences',{}).get('candidates',insights.get('candidates',[]))
    predictions=[p for p in predictions if any(a<=p['timestamp_seconds']<b for a,b in windows)]
    truth=labels['events']
    # Global closest pairs, one-to-one. Team attribution is reported separately;
    # it cannot create multiple event matches or erase timing false positives.
    pairs=sorted((abs(p['timestamp_seconds']-t['timestamp_seconds']),pi,ti)
        for pi,p in enumerate(predictions) for ti,t in enumerate(truth)
        if p['type']=='possible_'+t['type'] and abs(p['timestamp_seconds']-t['timestamp_seconds'])<=tolerance)
    edges={pi:[] for pi in range(len(predictions))}
    for error,pi,ti in pairs:edges[pi].append(ti)
    matched_truth={}
    def augment(pi,seen):
        for ti in edges[pi]:
            if ti in seen:continue
            seen.add(ti)
            if ti not in matched_truth or augment(matched_truth[ti],seen):
                matched_truth[ti]=pi
                return True
        return False
    for pi in sorted(edges,key=lambda pi:min([abs(predictions[pi]['timestamp_seconds']-truth[ti]['timestamp_seconds']) for ti in edges[pi]],default=math.inf)):
        augment(pi,set())
    matches=[dict(prediction_id=predictions[pi]['id'],truth_id=truth[ti]['id'],
                  timing_error_seconds=round(abs(predictions[pi]['timestamp_seconds']-truth[ti]['timestamp_seconds']),4)) for ti,pi in sorted(matched_truth.items())]
    tp,fp,fn=len(matches),len(predictions)-len(matches),len(truth)-len(matches)
    def ratio(a,b):return round(a/b,4) if b else None
    model=insights.get('sequences',{}).get('intervals',[])
    mapping=labels.get('kit_mappings',{})
    reference,assigned= {},{}
    correct=wrong=missed=reviewed_known=labelled_state_seconds=0.
    for state in labels.get('possession_intervals',[]):
        start,end=state['start_seconds'],state['end_seconds'];team=state['team']
        for wa,wb in windows:
            a,b=max(start,wa),min(end,wb)
            if b<=a:continue
            labelled_state_seconds+=b-a
            if team in {'unknown','contested','dead_ball'}:continue
            reviewed_known+=b-a;reference[team]=reference.get(team,0)+b-a
            covered=0.
            for row in model:
                overlap=max(0,min(b,row['end_seconds'])-max(a,row['start_seconds']))
                if not overlap:continue
                guess=mapping.get(row.get('team'),row.get('team'))
                if row['state'] in {'controlled','in_transit','control_gap'} and guess:
                    covered+=overlap;assigned[guess]=assigned.get(guess,0)+overlap
                    if guess==team:correct+=overlap
                    else:wrong+=overlap
            missed+=max(0,b-a-covered)
    teams=sorted(set(reference)|set(assigned))
    return dict(schema_version='sequence-score-1',annotation_source=labels['annotation_source'],reviewer=labels['reviewer'],
        independent_human_labels=labels['annotation_source']=='human',split=labels['split'],match_group=labels.get('match_group'),
        source_sha256=metadata['sha256'],reviewed_seconds=round(sum(b-a for a,b in windows),4),
        event_tolerance_seconds=tolerance,events=dict(true_positives=tp,false_positives=fp,false_negatives=fn,
            precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),f1=ratio(2*tp,2*tp+fp+fn),matches=matches),
        possession=dict(labelled_state_seconds=round(labelled_state_seconds,4),reviewed_known_team_seconds=round(reviewed_known,4),
            correct_team_seconds=round(correct,4),wrong_team_seconds=round(wrong,4),unclassified_seconds=round(missed,4),
            classified_coverage_pct=ratio(100*(correct+wrong),reviewed_known),accuracy_on_classified_pct=ratio(100*correct,correct+wrong),
            by_team=[dict(team=t,reference_seconds=round(reference.get(t,0),4),predicted_seconds=round(assigned.get(t,0),4),
                absolute_share_error_percentage_points=ratio(100*abs(reference.get(t,0)-assigned.get(t,0)),reviewed_known)) for t in teams]),
        limitations='Scores cover explicitly reviewed windows only. Unclassified time is retained; conditional accuracy alone can be misleading. Development/visual-pilot scores are not independent held-out accuracy evidence.')
