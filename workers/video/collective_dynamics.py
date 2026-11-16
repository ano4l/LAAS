"""Observed collective shape, adapted from Chao Zhuang's SoccerTrack notebook.

Sample-covariance ellipses describe visible camera-image points, not tactical
formation labels, pitch territory, uncertainty regions or complete teams.
"""
from collections import defaultdict, deque
import math

import numpy as np

VERSION='soccertrack-dynamics-1'
REFERENCE=dict(title='SoccerTrack: Collective Dynamics Analysis',version=21,
    url='https://www.kaggle.com/code/chaozhuang/soccertrack-collective-dynamics-analysis')


def pitch_status(pitch):
    if pitch is None:
        return dict(status='unavailable',reason='Pitch calibration was not run for this analysis.')
    from workers.video import pitch_keypoints as pk
    calibrated=sum(1 for record in pitch.values() if record.get('status')=='calibrated')
    stabilised=sum(1 for record in pitch.values() if record.get('stabilised'))
    return dict(status='estimated' if calibrated else 'unavailable',reason=None if calibrated else 'No analysed frame passed the pitch calibration checks.',
        review_status='unreviewed',accuracy_evaluated=False,method='landmark_homography_smoothed' if stabilised else 'per_frame_landmark_homography',
        assumed_pitch=dict(length_m=pk.LENGTH_M,width_m=pk.WIDTH_M,measured=False),outside_pitch_margin_m=pk.MARGIN_M,
        frames_tried=len(pitch),frames_calibrated=calibrated,frames_stabilised=stabilised,limitations=pk.LIMITATIONS,
        line_check=pk.lines_summary(pitch))


def ellipse(points,width,height):
    xy=np.asarray(points,dtype=float)
    if len(xy)<3 or xy.shape!=(len(xy),2) or not np.isfinite(xy).all():
        return None
    centre=xy.mean(axis=0)
    values,vectors=np.linalg.eigh(np.cov(xy.T,ddof=1))
    values=np.maximum(values,0);major,minor=np.sqrt(values[1]),np.sqrt(values[0])
    angle=math.degrees(math.atan2(vectors[1,1],vectors[0,1])) % 180 if major>1e-9 else 0.
    return dict(centroid_x_pct=round(100*centre[0]/width,4),centroid_y_pct=round(100*centre[1]/height,4),
        major_radius_px=round(float(major),4),minor_radius_px=round(float(minor),4),angle_degrees=round(angle,4),
        area_image_pct=round(100*math.pi*float(major*minor)/(width*height),6),
        rms_spread_px=round(float(np.sqrt(np.mean(np.sum((xy-centre)**2,axis=1)))),4),point_count=len(xy))


def pitch_snapshot(record,players,width,height):
    """Estimated pitch positions for one calibrated frame. Players that do not land on the pitch are counted, not placed."""
    import cv2
    from workers.video import pitch_keypoints as pk
    xy,usable=pk.to_pitch(record,[(p['x'],p['y']) for p in players])
    placed=[dict(track_id=p['track'],x_m=round(float(x),2),y_m=round(float(y),2)) for p,(x,y),ok in zip(players,xy,usable) if ok]
    out=dict(players=placed,players_not_placed=int((~usable).sum()),landmarks_used=record['landmarks_used'],fit_error_m=record.get('stable_error_m',record['mean_error_m']),stabilised=bool(record.get('stabilised')),
        visible_area_m=pk.visible_area(record,width,height),length_extent_m=None,width_extent_m=None,hull_area_m2=None)
    if len(placed)>=3:
        points=np.array([(p['x_m'],p['y_m']) for p in placed],dtype=np.float32)
        out.update(length_extent_m=round(float(np.ptp(points[:,0])),1),width_extent_m=round(float(np.ptp(points[:,1])),1),
            hull_area_m2=round(float(cv2.contourArea(cv2.convexHull(points))),0))
    return out


def collective_dynamics(frames,metadata,sample_fps,pitch=None):
    width,height=metadata['width'],metadata['height'];duration=metadata['duration_seconds']
    diagonal=math.hypot(width,height);step=1/sample_fps
    windows={};activity=defaultdict(list);history={};supported=defaultdict(float)
    for frame in frames:
        t,dt=frame['time'],frame['duration']
        calibration=(pitch or {}).get(frame.get('frame'))
        if not calibration or calibration.get('status')!='calibrated':calibration=None
        # Duplicate identities are ambiguous, rather than extra team members.
        counts=defaultdict(int)
        for p in frame['players']:
            counts[p['scene'],p['track']]+=1
        groups=defaultdict(list);directions=defaultdict(list)
        present=set()
        for p in frame['players']:
            if (not p['kit'] or not p['track'] or counts[p['scene'],p['track']]!=1 or not math.isfinite(p['confidence']) or p['confidence']<.45
                    or not all(math.isfinite(p[k]) for k in ['x','y','height'])
                    or not 0<=p['x']<=width or not 0<=p['y']<=height):continue
            key=(p['scene'],p['track'],p['kit']);present.add(key);group=(p['scene'],p['kit'])
            groups[group].append(p);activity[key].append((t,dt,p['x'],p['y']))
            trail=history.setdefault(key,deque())
            if trail and t-trail[-1][0]>1.5*step+1e-8:trail.clear()
            while trail and t-trail[0][0]>.65:trail.popleft()
            prior=next((old for old in trail if .4-1e-8<=t-old[0]<=.6+1e-8),None)
            if prior:
                dx,dy=p['x']-prior[1],p['y']-prior[2];length=math.hypot(dx,dy)
                if max(3.,.08*p['height'])<=length<=diagonal*.35*(t-prior[0]):
                    directions[group].append((dx/length,dy/length))
            trail.append((t,p['x'],p['y']))
        # No motion state may bridge missing observations, a scene or kit switch.
        for key in set(history)-present:history.pop(key)
        for group,players in groups.items():
            if len(players)<3:continue
            shape=ellipse([(p['x'],p['y']) for p in players],width,height)
            if shape is None:continue
            supported[group]+=dt
            movers=directions[group]
            alignment=math.hypot(sum(x for x,y in movers),sum(y for x,y in movers))/len(movers) if len(movers)>=3 else None
            for bucket in range(int(t+1e-8),math.ceil(min(duration,t+dt)-1e-8)):
                weight=max(0,min(duration,t+dt,bucket+1)-max(t,bucket))
                if weight<=1e-9:continue
                key=(*group,bucket)
                record=windows.setdefault(key,dict(start_seconds=float(bucket),end_seconds=min(duration,bucket+1.),
                    supported_seconds=0.,observed_frames=0,players_sum=0.,area_sum=0.,x_sum=0.,y_sum=0.,
                    alignment_sum=0.,alignment_supported_seconds=0.,snapshot=None,snapshot_rank=(True,math.inf)))
                record['supported_seconds']+=weight;record['observed_frames']+=1
                record['players_sum']+=len(players)*weight;record['area_sum']+=shape['area_image_pct']*weight
                record['x_sum']+=shape['centroid_x_pct']*weight;record['y_sum']+=shape['centroid_y_pct']*weight
                if alignment is not None:
                    record['alignment_sum']+=alignment*weight;record['alignment_supported_seconds']+=weight
                # The frame nearest the middle of the second; a frame with a pitch calibration is preferred over one without.
                rank=(calibration is None,abs(t-(record['start_seconds']+record['end_seconds'])/2))
                if rank<record['snapshot_rank']:
                    record['snapshot_rank']=rank
                    record['snapshot']=dict(timestamp_seconds=round(t,4),ellipse=shape,
                        players=[dict(track_id=p['track'],x_pct=round(100*p['x']/width,4),y_pct=round(100*p['y']/height,4)) for p in players])
                    if calibration:record['snapshot']['pitch']=pitch_snapshot(calibration,players,width,height)
    from workers.video.pitch_motion import pitch_motion
    teams=[]
    for scene,kit in sorted(supported,key=lambda g:(g[0],g[1])):
        records=[];total_area=total_players=alignment_sum=alignment_seconds=0.
        for (s,k,bucket),record in sorted(windows.items()):
            if (s,k)!=(scene,kit):continue
            dt=record['supported_seconds'];motion_dt=record['alignment_supported_seconds']
            total_area+=record['area_sum'];total_players+=record['players_sum']
            alignment_sum+=record['alignment_sum'];alignment_seconds+=motion_dt
            records.append(dict(start_seconds=record['start_seconds'],end_seconds=record['end_seconds'],
                supported_seconds=round(dt,4),observed_frames=record['observed_frames'],
                mean_visible_players=round(record['players_sum']/dt,3),mean_area_image_pct=round(record['area_sum']/dt,6),
                centroid_x_pct=round(record['x_sum']/dt,4),centroid_y_pct=round(record['y_sum']/dt,4),
                motion_alignment=round(record['alignment_sum']/motion_dt,4) if motion_dt else None,
                alignment_supported_seconds=round(motion_dt,4),snapshot=record['snapshot']))
        ellipses=[]
        for (s,track,k),points in activity.items():
            if (s,k)!=(scene,kit) or len(points)<8 or sum(p[1] for p in points)<.5:continue
            ellipses.append(dict(track_id=track,observed_seconds=round(sum(p[1] for p in points),4),observed_frames=len(points),
                ellipse=ellipse([(p[2],p[3]) for p in points],width,height)))
        ellipses.sort(key=lambda p:(-p['observed_seconds'],p['track_id']))
        seconds=supported[scene,kit]
        teams.append(dict(kit=kit,scene_id=scene,supported_seconds=round(seconds,4),source_coverage_pct=round(100*seconds/duration,2) if duration else None,
            mean_visible_players=round(total_players/seconds,3),mean_area_image_pct=round(total_area/seconds,6),
            mean_alignment=round(alignment_sum/alignment_seconds,4) if alignment_seconds else None,
            timeline=records,activity_ellipses=ellipses[:80],activity_ellipses_total=len(ellipses)))
    return dict(schema_version=VERSION,reference=REFERENCE,coordinate_system='camera_image',review_status='unreviewed',
        source_duration_seconds=duration,
        formation=dict(status='unavailable',label=None,
            reason='Named tactical formation needs calibrated pitch coordinates, full outfield coverage, goalkeeper/role separation and reliable identities. This notebook supplies collective-shape methods, not a formation classifier.'),
        pitch=pitch_status(pitch),pitch_motion=pitch_motion(frames,metadata,sample_fps,pitch),teams=teams,parameters=dict(min_visible_tracked_players=3,min_detection_confidence=.45,window_seconds=1,
            covariance='sample_ddof_1',ellipse_standard_deviations=1,motion_lag_seconds=[.4,.6],min_alignment_movers=3,
            min_activity_observations=8,min_activity_observed_seconds=.5,max_activity_ellipses_per_group=80),
        limitations=['Visible scene-scoped kit subsets, not complete identified teams; goalkeeper observations are included when grouped.',
            'Camera panning, zoom, perspective and temporary identity errors affect shape, activity and alignment. Image measures are pixels / percent of image area. Pitch metres, where a snapshot has them, are separate estimates from automatic calibration.',
            'One-standard-deviation ellipses describe spatial dispersion; they are not confidence intervals, occupied territory or named tactical formations.',
            'All points are observed foot-point detections. Missing frames/players are not interpolated; snapshots are real frames, not window-average formations.',
            'Motion alignment is a camera-image diagnostic among supported movers, not tactical coordination or possession. Stationary/uncertain movement remains unclassified.'])
