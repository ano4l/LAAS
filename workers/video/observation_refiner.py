"""Shared refinement for fresh predictions and explicitly replayed detection files."""
import cv2
import numpy as np
import os
from workers.video.accuracy import merge_ball_candidates
from workers.video.vision_quality import field_region, on_field, person_indices, BallContinuity
from workers.video.jersey_teams import JerseyTeams

HUMANS={'person','player','goalkeeper','referee'}
BALLS={'ball','sports ball'}


class ObservationRefiner:
    def __init__(self,names,sample_fps,object_confidence):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace,YAML
        from ultralytics.utils.checks import check_yaml
        self.names,self.object_confidence=names,object_confidence
        self.backend=os.getenv('TRACKER_BACKEND','botsort')
        if self.backend not in {'bytetrack','botsort'}:
            raise ValueError('TRACKER_BACKEND must be bytetrack or botsort')
        config=YAML.load(check_yaml(self.backend+'.yaml'))
        config['track_buffer']=max(1,round(sample_fps))
        config.update(track_high_thresh=max(.25,object_confidence),track_low_thresh=.1,
                      new_track_thresh=max(.5,object_confidence))
        if self.backend=='botsort':config.update(with_reid=False,gmc_method='sparseOptFlow')
        # Expanded-box matching keeps small, fast players on one track (see expanded_iou.py).
        expansion=float(os.getenv('TRACK_IOU_EXPANSION','0.5'))
        if not 0<=expansion<=1:
            raise ValueError('TRACK_IOU_EXPANSION must be between 0 and 1')
        base=BOTSORT if self.backend=='botsort' else BYTETracker
        if expansion:
            from workers.video.expanded_iou import expanded_tracker
            base=expanded_tracker(base,expansion)
        self.tracker=base(IterableSimpleNamespace(**config))
        self.tracking_config=dict(config,iou_expansion=expansion)
        self.ball_continuity=BallContinuity(sample_fps)
        self.jersey_teams=JerseyTeams()
        self.frame_number,self.scene_id,self.previous_scene=0,0,None
        self.quality_counts=dict(raw_people=0,retained_people=0,raw_balls=0,retained_balls=0,scene_cuts=0,field_mask_frames=0,low_confidence_recoveries=0)

    def refine(self,frame,observations):
        import torch
        from ultralytics.engine.results import Boxes
        scene=cv2.resize(frame,(64,36))
        if self.previous_scene is not None and np.mean(np.abs(scene.astype(float)-self.previous_scene.astype(float)))>65:
            self.tracker.reset();self.scene_id+=1
            self.ball_continuity.reset();self.jersey_teams=JerseyTeams()
            self.quality_counts['scene_cuts']+=1
        self.previous_scene=scene;self.frame_number+=1
        region=field_region(frame)
        self.quality_counts['field_mask_frames']+=int(region is not None)
        people=[dict(row) for row in observations if row['object_type'] in HUMANS]
        balls=[dict(row) for row in observations if row['object_type'] in BALLS]
        for row in balls:
            row['track_id']=None
        reverse={str(name).lower():int(key) for key,name in self.names.items()}
        data=[r['bbox']+[r['confidence'],reverse[r['object_type']]] for r in people]
        boxes=Boxes(torch.tensor(data,dtype=torch.float32).reshape(-1,6),frame.shape[:2])
        indices=person_indices(boxes,list(reverse.values()),self.names,.1,region,frame.shape[0])
        people=[people[index] for index in indices]
        filtered=boxes[indices]
        tracked=self.tracker.update(filtered.cpu().numpy(),frame)
        prefix=f'S{self.scene_id}:' if self.scene_id else ''
        identities={int(track[-1]):f'{prefix}T{int(track[4])}' for track in tracked}
        emitted=[]
        for index,row in enumerate(people):
            row.update(track_id=identities.get(index))
            low=row['confidence']<self.object_confidence
            if low and not row['track_id']:
                continue
            row['tracking_association']='low_confidence_recovery' if low else 'confirmed' if row['track_id'] else 'unconfirmed'
            self.quality_counts['low_confidence_recoveries']+=int(low)
            for name in ['team','team_assignment_source','team_assignment_confidence','team_review_status']:
                row.pop(name,None)
            emitted.append(row)
        people=emitted
        self.quality_counts['raw_people']+=len(data)
        self.quality_counts['retained_people']+=len(people)
        self.quality_counts['raw_balls']+=len(balls)
        # Aerial balls need not lie on the visible ground plane. Preserve that
        # uncertainty as a cue instead of rejecting them with the player mask.
        for row in balls:
            row['field_plausibility']='ground_region' if on_field(row['bbox'],region,frame.shape[0],ball=True) else 'outside_ground_region'
        balls=merge_ball_candidates(balls,iou_threshold=.45)
        self.quality_counts['retained_balls']+=len(balls)
        self.ball_continuity.select(balls,frame.shape[:2],self.frame_number)
        for row in people+balls:
            row['scene_id']=self.scene_id
        self.jersey_teams.annotate(frame,people)
        return people+balls
