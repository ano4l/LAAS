"""Independent ball predictions followed by shared geometry and tracking refinement."""
from pathlib import Path
from workers.video.accuracy import tile_windows,ball_search_window
import math
from workers.video.observation_refiner import ObservationRefiner

BALL_CLASSES={'ball','sports ball'}
PERSON_CLASSES={'person','player','goalkeeper','referee'}


class FootballDetector:
    def __init__(self,weights:Path,*,device,image_size,ball_confidence,object_confidence,
                 sample_fps,ball_weights=None,tile_size=0,ball_rescue_weights=None,person_weights=None):
        from ultralytics import YOLO
        self.model=YOLO(str(weights));self.names=self.model.names
        self.ball_model=YOLO(str(ball_weights)) if ball_weights else None
        self.rescue_model=YOLO(str(ball_rescue_weights)) if ball_rescue_weights else None
        # A general person detector finds players the football weights miss or box in pieces on closer views
        # (docs/testing/DETECTOR_2026-11-22.md). The football weights still supply the ball and say who is a
        # goalkeeper or a referee. The same file as the secondary ball model is loaded once.
        same=bool(person_weights and ball_rescue_weights and Path(person_weights).resolve()==Path(ball_rescue_weights).resolve())
        self.person_model=(self.rescue_model if same else YOLO(str(person_weights))) if person_weights else None
        self.general_person_ids=[k for k,n in self.person_model.names.items() if str(n).lower()=='person'] if self.person_model else []
        if self.person_model and not self.general_person_ids:
            raise ValueError('The person detector has no person class')
        self.rescue_ids=[k for k,n in self.rescue_model.names.items() if str(n).lower() in BALL_CLASSES] if self.rescue_model else []
        if self.rescue_model and not self.rescue_ids:
            raise ValueError('Secondary ball model has no supported ball class')
        self.rescue_stride=max(1,math.ceil(sample_fps/2));self.frame_number=0
        self.sample_fps=sample_fps;self.roi_stride=max(1,math.ceil(sample_fps/8))
        self.device,self.image_size=device,image_size
        self.ball_confidence,self.object_confidence=ball_confidence,object_confidence
        self.tile_size=tile_size
        self.person_ids=[k for k,v in self.names.items() if str(v).lower() in PERSON_CLASSES]
        self.ball_ids=[k for k,v in self.names.items() if str(v).lower() in BALL_CLASSES]
        if not self.person_ids and not self.ball_ids:
            raise ValueError('Detector has no supported person/player/ball classes.')
        if self.ball_model and not any(str(n).lower() in BALL_CLASSES for n in self.ball_model.names.values()):
            raise ValueError('The configured ball model has no ball or sports ball class.')
        self.refiner=ObservationRefiner(self.names,sample_fps,object_confidence)
        self.quality_counts=self.refiner.quality_counts
        self.tracking_config=self.refiner.tracking_config
        self.quality_counts.update(ball_rescue_frames=0,ball_rescue_candidates=0,ball_rescue_roi_frames=0)
        if self.person_model:self.quality_counts.update(general_people=0,roles_from_football_model=0)
        self.default_role=next((role for role in ('player','person') if role in {str(v).lower() for v in self.names.values()}),None)
        if self.person_model and not self.default_role:
            raise ValueError('The football detector needs a player or person class to label people from the person detector')

    def predict(self,model,frame,classes,confidence):
        return model.predict(frame,classes=classes,imgsz=self.image_size,conf=confidence,
                             iou=.45,device=self.device,verbose=False)[0]

    def detect(self,frame):
        self.frame_number+=1
        classes=self.person_ids if self.ball_model else self.person_ids+self.ball_ids
        result=self.predict(self.model,frame,classes,min(.1,self.ball_confidence,self.object_confidence))
        rows=[self._row(box,result.names,None,'object_detector') for box in result.boxes
              if int(box.cls.item()) in self.person_ids]
        if self.person_model:
            from workers.video.people_boxes import adopt_roles
            general=self.predict(self.person_model,frame,self.general_person_ids,min(.1,self.object_confidence))
            people=adopt_roles([self._row(box,general.names,None,'person_detector') for box in general.boxes],rows,self.default_role)
            self.quality_counts['general_people']+=len(people)
            self.quality_counts['roles_from_football_model']+=sum(row['object_type']!=self.default_role for row in people)
            rows=people
        if self.ball_model:
            ids=[k for k,n in self.ball_model.names.items() if str(n).lower() in BALL_CLASSES]
            result=self.predict(self.ball_model,frame,ids,self.ball_confidence)
        for box in result.boxes:
            if str(result.names[int(box.cls.item())]).lower() in BALL_CLASSES and float(box.conf.item())>=self.ball_confidence:
                rows.append(self._row(box,result.names,None,'ball_detector' if self.ball_model else 'object_detector'))
        model=self.ball_model or self.model
        ids=[k for k,n in model.names.items() if str(n).lower() in BALL_CLASSES]
        if self.tile_size and ids:
            height,width=frame.shape[:2]
            for x1,y1,x2,y2 in tile_windows(width,height,self.tile_size):
                if (x1,y1,x2,y2)==(0,0,width,height):continue
                crop=self.predict(model,frame[y1:y2,x1:x2],ids,self.ball_confidence)
                for box in crop.boxes:
                    if str(crop.names[int(box.cls.item())]).lower() not in BALL_CLASSES or float(box.conf.item())<self.ball_confidence:continue
                    row=self._row(box,crop.names,None,'ball_tile_detector');a,b,c,d=row['bbox']
                    row['bbox']=[a+x1,b+y1,c+x1,d+y1];rows.append(row)
        if self.rescue_model and not any(r['object_type'] in BALL_CLASSES and r['confidence']>=.3 for r in rows):
            window=ball_search_window(frame.shape[1],frame.shape[0],self.refiner.ball_continuity.last,
                                      self.refiner.ball_continuity.velocity,self.frame_number,self.sample_fps)
            use_global=(self.frame_number-1)%self.rescue_stride==0
            use_roi=window is not None and (self.frame_number-1)%self.roi_stride==0 and not use_global
            if not use_roi and not use_global:return self.refiner.refine(frame,rows)
            x1,y1,x2,y2=window if use_roi else (0,0,frame.shape[1],frame.shape[0])
            rescue=self.rescue_model.predict(frame[y1:y2,x1:x2],classes=self.rescue_ids,imgsz=640,conf=.3,iou=.45,device=self.device,verbose=False)[0]
            self.quality_counts['ball_rescue_frames']+=1
            self.quality_counts['ball_rescue_roi_frames']+=int(use_roi)
            for box in rescue.boxes:
                row=self._row(box,rescue.names,None,'secondary_ball_roi_detector' if use_roi else 'secondary_ball_detector')
                if row['object_type'] in BALL_CLASSES and row['confidence']>=.3:
                    a,b,c,d=row['bbox'];row['bbox']=[a+x1,b+y1,c+x1,d+y1]
                    row['search_region_px']=[x1,y1,x2,y2]
                    rows.append(row);self.quality_counts['ball_rescue_candidates']+=1
        return self.refiner.refine(frame,rows)

    @staticmethod
    def _row(box,names,identity,role):
        return dict(object_type=str(names[int(box.cls.item())]).lower(),track_id=identity,
                    bbox=[round(float(v),2) for v in box.xyxy[0].tolist()],confidence=round(float(box.conf.item()),4),
                    source='model_detection',model_role=role,interpolated=False)
