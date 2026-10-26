"""Re-track existing real detections with source-hash verification; no neural inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from itertools import groupby
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workers.video import real_analysis
from workers.video.observation_refiner import ObservationRefiner
from scripts.export_video_analytics import export


def refine(source,prior_run,output,ball_rescue_weights=None):
    manifest=json.loads((prior_run/'manifest.json').read_text(encoding='utf-8'))
    with source.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['metadata']['sha256']:
            raise ValueError('Source video hash does not match the original model run')
    if manifest.get('engine')!='yolo_bytetrack' or manifest['detection_summary']['analyzed_frames']!=manifest['metadata']['frame_count']:
        raise ValueError('Refinement requires a complete real per-frame model detection run')
    rows=json.loads((prior_run/'tracking.json').read_text(encoding='utf-8'))
    if any(r.get('source')!='model_detection' for r in rows):
        raise ValueError('Only real model observations may be refined')
    grouped={index:list(group) for index,group in groupby(rows,key=lambda r:r['frame_idx'])}
    names={int(k):v for k,v in manifest['detection_summary']['model_classes'].items()}
    rescue_model=None
    rescue_stats=dict(frames=0,candidates=0)
    rescue_sha=None
    if ball_rescue_weights:
        from ultralytics import YOLO
        rescue_model=YOLO(str(ball_rescue_weights))
        rescue_ids=[k for k,n in rescue_model.names.items() if str(n).lower() in {'ball','sports ball'}]
        if not rescue_ids:raise ValueError('Secondary model has no ball class')
        rescue_sha=hashlib.sha256(ball_rescue_weights.read_bytes()).hexdigest()
    class ReplayDetector:
        def __init__(self,*args,**kwargs):
            self.names=names;self.index=0
            self.refiner=ObservationRefiner(names,kwargs['sample_fps'],kwargs['object_confidence'])
            self.quality_counts=self.refiner.quality_counts
            self.tracking_config=self.refiner.tracking_config
        def detect(self,frame):
            observations=[{k:r[k] for k in ['object_type','bbox','confidence','source','model_role','interpolated','secondary_ball_model_sha256','source_detection_run_id'] if k in r}
                          for r in grouped.get(self.index,[])]
            for row in observations:row.setdefault('source_detection_run_id',manifest['run_id'])
            stride=max(1,round(manifest['metadata']['fps']/2))
            if rescue_model and self.index%stride==0 and not any(r['object_type'] in {'ball','sports ball'} and r['confidence']>=.3 for r in observations):
                prediction=rescue_model.predict(frame,classes=rescue_ids,imgsz=640,conf=.3,iou=.45,device='cpu',verbose=False)[0]
                rescue_stats['frames']+=1
                for box in prediction.boxes:
                    if str(prediction.names[int(box.cls.item())]).lower() in {'ball','sports ball'} and float(box.conf.item())>=.3:
                        observations.append(dict(object_type='ball',bbox=[round(float(v),2) for v in box.xyxy[0].tolist()],
                            confidence=round(float(box.conf.item()),4),source='model_detection',model_role='secondary_ball_detector',
                            interpolated=False,secondary_ball_model_sha256=rescue_sha))
                        rescue_stats['candidates']+=1
            self.index+=1
            return self.refiner.refine(frame,observations)
    settings=dict(YOLO_DEVICE='cpu',ANALYSIS_PROFILE='accurate',ANALYSIS_FPS=str(manifest['metadata']['fps']),
                  YOLO_IMAGE_SIZE='1280',OBJECT_CONFIDENCE='.45',BALL_TILE_SIZE='0',YOLO_CPU_THREADS='2')
    with patch.dict(os.environ,settings), patch.object(real_analysis,'FootballDetector',ReplayDetector):
        result=real_analysis.run_real_analysis(source,output)
    result['detection_summary'].update(inference_source='existing_model_observations_reprocessed',
        source_detection_run_id=manifest['run_id'],source_model_sha256=manifest['detection_summary']['model_sha256'],
        model_sha256=manifest['detection_summary']['model_sha256'],model=manifest['detection_summary']['model'],
        source_ultralytics_version=manifest['detection_summary'].get('ultralytics_version'))
    result['warnings'].insert(0,'Existing real model detections were filtered and re-tracked. This run did not perform new neural inference or recover previously undetected balls.')
    if rescue_model:
        result['warnings'][0]='Primary detections were reprocessed; new secondary-model ball inference was run at up to 2fps on weak/missing-ball frames. No missing positions were interpolated.'
        result['detection_summary'].update(inference_source='reprocessed_primary_detections_with_sampled_secondary_ball_inference',
            secondary_ball_inference=dict(model=ball_rescue_weights.name,model_sha256=rescue_sha,image_size=640,confidence=.3,**rescue_stats))
    directory=output/result['run_id']
    (directory/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    export(directory)
    return result,directory


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('prior_run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ball-rescue-model',type=Path)
    args=p.parse_args();result,directory=refine(args.source,args.prior_run,args.output,args.ball_rescue_model)
    print(json.dumps(dict(run_id=result['run_id'],directory=str(directory),metrics=result['metrics']['video_measurements']),indent=2))
