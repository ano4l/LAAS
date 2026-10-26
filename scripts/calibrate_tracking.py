"""Calibrate observed player movement only after landmark and camera stability checks."""
import argparse
import hashlib
import json
from itertools import groupby
from pathlib import Path
import sys
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workers.video.pitch_calibration import validate_calibration,FixedCameraCheck,calibrated_measurements

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('video',type=Path);p.add_argument('run',type=Path);p.add_argument('calibration',type=Path)
if __name__=='__main__':
    args=p.parse_args();manifest=json.loads((args.run/'manifest.json').read_text())
    if hashlib.sha256(args.video.read_bytes()).hexdigest()!=manifest['metadata']['sha256']:raise ValueError('Source video mismatch')
    calibration=validate_calibration(json.loads(args.calibration.read_text()),manifest['metadata'])
    rows=json.loads((args.run/'tracking.json').read_text());groups={i:list(g) for i,g in groupby(rows,key=lambda r:r['frame_idx'])}
    camera=FixedCameraCheck();cap=cv2.VideoCapture(str(args.video));index=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        camera.observe(frame,groups.get(index,[]));index+=1
    cap.release()
    if index!=manifest['metadata']['frame_count']:raise ValueError('Incomplete camera stability verification')
    result=calibrated_measurements(rows,manifest['metadata'],calibration,camera.result())
    (args.run/'calibrated-metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
