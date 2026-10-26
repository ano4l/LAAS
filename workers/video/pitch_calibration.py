"""Explicit fixed-camera calibration with held-out landmarks; never assumed pitch dimensions."""
import cv2
import numpy as np


def validate_calibration(config,metadata):
    if not isinstance(metadata.get('sha256'),str) or len(metadata['sha256'])!=64:raise ValueError('A source video SHA256 is required')
    if config.get('source_sha256')!=metadata.get('sha256'):raise ValueError('Calibration video hash mismatch')
    if config.get('camera_mode')!='fixed':raise ValueError('This calibration path requires a fixed camera')
    length,width=config.get('pitch_length_m'),config.get('pitch_width_m')
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not np.isfinite(v) for v in [length,width]):
        raise ValueError('Measured pitch dimensions are required')
    if not 20<=length<=140 or not 10<=width<=100 or length<=width:raise ValueError('Invalid pitch dimensions')
    points=config.get('points',[])
    if not isinstance(points,list) or not 6<=len(points)<=100:raise ValueError('At least six landmarks are required; final two are held out')
    try:
        image=np.array([p['image_xy'] for p in points],dtype=float)
        pitch=np.array([p['pitch_xy_m'] for p in points],dtype=float)
    except (KeyError,TypeError,ValueError) as exc:raise ValueError('Invalid landmark coordinates') from exc
    if image.shape!=(len(points),2) or pitch.shape!=image.shape or not np.isfinite(image).all() or not np.isfinite(pitch).all():
        raise ValueError('Landmarks must be finite coordinate pairs')
    if not ((image>=0).all() and (image[:,0]<metadata['width']).all() and (image[:,1]<metadata['height']).all()
            and (pitch>=0).all() and (pitch[:,0]<=length).all() and (pitch[:,1]<=width).all()):
        raise ValueError('Landmarks lie outside the source image or pitch')
    if len(np.unique(image,axis=0))!=len(points) or len(np.unique(pitch,axis=0))!=len(points):raise ValueError('Duplicate landmarks')
    if cv2.contourArea(cv2.convexHull(image[:-2].astype(np.float32)))<.02*metadata['width']*metadata['height']:
        raise ValueError('Fitting landmarks must span a useful non-collinear image area')
    matrix,_=cv2.findHomography(image[:-2],pitch[:-2],0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:raise ValueError('Degenerate calibration')
    projected=cv2.perspectiveTransform(image.reshape(-1,1,2),matrix).reshape(-1,2)
    inverse=np.linalg.inv(matrix)
    reprojected=cv2.perspectiveTransform(pitch.reshape(-1,1,2),inverse).reshape(-1,2)
    pixel_error=np.linalg.norm(image-reprojected,axis=1)
    world_error=np.linalg.norm(projected-pitch,axis=1)
    if pixel_error.max()>3 or world_error.max()>.75:raise ValueError('Calibration failed reprojection or held-out landmark checks')
    return dict(matrix=matrix,pitch_length_m=length,pitch_width_m=width,
                held_out_error_m=round(float(world_error[-2:].max()),4),max_reprojection_error_px=round(float(pixel_error.max()),4),
                provenance='user_supplied_fixed_camera_landmarks_and_dimensions')


class FixedCameraCheck:
    def __init__(self):self.reference=None;self.points=None;self.scale=1;self.reason=None;self.checked_frames=0;self.max_drift=0.

    def observe(self,frame,rows):
        height,width=frame.shape[:2];self.scale=min(1,640/width)
        gray=cv2.cvtColor(cv2.resize(frame,(round(width*self.scale),round(height*self.scale))),cv2.COLOR_BGR2GRAY)
        if self.reference is None:
            self.reference=gray
            mask=np.full(gray.shape,255,np.uint8)
            for r in rows:
                a,b,c,d=[round(v*self.scale) for v in r['bbox']]
                cv2.rectangle(mask,(max(0,a-5),max(0,b-5)),(min(mask.shape[1]-1,c+5),min(mask.shape[0]-1,d+5)),0,-1)
            self.points=cv2.goodFeaturesToTrack(gray,200,.02,8,mask=mask)
            if self.points is None or len(self.points)<12:self.reason='Insufficient background features to verify a fixed camera'
            return
        if self.reason:return
        found,status,_=cv2.calcOpticalFlowPyrLK(self.reference,gray,self.points,None)
        if found is None or status is None:self.reason='Camera stability could not be verified';return
        good=status.ravel().astype(bool)
        if good.sum()<12:self.reason='Insufficient camera-motion correspondences';return
        transform,inliers=cv2.estimateAffinePartial2D(self.points[good],found[good],method=cv2.RANSAC,ransacReprojThreshold=1.5)
        if transform is None or inliers.sum()<12 or inliers.mean()<.6:self.reason='Camera-motion estimate is unreliable';return
        h,w=gray.shape;corners=np.array([[0,0],[w,0],[w,h],[0,h]],dtype=float)
        drift=np.linalg.norm(corners@transform[:,:2].T+transform[:,2]-corners,axis=1).max()/self.scale
        self.max_drift=max(self.max_drift,float(drift));self.checked_frames+=1
        if drift>max(2,.001*np.hypot(width,height)):self.reason='Camera moved beyond the fixed-calibration tolerance'

    def result(self):
        return dict(verified=self.reason is None and self.checked_frames>0,reason=self.reason or ('No camera transitions verified' if not self.checked_frames else None),
                    checked_frames=self.checked_frames,max_background_drift_px=round(self.max_drift,3))


def calibrated_measurements(rows,metadata,calibration,camera_check):
    if not camera_check.get('verified'):return dict(available=False,reason=camera_check.get('reason'),tracks=[],camera_check=camera_check)
    matrix=calibration['matrix'];tracks={}
    for row in rows:
        if row['object_type'] not in {'player','goalkeeper','person'} or not row.get('track_id'):continue
        a,b,c,d=row['bbox'];point=np.array([[[(a+c)/2,d]]],dtype=float)
        x,y=cv2.perspectiveTransform(point,matrix)[0,0]
        if not -1e-6<=x<=calibration['pitch_length_m']+1e-6 or not -1e-6<=y<=calibration['pitch_width_m']+1e-6:continue
        x=np.clip(x,0,calibration['pitch_length_m']);y=np.clip(y,0,calibration['pitch_width_m'])
        t=row['frame_idx']/metadata['fps']
        track=tracks.setdefault(row['track_id'],dict(track_id=row['track_id'],distance_observed_m=0.,observed_transition_seconds=0.,excluded_steps=0,last=None))
        if track['last']:
            old_t,old_x,old_y=track['last'];dt=t-old_t;distance=float(np.hypot(x-old_x,y-old_y))
            if 0<dt<=.2 and distance/dt<=15:
                track['distance_observed_m']+=distance;track['observed_transition_seconds']+=dt
            else:track['excluded_steps']+=1
        track['last']=(t,float(x),float(y))
    output=[]
    for track in tracks.values():
        track.pop('last');dt=track['observed_transition_seconds']
        track['mean_observed_speed_m_s']=round(track['distance_observed_m']/dt,3) if dt else None
        track['distance_observed_m']=round(track['distance_observed_m'],3) if dt else None
        track['observed_transition_seconds']=round(dt,3);output.append(track)
    return dict(available=True,tracks=output,camera_check=camera_check,
                calibration={k:v for k,v in calibration.items() if k!='matrix'},
                limitations='Temporary identities and foot-point predictions remain unreviewed. Distances cover accepted observed transitions only; gaps >0.2s and steps >15m/s are excluded. No ball-ground projection or complete workload is inferred.')
