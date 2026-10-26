import cv2
import numpy as np
import pytest
from workers.video.pitch_calibration import validate_calibration,calibrated_measurements,FixedCameraCheck

META=dict(sha256='0'*64,width=1000,height=600,fps=10)


def config():
    image=[[100,100],[900,100],[900,500],[100,500],[500,100],[500,500]]
    pitch=[[0,0],[100,0],[100,60],[0,60],[50,0],[50,60]]
    return dict(source_sha256='0'*64,camera_mode='fixed',pitch_length_m=100,pitch_width_m=60,
                points=[dict(image_xy=a,pitch_xy_m=b) for a,b in zip(image,pitch)])


def test_heldout_calibration_and_observed_distance_have_real_units_and_gaps():
    calibration=validate_calibration(config(),META)
    rows=[dict(frame_idx=i,object_type='player',track_id='T1',bbox=[95+i*4,100,105+i*4,200]) for i in range(6)]
    result=calibrated_measurements(rows,META,calibration,dict(verified=True))
    track=result['tracks'][0]
    assert track['distance_observed_m']==2.5
    assert track['mean_observed_speed_m_s']==5
    assert track['observed_transition_seconds']==.5
    rejected=calibrated_measurements(rows,META,calibration,dict(verified=False,reason='camera moved'))
    assert rejected['available'] is False and rejected['tracks']==[]


def test_wrong_video_collinear_and_failed_holdout_rejected():
    bad=config();bad['source_sha256']='wrong'
    with pytest.raises(ValueError,match='hash'):validate_calibration(bad,META)
    bad=config();bad['points'][-1]['pitch_xy_m']=[50,50]
    with pytest.raises(ValueError,match='held-out'):validate_calibration(bad,META)
    bad=config()
    for i,p in enumerate(bad['points'][:4]):p['image_xy']=[50+200*i,100]
    with pytest.raises(ValueError,match='non-collinear'):validate_calibration(bad,META)


def test_camera_motion_and_insufficient_features_fail_closed():
    image=np.random.default_rng(1).integers(0,255,(200,300,3),dtype=np.uint8)
    camera=FixedCameraCheck();camera.observe(image,[]);camera.observe(image,[])
    assert camera.result()['verified']
    moved=cv2.warpAffine(image,np.float32([[1,0,12],[0,1,0]]),(300,200))
    camera.observe(moved,[])
    assert camera.result()['verified'] is False
    blank=FixedCameraCheck();blank.observe(np.zeros_like(image),[]);blank.observe(np.zeros_like(image),[])
    assert blank.result()['verified'] is False
