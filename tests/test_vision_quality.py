import cv2
import numpy as np
from workers.video.vision_quality import field_region, on_field, person_indices, BallContinuity
from workers.video.jersey_teams import JerseyTeams
from workers.video.measurements import VideoMeasurements


def test_field_filter_is_bounded_and_unreliable_masks_fail_open():
    frame=np.zeros((400,640,3),np.uint8)
    frame[140:380,20:620]=(40,150,40)
    region=field_region(frame)
    assert region is not None
    assert on_field([200,150,220,250],region,400)
    assert not on_field([200,10,220,50],region,400)
    assert field_region(np.zeros_like(frame)) is None
    assert on_field([0,0,5,5],None,400)


def test_overlapping_human_classes_are_filtered_before_tracking():
    import torch
    from ultralytics.engine.results import Boxes
    boxes=Boxes(torch.tensor([[10,10,30,60,.9,2],[11,11,31,61,.8,3],
                              [40,10,60,60,.8,2],[80,10,100,60,.2,2]]),(100,100))
    assert person_indices(boxes,[2,3],{2:'player',3:'referee'},.35,None,100)==[0,2]
    fragments=Boxes(torch.tensor([[10,10,30,30,.95,2],[10,10,30,60,.8,2]]),(100,100))
    assert person_indices(fragments,[2],{2:'player'},.35,None,100)==[1]


def test_ball_selection_rejects_teleports_and_ambiguity_without_interpolation():
    def ball(x,confidence=.8):return dict(bbox=[x,100,x+4,104],confidence=confidence)
    selector=BallContinuity(25)
    first=[ball(100)];selector.select(first,(600,1000),1)
    assert first[0]['ball_selected']
    next_frame=[ball(105),ball(700,.95)]
    selector.select(next_frame,(600,1000),2)
    assert [r['ball_selected'] for r in next_frame]==[True,False]
    selector.select([],(600,1000),3)
    selector.reset()
    ambiguous=[ball(100,.6),ball(500,.6)]
    selector.select(ambiguous,(600,1000),4)
    assert not any(r['ball_selected'] for r in ambiguous)


def test_jersey_groups_require_support_and_keep_unverified_provenance():
    model=JerseyTeams()
    frame=np.zeros((200,300,3),np.uint8)
    frame[20:180,20:80]=(0,0,200)
    frame[20:180,150:210]=(210,210,210)
    rows=[dict(object_type='player',track_id='T1',bbox=[20,20,80,180]),
          dict(object_type='player',track_id='T2',bbox=[150,20,210,180])]
    rows+= [dict(row,track_id=row['track_id']+'b') for row in rows]
    for _ in range(25):model.annotate(frame,rows)
    assert rows[0]['team']!=rows[1]['team']
    assert all(r['team_review_status']=='unreviewed' for r in rows)
    assert model.feature(frame,[0,0,0,0]) is None


def test_low_confidence_boxes_recover_existing_tracks_but_cannot_start_new_ids(monkeypatch):
    from workers.video.observation_refiner import ObservationRefiner
    monkeypatch.setenv('TRACKER_BACKEND','bytetrack')
    ref=ObservationRefiner({0:'player'},25,.45)
    frame=np.zeros((100,100,3),np.uint8)
    def person(conf):return dict(object_type='player',bbox=[10,10,30,70],confidence=conf,source='model_detection')
    high=ref.refine(frame,[person(.9)])[0]
    low=ref.refine(frame,[person(.2)])[0]
    assert low['track_id']==high['track_id']
    assert low['tracking_association']=='low_confidence_recovery'
    empty=ObservationRefiner({0:'player'},25,.45)
    assert empty.refine(frame,[person(.2)])==[]


def test_analytics_preserve_unknown_football_events_and_selected_ball_coverage():
    meta=dict(width=1000,height=600,fps=5,frame_count=10,duration_seconds=2)
    calc=VideoMeasurements(meta,5)
    for index in range(10):
        players=[dict(object_type='player',track_id=f'T{i}',bbox=[100*i,100,100*i+20,200],confidence=.9,team='kit-A',scene_id=0) for i in range(1,4)]
        calc.observe(index,players+[dict(object_type='ball',bbox=[108,198,112,202],confidence=.8,ball_selected=True)])
    result=calc.finish()
    assert result['selected_ball_candidate_coverage_pct']==100
    kit=result['provisional_kit_analytics'][0]
    assert kit['near_ball_seconds']==2
    assert kit['image_shape']['width_pct']==20
    assert result['event_candidates']==[]
    assert 'possession' not in result and 'passes' not in result


def test_secondary_ball_inference_is_sampled_and_preserves_its_provenance(monkeypatch,tmp_path):
    import torch
    import ultralytics
    from types import SimpleNamespace
    from ultralytics.engine.results import Boxes
    from workers.video.detector import FootballDetector
    calls=[]
    primary_confidences=[]
    class Model:
        def __init__(self,path):
            self.secondary='rescue' in str(path)
            self.names={32:'sports ball'} if self.secondary else {0:'ball',1:'player'}
        def predict(self,frame,**kwargs):
            calls.append(self.secondary)
            if not self.secondary:primary_confidences.append(kwargs['conf'])
            data=[[20,20,24,24,.8,32]] if self.secondary else []
            return [SimpleNamespace(names=self.names,boxes=Boxes(torch.tensor(data).reshape(-1,6),frame.shape[:2]))]
    monkeypatch.setattr(ultralytics,'YOLO',Model)
    detector=FootballDetector(tmp_path/'primary.pt',device='cpu',image_size=640,ball_confidence=.5,
        object_confidence=.45,sample_fps=25,ball_rescue_weights=tmp_path/'rescue.pt')
    frame=np.zeros((80,120,3),np.uint8)
    first=detector.detect(frame)
    assert len(first)==1 and first[0]['model_role']=='secondary_ball_detector'
    assert first[0]['source']=='model_detection'
    assert detector.detect(frame)==[]
    assert calls.count(True)==1
    assert primary_confidences==[.1,.1]  # Raising the ball threshold cannot disable weak-person recovery.
    for _ in range(23):detector.detect(frame)
    assert calls.count(True)==2  # At 25fps the global fallback stays at or below 2fps.


def test_a_green_shirt_on_grass_is_still_a_shirt_colour():
    # A lime shirt on darker grass. Dropping every green pixel left nothing of the shirt; only grass-coloured pixels are dropped now.
    grass, lime, white = (40, 120, 50), (90, 240, 170), (235, 235, 235)   # BGR
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = grass
    model = JerseyTeams()
    def shirt(x, colour):
        frame[60:200, x:x + 40] = colour
        return [x, 60, x + 40, 200]
    green_box, white_box = shirt(60, lime), shirt(200, white)
    green_feature, white_feature = model.feature(frame, green_box), model.feature(frame, white_box)
    assert green_feature is not None and white_feature is not None
    assert np.linalg.norm(green_feature - white_feature) > 25          # far enough apart to be two kit groups
    # A box holding nothing but grass has no shirt in it.
    assert model.feature(frame, [130, 60, 170, 200]) is None
    # With no grass beside the player to compare against, green pixels are dropped as before.
    indoor = np.full((240, 320, 3), 128, dtype=np.uint8)
    indoor[60:200, 60:100] = lime
    assert model.grass_colour(indoor, [60, 60, 100, 200]) is None and model.feature(indoor, [60, 60, 100, 200]) is None
