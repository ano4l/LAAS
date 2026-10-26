import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from scripts.refine_video_detections import refine
from workers.video import real_analysis


def test_refinement_checks_source_and_does_not_call_neural_model(tmp_path,monkeypatch):
    import ultralytics
    source=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),25,(64,48))
    for _ in range(10):writer.write(np.zeros((48,64,3),np.uint8))
    writer.release()
    prior=tmp_path/'prior';prior.mkdir()
    manifest=dict(run_id='a'*32,engine='yolo_bytetrack',metadata=dict(sha256='incorrect',frame_count=10,fps=25),
                  detection_summary=dict(analyzed_frames=10,model_classes={'0':'ball','1':'player'},model='prior.pt',model_sha256='old-source-model'))
    (prior/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='hash'):
        refine(source,prior,tmp_path/'outputs')
    manifest['metadata']['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    (prior/'manifest.json').write_text(json.dumps(manifest))
    rows=[dict(frame_idx=i,object_type='player',bbox=[5,5,15,40],confidence=.9,source='model_detection',model_role='object_detector',interpolated=False) for i in range(10)]
    (prior/'tracking.json').write_text(json.dumps(rows))
    def forbidden(*args,**kwargs):raise AssertionError('Reprocessing must not perform neural inference')
    monkeypatch.setattr(ultralytics,'YOLO',forbidden)
    monkeypatch.setattr(real_analysis,'yolo_capabilities',lambda _:dict(available=True))
    weights=tmp_path/'weights.pt';weights.write_bytes(b'controlled-test-weights')
    monkeypatch.setattr(real_analysis,'model_path',lambda _:weights)
    result,directory=refine(source,prior,tmp_path/'outputs')
    assert result['detection_summary']['model_sha256']=='old-source-model'
    assert result['detection_summary']['inference_source']=='existing_model_observations_reprocessed'
    assert result['metrics']['video_measurements']['analyzed_frames']==10
    assert real_analysis.video_metadata(directory/'annotated.mp4')['frame_count']==10
    assert (directory/'kit-groups.csv').is_file()
    assert (directory/'tracks.csv').is_file()
