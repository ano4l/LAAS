import numpy as np
import pytest

from workers.video.expanded_iou import expanded, expanded_tracker


def test_expansion_grows_each_side_by_a_share_of_box_size():
    assert expanded([[10, 20, 30, 60]], .5).tolist() == [[0, 0, 40, 80]]
    assert expanded([], .5).shape == (0, 4)


def test_small_fast_player_keeps_one_id_only_with_expanded_matching():
    pytest.importorskip("ultralytics")
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace
    import torch

    config = IterableSimpleNamespace(tracker_type="bytetrack", track_high_thresh=.45, track_low_thresh=.1, new_track_thresh=.5,
                                     track_buffer=25, match_thresh=.8, fuse_score=True)

    def identities(tracker):
        seen = set()
        for frame in range(12):
            # A 12x30 px player shifts 14 px in one frame (e.g. a camera jolt): plain IoU drops to zero.
            x = 100 if frame < 6 else 114
            boxes = Boxes(torch.tensor([[x, 100, x + 12, 130, .9, 0]], dtype=torch.float32), (480, 848))
            seen.update(int(t[4]) for t in tracker.update(boxes.cpu().numpy(), np.zeros((480, 848, 3), np.uint8)))
        return seen

    assert len(identities(BYTETracker(config))) > 1
    assert len(identities(expanded_tracker(BYTETracker, .5)(config))) == 1
