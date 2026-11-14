"""BoT-SORT/ByteTrack association with expanded boxes for small, fast players.

Plain IoU collapses when a player only a few dozen pixels tall moves a large share of
their own width between frames, so the tracker drops the track and starts a new ID.
Growing both boxes before measuring overlap (the idea behind Deep-EIoU) keeps such
pairs comparable. Expansion affects matching only; reported boxes are unchanged.
"""
import numpy as np


def expanded(boxes, expansion):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    grow_x = (boxes[:, 2] - boxes[:, 0]) * expansion
    grow_y = (boxes[:, 3] - boxes[:, 1]) * expansion
    return boxes + np.stack([-grow_x, -grow_y, grow_x, grow_y], axis=1)


def expanded_tracker(base, expansion):
    """Subclass of an Ultralytics tracker whose first-stage and unconfirmed matching use expanded IoU."""
    from ultralytics.trackers.utils import matching

    class ExpandedIoUTracker(base):
        def get_dists(self, tracks, detections):
            dists = matching.iou_distance(list(expanded([t.xyxy for t in tracks], expansion)),
                                          list(expanded([d.xyxy for d in detections], expansion))) \
                if tracks and detections else matching.iou_distance(tracks, detections)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, detections)
            return dists

    return ExpandedIoUTracker
