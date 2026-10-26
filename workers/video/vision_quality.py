"""Conservative geometry/continuity filters; these are not ground-truth labels."""
import math
import cv2
import numpy as np
from workers.video.accuracy import box_iou


def field_region(frame):
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (320, max(32, round(height * 320 / width))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (28, 35, 30), (95, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour) / grass.size
    if area < .20 or cv2.boundingRect(contour)[2] < grass.shape[1] * .55:
        return None  # Closeups, unusual surfaces and unreliable masks fail open.
    hull = cv2.convexHull(contour).astype(np.float32)
    hull[:, :, 0] *= width / grass.shape[1]
    hull[:, :, 1] *= height / grass.shape[0]
    return hull


def on_field(box, region, height, *, ball=False):
    if region is None:
        return True
    a, b, c, d = box
    point = ((a+c)/2, (b+d)/2 if ball else d)
    return cv2.pointPolygonTest(region, point, True) >= -height * (.04 if ball else .015)


def person_indices(boxes, class_ids, names, minimum_confidence, region, height):
    """Filter before ByteTrack; suppress overlapping human classes together."""
    candidates = []
    for index, box in enumerate(boxes):
        if int(box.cls.item()) not in class_ids or float(box.conf.item()) < minimum_confidence:
            continue
        coords = box.xyxy[0].tolist()
        if on_field(coords, region, height):
            candidates.append((index, coords, float(box.conf.item())))
    kept = []
    def fragment(a,b):
        intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        area_a=max(1,(a[2]-a[0])*(a[3]-a[1]));area_b=max(1,(b[2]-b[0])*(b[3]-b[1]))
        width_a,width_b=max(1,a[2]-a[0]),max(1,b[2]-b[0])
        aligned=abs((a[0]+a[2]-b[0]-b[2])/2)<.25*max(width_a,width_b) and min(width_a,width_b)/max(width_a,width_b)>.5
        return aligned and intersection/min(area_a,area_b)>.8 and min(area_a,area_b)/max(area_a,area_b)<.65
    for index, coords, score in sorted(candidates, key=lambda item: -item[2]):
        contained=[(i,prior) for i,prior in kept if fragment(coords,prior)]
        area=lambda box:(box[2]-box[0])*(box[3]-box[1])
        if any(area(prior)>area(coords) for _,prior in contained):
            continue
        if contained:
            kept=[(i,prior) for i,prior in kept if (i,prior) not in contained]
        if not any(box_iou(coords, prior) > .40 for _, prior in kept):
            kept.append((index, coords))
    return sorted(index for index, _ in kept)


class BallContinuity:
    def __init__(self, sample_fps):
        self.fps, self.last, self.velocity = sample_fps, None, (0., 0.)

    def reset(self):
        self.last, self.velocity = None, (0., 0.)

    def select(self, rows, shape, frame_number):
        """Select at most one observed candidate; never invent positions or fill gaps."""
        diagonal = math.hypot(shape[0], shape[1])
        candidates = []
        elapsed = (frame_number-self.last[0])/self.fps if self.last else None
        recent = elapsed is not None and 0 < elapsed <= .5
        for row in rows:
            a,b,c,d = row['bbox']; x,y = (a+c)/2,(b+d)/2
            row['ball_selected'] = False
            if row['confidence'] < .15:
                continue
            if recent:
                px = self.last[1]+self.velocity[0]*elapsed
                py = self.last[2]+self.velocity[1]*elapsed
                distance = math.hypot(x-px,y-py)
                gate = diagonal * (.02 + .30*elapsed)
                if distance > gate:
                    continue
                score = row['confidence'] - .3*distance/gate
            else:
                score = row['confidence']
            if row.get('field_plausibility')=='outside_ground_region':score-=.15
            candidates.append((score,x,y,row))
        candidates.sort(key=lambda item: -item[0])
        if not candidates:
            return
        best = candidates[0]
        if len(candidates)>1 and best[0]-candidates[1][0]<.15:
            return
        if not recent and len(rows)>1 and best[3]['confidence']<.4:
            return
        best[3]['ball_selected'] = True
        if recent:
            self.velocity = ((best[1]-self.last[1])/elapsed,(best[2]-self.last[2])/elapsed)
        else:
            self.velocity = (0.,0.)
        self.last=(frame_number,best[1],best[2])
