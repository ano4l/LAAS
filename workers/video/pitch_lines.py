"""Check a pitch mapping against the image: do the pitch's lines, drawn through it, land on white lines?

The landmark checks in pitch_keypoints.py only test whether the landmarks a model found agree
with each other. A model that invents a consistent set of landmarks, such as a penalty area
in midfield, passes them with a small error. Drawn through such a mapping, the pitch's
markings fall on plain grass. This module measures that:

1. White line pixels: thin bright structures (a morphological top-hat on brightness) on or
   next to grass.
2. The pitch's markings (touchlines, goal lines, halfway line, penalty and goal areas, centre
   circle, penalty arcs), sampled every half metre, are drawn into the image through the
   mapping. Points outside the image or beyond the horizon are left out.
3. Support is the share of the drawn points that have a white line pixel within a few pixels.

Players standing on a line, worn or shadowed lines and a blurred frame lower the support of
a correct mapping too, so the threshold is set from labelled data, not from first principles.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from workers.video import pitch_keypoints as pk

SAMPLE_M = .5  # spacing of the points drawn along each marking
TOLERANCE = .005  # a drawn point counts as on a line within this share of the image height (5 px at 1080p)
TOP_HAT = 1 / 70  # structuring element as a share of the image height: wider than a line, narrower than a mowing stripe
MIN_BRIGHTER = 25  # grey levels a line pixel must stand above its surroundings
MIN_STREAK = .03  # connected line pixels, as a share of the image height, below which a bright patch is texture
MIN_POINTS = 40  # fewer drawn points in view than this cannot be judged
MIN_MARKING_POINTS = 16  # 8 m of one marking in view before it is judged on its own


def markings(length=pk.LENGTH_M, width=pk.WIDTH_M, step=SAMPLE_M):
    """[(name, points in metres)] for every marking of a pitch of the assumed size, points about `step` apart."""
    mid, half = width / 2, length / 2
    area_w, area_l, goal_w, goal_l, radius, spot = 40.32, 16.5, 18.32, 5.5, 9.15, 11.

    def segment(a, b):
        a, b = np.array(a, float), np.array(b, float)
        count = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        return a + np.linspace(0, 1, count)[:, None] * (b - a)

    def arc(centre, start, end):
        count = max(2, int(math.ceil(radius * abs(end - start) / step)) + 1)
        angles = np.linspace(start, end, count)
        return np.column_stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)])

    parts = [("far touchline", segment((0, 0), (length, 0))), ("near touchline", segment((0, width), (length, width))),
             ("left goal line", segment((0, 0), (0, width))), ("right goal line", segment((length, 0), (length, width))),
             ("halfway line", segment((half, 0), (half, width))), ("centre circle", arc((half, mid), 0, 2 * math.pi))]
    # The arc outside each penalty area: where the circle round the spot crosses the area's line.
    reach = math.acos((area_l - spot) / radius)
    for side, x0, sign in (("left", 0., 1.), ("right", length, -1.)):
        def at(dx, y):
            return (x0 + sign * dx, y)
        parts += [(f"{side} penalty area, far side", segment(at(0, mid - area_w / 2), at(area_l, mid - area_w / 2))),
                  (f"{side} penalty area, front", segment(at(area_l, mid - area_w / 2), at(area_l, mid + area_w / 2))),
                  (f"{side} penalty area, near side", segment(at(area_l, mid + area_w / 2), at(0, mid + area_w / 2))),
                  (f"{side} goal area, far side", segment(at(0, mid - goal_w / 2), at(goal_l, mid - goal_w / 2))),
                  (f"{side} goal area, front", segment(at(goal_l, mid - goal_w / 2), at(goal_l, mid + goal_w / 2))),
                  (f"{side} goal area, near side", segment(at(goal_l, mid + goal_w / 2), at(0, mid + goal_w / 2))),
                  (f"{side} penalty arc", arc((x0 + sign * spot, mid), -reach, reach) if sign > 0 else arc((x0 - spot, mid), math.pi - reach, math.pi + reach))]
    return parts


PARTS = markings()
MARKINGS = np.concatenate([points for _, points in PARTS])
PART_OF = np.concatenate([np.full(len(points), index) for index, (_, points) in enumerate(PARTS)])


def line_mask(frame):
    """Boolean image of white line pixels: brighter than their surroundings, inside or next to a large grass area, and part of a
    connected streak rather than a speck of grass texture."""
    height = frame.shape[0]
    size = max(9, int(round(height * TOP_HAT)) | 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    grass = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (30, 40, 25), (95, 255, 255))
    grass = cv2.morphologyEx(cv2.morphologyEx(grass, cv2.MORPH_OPEN, disc), cv2.MORPH_CLOSE, disc)  # large grass areas, lines and players filled in
    near_grass = cv2.dilate(grass, disc) > 0
    brighter = cv2.morphologyEx(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.MORPH_TOPHAT, disc)
    candidates = ((brighter >= MIN_BRIGHTER) & near_grass).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(12, int(MIN_STREAK * height))
    keep[0] = False
    return keep[labels]


def drawn(matrix, width, height, world=False):
    """(image positions, marking index) of the pitch markings drawn through `matrix` (image -> pitch), only those in view.
    With `world`, also their pitch positions."""
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    inverse = np.linalg.inv(matrix)
    w = inverse[2, 0] * MARKINGS[:, 0] + inverse[2, 1] * MARKINGS[:, 1] + inverse[2, 2]
    front = np.abs(w) > 1e-9
    image = np.full((len(MARKINGS), 2), np.nan)
    image[front] = pk._project(inverse, MARKINGS[front])
    inside = front & (image[:, 0] >= 0) & (image[:, 0] < width) & (image[:, 1] >= 0) & (image[:, 1] < height)
    if inside.any():
        # Keep only points that map back to themselves on the same side of the horizon as the fit.
        back, usable = pk.to_pitch(dict(matrix=matrix.ravel()), image[inside])
        ok = usable & (np.linalg.norm(back - MARKINGS[inside], axis=1) < .05)
        inside[np.flatnonzero(inside)[~ok]] = False
    return (image[inside], PART_OF[inside], MARKINGS[inside]) if world else (image[inside], PART_OF[inside])


def support(mask_or_frame, matrix, distance=None, tolerance=TOLERANCE):
    """How well the markings drawn through `matrix` land on white lines in the image.

    Returns dict(share, points, markings_seen, markings_on_lines, weakest, weakest_share):
    `share` is the share of drawn points with a line pixel nearby (None when fewer than
    MIN_POINTS are in view); a marking counts as seen with at least MIN_MARKING_POINTS points
    in view, and as on lines when at least half of those are. Pass a frame, or a line mask
    with its `distance` transform to reuse them for several mappings of one frame.
    """
    if distance is None:
        mask = mask_or_frame if mask_or_frame.dtype == bool else line_mask(mask_or_frame)
        distance = cv2.distanceTransform(np.where(mask, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    height, width = distance.shape[:2]
    points, parts = drawn(matrix, width, height)
    out = dict(share=None, points=int(len(points)), markings_seen=0, markings_on_lines=0, weakest=None, weakest_share=None)
    if len(points) < MIN_POINTS:
        return out
    near = distance[points[:, 1].astype(int), points[:, 0].astype(int)] <= max(2., tolerance * height)
    out["share"] = round(float(near.mean()), 3)
    shares = {int(part): float(near[parts == part].mean()) for part in np.unique(parts) if (parts == part).sum() >= MIN_MARKING_POINTS}
    out["markings_seen"] = len(shares)
    out["markings_on_lines"] = sum(value >= .5 for value in shares.values())
    if shares:
        weakest = min(shares, key=shares.get)
        out.update(weakest=PARTS[weakest][0], weakest_share=round(shares[weakest], 3))
    return out


def distance_to_lines(frame):
    """Distance (px) from every pixel to the nearest white line pixel."""
    return cv2.distanceTransform(np.where(line_mask(frame), 0, 255).astype(np.uint8), cv2.DIST_L2, 3)


def refine(mask, matrix, radii=(.06, .04, .03, .02, .015, .01, .01, .01)):
    """Snap a mapping onto the white lines: match each drawn marking point to the nearest line pixel within a shrinking radius
    (a share of the image height) and refit. Returns the new image -> pitch matrix, or None when too little matches."""
    height, width = mask.shape
    distance, labels = cv2.distanceTransformWithLabels(np.where(mask, 0, 255).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask)  # line pixels in the order the labels number them
    coords = np.column_stack([xs, ys]).astype(float)
    current = np.asarray(matrix, dtype=float).reshape(3, 3)
    for radius in radii:
        image, parts, world = drawn(current, width, height, world=True)
        if len(image) < MIN_POINTS:
            return None
        px = image.astype(int)
        close = distance[px[:, 1], px[:, 0]] <= radius * height
        if close.sum() < 30 or len(set(parts[close].tolist())) < 2:
            return None
        target = coords[labels[px[:, 1], px[:, 0]][close] - 1]
        fitted, _ = cv2.findHomography(target, world[close], cv2.RANSAC, .5)
        if fitted is None or not np.isfinite(fitted).all() or abs(np.linalg.det(fitted)) < 1e-12:
            return None
        orientation, w = pk._orientation(fitted, target)
        if (orientation > 0).mean() <= .9:
            return None
        current = fitted * np.sign(np.median(w)) / np.linalg.norm(fitted)
    return current
