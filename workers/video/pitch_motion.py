"""Estimated movement on the pitch: where each kit group was, how far and how fast it moved.

Everything here rests on automatic pitch calibration (pitch_keypoints.py, pitch_camera.py)
and inherits its assumptions. These are estimates for the time a player was tracked in a
calibrated frame, not match totals:

- Only frames with a pitch calibration count. Nothing is filled in for other frames.
- A track is one unbroken stretch of one tracker identity. Identities are unreviewed, so a
  track is not a verified player, and one player can be several tracks.
- Positions are averaged over about half a second and distance is summed in half-second
  hops. That keeps box jitter from being counted as running, but a standing player still
  collects a little distance from noise, and sharp turns inside a hop are cut short.
- Accuracy has not been evaluated against measured positions or speeds.
"""
from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

from workers.video import pitch_keypoints as pk

VERSION = "pitch-motion-1"
CELL_M = 5.
SMOOTH_S = .5  # positions are averaged over this long
HOP_S = .5  # distance and speed are taken over hops this long
MAX_SPEED_M_S = 11.  # faster than any footballer sprints; such a hop is an error, not movement
MIN_TRACK_S = 2.  # a track must be measured this long to be listed
MIN_CONFIDENCE = .45
LIMITATIONS = ["Estimates from automatic pitch calibration on an assumed 105 x 68 m pitch. Not checked against measured positions, distances or speeds.",
               "Only time a player was tracked in a calibrated frame counts. These are not match totals, and the camera does not see the whole pitch.",
               "A track is one unbroken stretch of one unreviewed tracker identity, not a verified player. One player can be several tracks.",
               "Positions are averaged over half a second and distance is summed in half-second hops. A standing player still collects a little distance from noise; sharp turns inside a hop are cut short.",
               "A cell that was never in the camera's view is unknown, not empty."]


def _grid():
    columns, rows = round(pk.LENGTH_M / CELL_M), round(pk.WIDTH_M / CELL_M)
    return columns, rows, pk.LENGTH_M / columns, pk.WIDTH_M / rows


def _cell(x, y):
    columns, rows, dx, dy = _grid()
    return min(rows - 1, max(0, int(y // dy))), min(columns - 1, max(0, int(x // dx)))


def _measure(samples, step):
    """Distance (m), measured seconds, fastest hop (m/s) and rejected hops for one unbroken stretch of (t, x, y)."""
    points = np.array([(x, y) for _, x, y in samples], dtype=float)
    half = max(1, round(SMOOTH_S / step / 2))
    hop = max(1, round(HOP_S / step))
    if len(points) < 2 * half + 1 + hop:
        return 0., 0., None, 0
    # Centred average; the ends, where a full window does not fit, are left out rather than averaged lopsidedly.
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    smooth = np.column_stack([np.convolve(points[:, axis], kernel, mode="valid") for axis in (0, 1)])
    times = np.array([t for t, _, _ in samples], dtype=float)[half:len(points) - half]
    distance = seconds = 0.
    fastest, rejected = None, 0
    for i in range(0, len(smooth) - hop, hop):
        length, elapsed = float(np.linalg.norm(smooth[i + hop] - smooth[i])), float(times[i + hop] - times[i])
        if elapsed <= 0:
            continue
        if length / elapsed > MAX_SPEED_M_S:
            rejected += 1
            continue
        distance, seconds = distance + length, seconds + elapsed
        fastest = max(fastest or 0., length / elapsed)
    return distance, seconds, fastest, rejected


def pitch_motion(frames, metadata, sample_fps, pitch=None):
    columns, rows, cell_length, cell_width = _grid()
    base = dict(schema_version=VERSION, review_status="unreviewed", accuracy_evaluated=False,
                assumed_pitch=dict(length_m=pk.LENGTH_M, width_m=pk.WIDTH_M, measured=False),
                grid=dict(columns=columns, rows=rows, cell_length_m=round(cell_length, 3), cell_width_m=round(cell_width, 3)),
                parameters=dict(position_average_seconds=SMOOTH_S, hop_seconds=HOP_S, max_speed_m_s=MAX_SPEED_M_S,
                                min_listed_track_seconds=MIN_TRACK_S, min_detection_confidence=MIN_CONFIDENCE),
                limitations=LIMITATIONS)
    if pitch is None:
        return dict(base, status="unavailable", reason="Pitch calibration was not run for this analysis.", teams=[])
    width, height = metadata["width"], metadata["height"]
    # Calibration may have been tried on every Nth analysed frame only; an untried frame is not a missed one.
    tried = [frame["time"] for frame in frames if frame.get("frame") in pitch]
    gaps = np.diff(tried)
    step = float(np.median(gaps)) if len(gaps) else 1 / sample_fps
    centres = np.array([((c + .5) * cell_length, (r + .5) * cell_width) for r in range(rows) for c in range(columns)], dtype=np.float32)
    coverage = np.zeros(rows * columns)
    heat = defaultdict(lambda: np.zeros((rows, columns)))
    stretches = defaultdict(list)  # (scene, track, kit) -> list of unbroken [(t, x, y), ...]
    last_seen, calibrated_seconds = {}, 0.
    for frame in frames:
        record = pitch.get(frame.get("frame"))
        if not record or record.get("status") != "calibrated":
            continue
        # Each calibrated sample stands for one calibration interval, which is longer than a frame when only every Nth frame was tried.
        t, dt = frame["time"], step
        calibrated_seconds += dt
        outline = pk.visible_area(record, width, height)
        if len(outline) >= 3:
            polygon = np.array(outline, dtype=np.float32)
            coverage += dt * np.array([cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0 for x, y in centres])
        counts = defaultdict(int)
        for p in frame["players"]:
            counts[p["scene"], p["track"]] += 1
        players = [p for p in frame["players"] if p["kit"] and p["track"] and counts[p["scene"], p["track"]] == 1
                   and math.isfinite(p["confidence"]) and p["confidence"] >= MIN_CONFIDENCE
                   and all(math.isfinite(p[k]) for k in ("x", "y")) and 0 <= p["x"] <= width and 0 <= p["y"] <= height]
        if not players:
            continue
        xy, usable = pk.to_pitch(record, [(p["x"], p["y"]) for p in players])
        for p, (x, y), ok in zip(players, xy, usable):
            if not ok:
                continue
            key = (p["scene"], p["track"], p["kit"])
            if 0 <= x <= pk.LENGTH_M and 0 <= y <= pk.WIDTH_M:
                heat[p["scene"], p["kit"]][_cell(x, y)] += dt
            # A missed sample, in tracking or in calibration, ends the stretch: nothing is bridged.
            if key not in last_seen or t - last_seen[key] > 1.5 * step + 1e-8:
                stretches[key].append([])
            stretches[key][-1].append((t, float(x), float(y)))
            last_seen[key] = t
    if not calibrated_seconds:
        return dict(base, status="unavailable", reason="No analysed frame passed the pitch calibration checks.", teams=[])
    tracks = defaultdict(list)
    for (scene, track, kit), parts in stretches.items():
        distance = seconds = 0.
        fastest, rejected = None, 0
        for part in parts:
            d, s, f, r = _measure(part, step)
            distance, seconds, rejected = distance + d, seconds + s, rejected + r
            if f is not None:
                fastest = max(fastest or 0., f)
        if seconds > 0:
            tracks[scene, kit].append(dict(track_id=track, measured_seconds=round(seconds, 2), distance_m=round(distance, 1),
                                           mean_speed_m_s=round(distance / seconds, 2), fastest_hop_m_s=round(fastest, 2), rejected_hops=rejected))
    teams = []
    for scene, kit in sorted(set(heat) | set(tracks), key=lambda g: (g[0], g[1])):
        rows_ = sorted(tracks.get((scene, kit), []), key=lambda row: (-row["measured_seconds"], row["track_id"]))
        seconds, distance = sum(row["measured_seconds"] for row in rows_), sum(row["distance_m"] for row in rows_)
        listed = [row for row in rows_ if row["measured_seconds"] >= MIN_TRACK_S]
        quickest = max(listed, key=lambda row: row["fastest_hop_m_s"], default=None)
        grid = heat.get((scene, kit), np.zeros((rows, columns)))
        teams.append(dict(kit=kit, scene_id=scene, placed_player_seconds=round(float(grid.sum()), 2),
                          heatmap_player_seconds=[[round(float(v), 2) for v in row] for row in grid],
                          measured_player_seconds=round(seconds, 2), distance_m=round(distance, 1),
                          mean_speed_m_s=round(distance / seconds, 2) if seconds else None,
                          fastest_hop_m_s=quickest["fastest_hop_m_s"] if quickest else None, fastest_hop_track=quickest["track_id"] if quickest else None,
                          tracks_measured=len(rows_), tracks=listed[:40], tracks_listed_total=len(listed)))
    return dict(base, status="estimated", reason=None, calibrated_seconds=round(calibrated_seconds, 2),
                coverage_seconds=[[round(float(v), 2) for v in row] for row in coverage.reshape(rows, columns)], teams=teams)
