"""Versioned, camera-relative measurements from real detection observations.

No football events, verified identities, pitch distances or possession are inferred.
"""
from collections import Counter
from itertools import groupby
import math

PLAYERS = {"player", "goalkeeper", "person"}
BALLS = {"ball", "sports ball"}


class VideoMeasurements:
    def __init__(self, metadata, sample_fps):
        self.metadata, self.fps = metadata, sample_fps
        self.frames = self.ball_frames = self.visible_total = self.max_visible = 0
        self.missing_run = self.longest_missing = self.ambiguous = self.associated = 0
        self.confidence_total = self.ball_count = 0
        self.tracks = {}
        self.grid = [[0 for _ in range(12)] for _ in range(8)]
        self.timeline = {}
        self.selected_ball_frames = 0
        self.kits, self.kit_proximity, self.kit_shapes = Counter(), Counter(), {}
        self.transfers, self.control = [], None

    def observe(self, frame_idx, rows):
        width, height = self.metadata["width"], self.metadata["height"]
        timestamp = frame_idx / self.metadata["fps"]
        players = [r for r in rows if r["object_type"] in PLAYERS]
        balls = [r for r in rows if r["object_type"] in BALLS]
        selected = [r for r in balls if r.get('ball_selected')]
        self.selected_ball_frames += bool(selected)
        self.frames += 1
        self.visible_total += len(players)
        self.max_visible = max(self.max_visible, len(players))
        self.ball_frames += bool(balls)
        self.ambiguous += len(balls) > 1
        self.missing_run = 0 if balls else self.missing_run + 1
        self.longest_missing = max(self.longest_missing, self.missing_run)
        self.ball_count += len(balls)
        self.confidence_total += sum(r["confidence"] for r in balls)
        duration = self.metadata["duration_seconds"]
        bucket = min(11, int(timestamp / max(duration / 12, 0.001)))
        segment = self.timeline.setdefault(bucket, Counter())
        segment.update(frames=1, ball_frames=int(bool(balls)), visible_players=len(players))
        known = []
        kit_positions = {}
        for row in players:
            a, b, c, d = row["bbox"]
            x, y = (a + c) / 2, d
            col = max(0, min(11, int(x / width * 12)))
            line = max(0, min(7, int(y / height * 8)))
            self.grid[line][col] += 1
            kit = f"scene-{row.get('scene_id', 0)}/{row['team']}" if row.get('team') else None
            if kit:
                self.kits[kit] += 1
                kit_positions.setdefault(kit, []).append((x / width, y / height))
            identity = row.get("track_id")
            if not identity:
                continue
            track = self.tracks.setdefault(identity, dict(track_id=identity, object_type=row["object_type"],
                observations=0, first_seen_seconds=timestamp, last_seen_seconds=timestamp,
                image_displacement_px=0.0, excluded_movement_steps=0, near_ball_frames=0,
                position_sum_x=0.0, position_sum_y=0.0, last=None))
            if track["last"]:
                old_t, old_x, old_y = track["last"]
                distance = math.hypot(x - old_x, y - old_y)
                if 0 < timestamp - old_t <= 2.5 / self.fps and distance <= 0.15 * math.hypot(width, height):
                    track["image_displacement_px"] += distance
                else:
                    track["excluded_movement_steps"] += 1
            track.update(observations=track["observations"] + 1, last_seen_seconds=timestamp,
                         last=(timestamp, x, y), position_sum_x=track["position_sum_x"] + x,
                         position_sum_y=track["position_sum_y"] + y)
            known.append((identity, x, y, max(1, d - b), kit))
        for kit, positions in kit_positions.items():
            if len(positions) >= 3:
                values = self.kit_shapes.setdefault(kit, Counter())
                xs, ys = zip(*positions)
                values.update(frames=1, centroid_x=sum(xs)/len(xs), centroid_y=sum(ys)/len(ys),
                              width=max(xs)-min(xs), depth=max(ys)-min(ys))
        # A candidate association only. Ambiguous/missing balls and distant players stay unknown.
        association_balls = selected if any('ball_selected' in r for r in balls) else balls
        associated = None
        if len(association_balls) == 1 and known:
            a, b, c, d = association_balls[0]["bbox"]
            bx, by = (a + c) / 2, (b + d) / 2
            candidates = sorted((math.hypot(x - bx, y - by) / body_height, identity, kit, x, y)
                                for identity, x, y, body_height, kit in known)
            nearest, identity, kit, x, y = candidates[0]
            separated = len(candidates) == 1 or candidates[1][0] - nearest >= 0.15
            if nearest <= 0.75 and separated:
                self.tracks[identity]["near_ball_frames"] += 1
                self.associated += 1
                if kit:
                    self.kit_proximity[kit] += 1
                    associated = dict(identity=identity, kit=kit, x=x, y=y, time=timestamp, frames=1)
        if associated:
            old = self.control
            if old and old['identity'] == associated['identity'] and timestamp-old['time'] <= 2.5/self.fps:
                associated['frames'] = old['frames']+1
            elif (old and old['identity'] != associated['identity'] and old['kit']==associated['kit']
                  and old['frames']/self.fps >= .2 and timestamp-old['time'] <= .5):
                distance=math.hypot(x-old['x'],y-old['y'])
                if distance >= .02*math.hypot(width,height):
                    self.transfers.append(dict(type='possible_same_kit_transfer',timestamp_seconds=round(timestamp,3),
                        from_track=old['identity'],to_track=identity,kit=kit,image_separation_px=round(distance,2),
                        review_status='unreviewed',source='ball_proximity_heuristic'))
            self.control=associated

    def finish(self):
        def pct(value, total):
            return round(100 * value / total, 2) if total else None
        duration = self.metadata["duration_seconds"]
        tracks = []
        for item in self.tracks.values():
            n = item["observations"]
            tracks.append(dict(track_id=item["track_id"], object_type=item["object_type"], observations=n,
                observed_seconds=round(min(duration, n / self.fps), 3),
                visibility_pct=pct(n, self.frames),
                first_seen_seconds=round(item["first_seen_seconds"], 3), last_seen_seconds=round(item["last_seen_seconds"], 3),
                image_displacement_px=round(item["image_displacement_px"], 2),
                excluded_movement_steps=item["excluded_movement_steps"],
                average_image_x_pct=round(100 * item["position_sum_x"] / n / self.metadata["width"], 2),
                average_image_y_pct=round(100 * item["position_sum_y"] / n / self.metadata["height"], 2),
                near_ball_frames=item["near_ball_frames"]))
        timeline = [dict(start_seconds=round(i * duration / 12, 3), end_seconds=round((i + 1) * duration / 12, 3),
                         analyzed_frames=values["frames"], ball_coverage_pct=pct(values["ball_frames"], values["frames"]),
                         mean_visible_players=round(values["visible_players"] / values["frames"], 2))
                    for i, values in sorted(self.timeline.items())]
        return dict(schema_version="video-measurements-1", source="model_detection", review_status="unreviewed",
            analyzed_frames=self.frames, source_duration_seconds=duration,
            analyzed_frame_pct=pct(self.frames, self.metadata["frame_count"]),
            mean_visible_players=round(self.visible_total / self.frames, 2) if self.frames else None,
            max_visible_players=self.max_visible, ball_coverage_pct=pct(self.ball_frames, self.frames),
            ball_detected_frames=self.ball_frames, ambiguous_ball_frames=self.ambiguous,
            mean_ball_confidence=round(self.confidence_total / self.ball_count, 4) if self.ball_count else None,
            longest_ball_detection_gap_seconds=round(min(duration, self.longest_missing / self.fps), 3),
            ball_proximity_frames=self.associated, proximity_association_coverage_pct=pct(self.associated, self.frames),
            selected_ball_candidate_frames=self.selected_ball_frames,
            selected_ball_candidate_coverage_pct=pct(self.selected_ball_frames, self.frames),
            provisional_kit_analytics=[dict(kit=kit, player_observations=n,
                near_ball_frames=self.kit_proximity[kit],near_ball_seconds=round(self.kit_proximity[kit]/self.fps,3),
                share_of_known_kit_proximity_pct=pct(self.kit_proximity[kit],sum(self.kit_proximity.values())),
                image_shape={key+'_pct':round(100*values[key]/values['frames'],2) for key in ['centroid_x','centroid_y','width','depth']}
                    if (values:=self.kit_shapes.get(kit)) else None)
                for kit,n in self.kits.items()],
            event_candidates=self.transfers,
            tracks=sorted(tracks, key=lambda t: (-t["observations"], t["track_id"])),
            timeline=timeline, image_occupancy_grid=self.grid,
            definitions=dict(visible_players="Player/goalkeeper/person detections per analyzed frame; not unique verified players.",
                movement="Sum of foot-point image displacements in pixels; excludes gaps >2.5 sample intervals and jumps >15% image diagonal. Includes camera motion and tracking error.",
                proximity="One continuity-selected observed ball candidate (or exactly one raw candidate in legacy runs), nearest confirmed player foot point within 0.75 player-box height, with at least 0.15 height-normalized separation from the next player. Not possession or touches.",
                occupancy="Counts of player foot-point observations in a 12x8 IMAGE grid; not a calibrated pitch heatmap.",
                confidence="Model confidence, not measured detection accuracy.",
                kits="Unreviewed anonymous jersey-colour groups, scoped to each scene; not verified teams or roster identity. Shape requires at least three visible grouped players and uses image percentages, not pitch metres.",
                selected_ball="At most one observed candidate passes confidence and short-gap continuity/margin checks. No missing positions are interpolated. Selection is not proof of a correct detection.",
                event_candidates="Possible same-kit transfers between distinct temporary tracks require prior association >=0.2 seconds, gap <=0.5 seconds and image separation >=2% of image diagonal. They are review candidates, not counted completed passes or possession.",
                visibility="Observed sampled intervals / all analyzed intervals. Gaps are not filled; observed seconds are estimates at the analysis sampling rate."))


def from_tracking(rows, metadata, summary):
    """Backfill existing completed short-video runs from their FULL tracking file."""
    from workers.video.track_linking import linked_view
    rows, identity_source = linked_view(rows)
    fps = summary["sample_fps"]
    calculator = VideoMeasurements(metadata, fps)
    groups = {index: list(group) for index, group in groupby(rows, key=lambda row: row["frame_idx"])}
    stride = max(1, round(metadata["fps"] / fps))
    for index in range(0, summary["decoded_frames"], stride):
        calculator.observe(index, groups.get(index, []))
    result = calculator.finish()
    if identity_source == "linked_track_id":
        # Track entries are offline-linked identities (L…), still anonymous and unreviewed.
        result["track_identity"] = identity_source
    return result
