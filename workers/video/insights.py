"""Automatic, unreviewed video insights with explicit unknown time.

Image geometry is used throughout. These rules are not a trained football event
classifier, and do not write approved events or infer named/home-away identity.
"""
from collections import Counter, defaultdict
from itertools import groupby
import math

VERSION = "video-insights-10"
PLAYERS = {"player", "goalkeeper", "person"}
BALLS = {"ball", "sports ball"}


def pct(value, total):
    return round(100 * value / total, 2) if total else None


class VideoInsights:
    def __init__(self, metadata, sample_fps, context=None, pitch=None):
        self.meta, self.fps = metadata, sample_fps
        self.pitch = pitch  # {frame_idx: calibration record}, or None when calibration was not run
        self.context = context
        self.frames = []

    def observe(self, frame_idx, rows):
        t = frame_idx / self.meta["fps"]
        duration = max(0, min(1 / self.fps, self.meta["duration_seconds"] - t))
        if not duration:
            return
        players = []
        for r in rows:
            if r["object_type"] not in PLAYERS:
                continue
            a, b, c, d = r["bbox"]
            kit = (f"scene-{r.get('scene_id', 0)}/{r['team']}" if r.get("team")
                   and r.get("team_assignment_confidence", 1) >= .25 else None)
            players.append(dict(track=r.get("track_id"), kit=kit, scene=r.get("scene_id", 0),
                                x=(a+c)/2, y=d, height=max(1, d-b), confidence=r["confidence"], role=r["object_type"]))
        balls = [r for r in rows if r["object_type"] in BALLS]
        # In a legacy run with no continuity labels, only a single raw ball is eligible.
        selected = ([r for r in balls if r.get("ball_selected")]
                    if any("ball_selected" in r for r in balls) else balls)
        ball = None
        # A weak detection is used only when the offline ball path placed it between confident ones.
        if len(selected) == 1 and (selected[0]["confidence"] >= .25 or selected[0].get("ball_path_support")):
            r = selected[0]
            a, b, c, d = r["bbox"]
            ball = dict(x=(a+c)/2, y=(b+d)/2, confidence=r["confidence"],
                        scene=r.get("scene_id", 0), ground=r.get("field_plausibility") != "outside_ground_region")
        controller = None
        if ball and ball["ground"]:
            ranked = sorted(((math.hypot(p["x"]-ball["x"], p["y"]-ball["y"])/p["height"], i)
                             for i, p in enumerate(players) if p["track"] and p["confidence"] >= .25
                             and p["scene"] == ball["scene"]))
            if ranked and ranked[0][0] <= .65 and (len(ranked) == 1 or ranked[1][0]-ranked[0][0] >= .2):
                controller = dict(players[ranked[0][1]], distance=ranked[0][0])
        scene = ball["scene"] if ball else (players[0]["scene"] if players else None)
        self.frames.append(dict(frame=frame_idx, time=t, duration=duration, players=players, ball=ball,
                                controller=controller, scene=scene))

    def finish(self):
        duration = self.meta["duration_seconds"]
        width, height = self.meta["width"], self.meta["height"]
        diagonal = math.hypot(width, height)
        groups, timeline = {}, {}
        ball_grid = [[0]*12 for _ in range(8)]
        ball_thirds = [0., 0., 0.]
        selected_seconds, player_count, grouped_count = 0., 0, 0
        path, part, prior = [], 0, None
        motion_steps, excluded_steps, motion_px = 0, 0, 0.
        spells, pending = [], []

        def close_spell():
            nonlocal pending
            if pending:
                seconds = sum(f["duration"] for f in pending)
                # Reject single-frame associations and fleeting fly-bys. No gap filling.
                if len(pending) >= 3 and seconds >= .24 - 1e-9:
                    first, last = pending[0], pending[-1]
                    p = first["controller"]
                    spells.append(dict(id=f"control-{len(spells)+1}", scene_id=p["scene"],
                        kit=p["kit"], track_id=p["track"], start_seconds=round(first["time"], 4),
                        end_seconds=round(last["time"]+last["duration"], 4),
                        observed_seconds=round(seconds, 4), observed_frames=len(pending),
                        mean_ball_confidence=round(sum(f["ball"]["confidence"] for f in pending)/len(pending), 3),
                        mean_foot_distance_heights=round(sum(f["controller"]["distance"] for f in pending)/len(pending), 3),
                        image_start={"x_pct": round(100*first["ball"]["x"]/width, 2), "y_pct": round(100*first["ball"]["y"]/height, 2)},
                        image_end={"x_pct": round(100*last["ball"]["x"]/width, 2), "y_pct": round(100*last["ball"]["y"]/height, 2)},
                        review_status="unreviewed"))
                pending = []

        for f in self.frames:
            t, dt, ball, controller = f["time"], f["duration"], f["ball"], f["controller"]
            bucket = min(max(0, math.ceil(duration/5)-1), int(t/5))
            segment = timeline.setdefault(bucket, dict(frames=0, observed_seconds=0., ball_seconds=0.,
                                                       control_seconds=0., kit_seconds=Counter()))
            segment["frames"] += 1
            segment["observed_seconds"] += dt
            player_count += len(f["players"])
            positions = defaultdict(list)
            for p in f["players"]:
                if p["kit"]:
                    grouped_count += 1
                    positions[p["kit"]].append(p)
                    g = groups.setdefault(p["kit"], dict(kit=p["kit"], scene_id=p["scene"], observations=0,
                        seconds=0., frames=0, thirds=[0, 0, 0], grid=[[0]*12 for _ in range(8)], shapes={}))
                    g["observations"] += 1
                    g["thirds"][min(2, max(0, int(p["x"]/width*3)))] += 1
                    g["grid"][min(7, max(0, int(p["y"]/height*8)))][min(11, max(0, int(p["x"]/width*12)))] += 1
            for kit, ps in positions.items():
                g = groups[kit]
                g["frames"] += 1
                g["seconds"] += dt
                if len(ps) >= 3:
                    shape = g["shapes"].setdefault(bucket, dict(frames=0, seconds=0., count=0., x=0., y=0., width=0., depth=0.))
                    xs, ys = [p["x"]/width for p in ps], [p["y"]/height for p in ps]
                    shape["frames"] += 1
                    shape["seconds"] += dt
                    for k, value in dict(count=len(ps), x=sum(xs)/len(xs), y=sum(ys)/len(ys), width=max(xs)-min(xs), depth=max(ys)-min(ys)).items():
                        shape[k] += value*dt
            if ball:
                selected_seconds += dt
                segment["ball_seconds"] += dt
                x, y = ball["x"], ball["y"]
                ball_grid[min(7, max(0, int(y/height*8)))][min(11, max(0, int(x/width*12)))] += 1
                ball_thirds[min(2, max(0, int(x/width*3)))] += dt
                valid_step = False
                if prior:
                    elapsed = t-prior["time"]
                    distance = math.hypot(x-prior["ball"]["x"], y-prior["ball"]["y"])
                    valid_step = (0 < elapsed <= 1.5/self.fps and ball["scene"] == prior["ball"]["scene"]
                                  and distance <= diagonal*(.015+elapsed))
                    if valid_step:
                        motion_steps += 1
                        motion_px += distance
                    else:
                        excluded_steps += 1
                if not valid_step:
                    part += 1
                path.append(dict(timestamp_seconds=round(t, 4), x_pct=round(100*x/width, 2),
                                 y_pct=round(100*y/height, 2), scene_id=ball["scene"], segment_id=part))
                prior = f
            else:
                prior = None
            same = (pending and controller and valid_step and pending[-1]["controller"]["track"] == controller["track"]
                    and pending[-1]["controller"]["kit"] == controller["kit"]
                    and pending[-1]["controller"]["scene"] == controller["scene"]
                    and 0 < t-pending[-1]["time"] <= 1.5/self.fps)
            if not same:
                close_spell()
            if controller:
                pending.append(f)
        close_spell()
        kit_seconds = Counter()
        control_seconds = sum(s["observed_seconds"] for s in spells)
        for s in spells:
            if s["kit"]:
                kit_seconds[s["kit"]] += s["observed_seconds"]
            for bucket, segment in timeline.items():
                overlap = max(0, min(s["end_seconds"], (bucket+1)*5, duration)-max(s["start_seconds"], bucket*5))
                segment["control_seconds"] += overlap
                if s["kit"]:
                    segment["kit_seconds"][s["kit"]] += overlap

        candidates = []
        for a, b in zip(spells, spells[1:]):
            gap = b["start_seconds"]-a["end_seconds"]
            if (a["scene_id"] != b["scene_id"] or a["track_id"] == b["track_id"] or not a["kit"] or not b["kit"]
                    or gap < -1e-6 or gap > 1.2):
                continue
            bridge = [f for f in self.frames if a["end_seconds"]-1e-5 <= f["time"] < b["start_seconds"]-1e-5]
            # Any missing, aerial/outside-ground, cut or discontinuous bridge invalidates the candidate.
            if any(not f["ball"] or not f["ball"]["ground"] or f["scene"] != a["scene_id"] for f in bridge):
                continue
            ax, ay = a["image_end"]["x_pct"]*width/100, a["image_end"]["y_pct"]*height/100
            bx, by = b["image_start"]["x_pct"]*width/100, b["image_start"]["y_pct"]*height/100
            separation = math.hypot(bx-ax, by-ay)
            # Suppress track splits at virtually the same position.
            if separation < .012*diagonal:
                continue
            points = [(a["end_seconds"]-1/self.fps, ax, ay)] + [(f["time"], f["ball"]["x"], f["ball"]["y"]) for f in bridge] + [(b["start_seconds"], bx, by)]
            if any(t1-t0 > 1.5/self.fps+1e-5 or math.hypot(x1-x0, y1-y0) > diagonal*(.015+max(0, t1-t0))
                   for (t0, x0, y0), (t1, x1, y1) in zip(points, points[1:])):
                continue
            candidates.append(dict(id=f"candidate-{len(candidates)+1}", type="possible_pass" if a["kit"] == b["kit"] else "possible_turnover",
                timestamp_seconds=b["start_seconds"], start_seconds=a["end_seconds"], end_seconds=b["start_seconds"],
                evidence_start_seconds=round(max(0, a["start_seconds"]-1), 3), evidence_end_seconds=round(min(duration, b["end_seconds"]+1), 3),
                scene_id=a["scene_id"], from_track=a["track_id"], to_track=b["track_id"], from_kit=a["kit"], to_kit=b["kit"],
                image_separation_px=round(separation, 2), supported_control_seconds=round(a["observed_seconds"]+b["observed_seconds"], 3),
                observed_bridge_frames=len(bridge), review_status="unreviewed", evidence="Sustained control at both ends; observed ball bridge; distinct tracks and separated image positions."))

        from workers.video.sequences import infer_sequences
        sequences = infer_sequences(self.frames, self.meta, self.fps, self.context, self.pitch)
        # One review queue: passes and turnovers, then the rule-based shots, tackles, duels and dribbles.
        # A pass estimated on the pitch map joins the queue unless the stricter image rule already lists the same two players
        # at that moment; that one then carries the pass's place and outcome, for the review form.
        listed = sequences['candidates']
        extra = []
        for row in sequences['passes']['passes']:
            same = next((e for e in listed if e['from_track'] == row['from_track'] and e['to_track'] == row['to_track']
                         and abs(e['timestamp_seconds']-row['timestamp_seconds']) <= 2), None)
            if same:
                same.update({key: row[key] for key in ('start_m', 'end_m', 'length_m', 'suggested_outcome')})
            else:
                extra.append(row)
        candidates = sorted(listed+extra+sequences['actions']['candidates'], key=lambda e: e['timestamp_seconds'])
        grouped_seconds = sum(kit_seconds.values())
        controls = dict(estimated_control_seconds=round(control_seconds, 3), unknown_seconds=round(max(0, duration-control_seconds), 3),
                        coverage_pct=pct(control_seconds, duration), grouped_control_seconds=round(grouped_seconds, 3),
                        grouped_coverage_pct=pct(grouped_seconds, duration), spells=spells,
                        by_kit=[dict(kit=kit, estimated_seconds=round(seconds, 3), share_of_known_control_pct=pct(seconds, grouped_seconds), clip_share_pct=pct(seconds, duration))
                                for kit, seconds in sorted(kit_seconds.items())])
        kit_result = []
        for kit, g in sorted(groups.items()):
            shape_timeline = [dict(start_seconds=i*5, end_seconds=round(min(duration, (i+1)*5), 3), supported_frames=s["frames"],
                              supported_seconds=round(s["seconds"], 3), mean_visible_players=round(s["count"]/s["seconds"], 2),
                              **{out: round(100*s[key]/s["seconds"], 2) for key, out in [("x", "centroid_x_pct"), ("y", "centroid_y_pct"), ("width", "width_pct"), ("depth", "depth_pct")]})
                              for i, s in sorted(g["shapes"].items())]
            kit_result.append(dict(kit=kit, scene_id=g["scene_id"], player_observations=g["observations"], visible_frames=g["frames"],
                visible_seconds=round(g["seconds"], 3), image_thirds_pct=[pct(n, g["observations"]) for n in g["thirds"]],
                image_occupancy_grid=g["grid"], shape_timeline=shape_timeline))
        slices = [dict(start_seconds=i*5, end_seconds=round(min(duration, (i+1)*5), 3), analyzed_frames=s["frames"],
                      ball_observed_seconds=round(s["ball_seconds"], 3), ball_coverage_pct=pct(s["ball_seconds"], min(5, duration-i*5)),
                      estimated_control_seconds=round(s["control_seconds"], 3), unknown_seconds=round(max(0, min(5, duration-i*5)-s["control_seconds"]), 3),
                      kit_control_seconds={k: round(v, 3) for k, v in s["kit_seconds"].items()}) for i, s in sorted(timeline.items())]
        # Bound UI payload. Segment IDs survive downsampling; gaps/cuts must never be connected.
        display_stride = max(1, math.ceil(len(path)/600))
        display_path = path[::display_stride]
        notes = [dict(kind="coverage", text=f"Sustained control supports {control_seconds:.2f}s of {duration:.2f}s; {max(0, duration-control_seconds):.2f}s remains unknown.")]
        if grouped_seconds < duration*.2:
            notes.append(dict(kind="limitation", text="Kit control coverage is below 20% of this clip. Group shares cannot describe match possession."))
        notes.append(dict(kind="events", text=f"{len([e for e in candidates if e['type']=='possible_pass'])} possible passes and {len([e for e in candidates if e['type']=='possible_turnover'])} possible turnovers meet the evidence rules. Review each in the source video."))
        counts = sequences['actions']['counts']
        notes.append(dict(kind="actions", text=f"{counts['shot']} possible shots, {counts['tackle']} possible tackles, {counts['duel']} possible duels and {counts['dribble']} possible dribbles were proposed by rules over the tracked ball and players. They have not been checked against labelled events and miss most real ones; none is counted until a person records it under Review."))
        notes.append(dict(kind="possession", text=f"Temporal team estimates support {sequences['known_seconds']:.2f}s; {sequences['unknown_seconds']:.2f}s is unknown and {sequences['contested_seconds']:.2f}s contested. Shares cover known time only."))
        bookmarks = []
        if slices and selected_seconds:
            best = max(slices, key=lambda s: (s["ball_coverage_pct"] or 0, s["ball_observed_seconds"]))
            bookmarks.append(dict(id="best-ball-window", type="ball_visibility", label="Clearest observed ball window",
                start_seconds=best["start_seconds"], end_seconds=best["end_seconds"],
                detail=f"{best['ball_observed_seconds']:.2f}s of selected ball observations ({best['ball_coverage_pct']:.1f}% of this window)."))
        for g in kit_result:
            supported = [s for s in g["shape_timeline"] if s["supported_seconds"] >= .5]
            if supported:
                widest = max(supported, key=lambda s: s["width_pct"])
                bookmarks.append(dict(id=f"width-{g['kit']}", type="kit_shape", label=f"{g['kit']} · widest observed window",
                    start_seconds=widest["start_seconds"], end_seconds=widest["end_seconds"],
                    detail=f"Mean image width {widest['width_pct']:.1f}%, supported by {widest['supported_seconds']:.2f}s with at least three grouped players."))
                if len(supported) >= 2:
                    first, last = supported[0], supported[-1]
                    notes.append(dict(kind="shape", text=f"{g['kit']} image width changes from {first['width_pct']:.1f}% to {last['width_pct']:.1f}% between supported windows starting at {first['start_seconds']}s and {last['start_seconds']}s. Camera motion and perspective affect this comparison."))
        from workers.video.collective_dynamics import collective_dynamics
        dynamics=collective_dynamics(self.frames,self.meta,self.fps,self.pitch)
        return dict(schema_version=VERSION, source="full_tracking_observations", review_status="unreviewed", coordinate_system="camera_image",
            source_video={key: self.meta.get(key) for key in ["filename", "sha256", "width", "height", "fps", "frame_count"]}, sample_fps=self.fps,
            source_duration_seconds=duration, analyzed_frames=len(self.frames), analyzed_seconds=round(sum(f["duration"] for f in self.frames), 3),
            quality=dict(selected_ball_seconds=round(selected_seconds, 3), selected_ball_coverage_pct=pct(selected_seconds, duration),
                         grouped_player_observations=grouped_count, player_observations=player_count, kit_assignment_coverage_pct=pct(grouped_count, player_count), accuracy_evaluated=False),
            control=controls, sequences=sequences, collective_dynamics=dynamics, candidates=candidates, timeline=slices, kits=kit_result,
            ball=dict(observed_seconds=round(selected_seconds, 3), image_thirds_pct=[pct(n, selected_seconds) for n in ball_thirds],
                      image_occupancy_grid=ball_grid, path=display_path, path_observations=len(path), path_display_stride=display_stride,
                      linked_motion_steps=motion_steps, excluded_motion_steps=excluded_steps, linked_image_displacement_px=round(motion_px, 2)),
            findings=notes, bookmarks=bookmarks, parameters=dict(min_ball_confidence=.25, max_foot_distance_heights=.65, runner_up_margin_heights=.2,
                min_control_seconds=.24, min_control_frames=3, max_transfer_gap_seconds=1.2, image_step_gate="1.5 sample intervals; displacement <= diagonal * (0.015 + elapsed_seconds)"),
            definitions=dict(control="Estimated control requires one observed ball, a clearly nearer tracked player foot point and sustained consecutive support. Unknown gaps are never filled. This is not verified possession.",
                shares="Kit shares divide only sustained control assigned to anonymous scene-scoped kit groups. Clip shares and unknown time retain the full source duration denominator.",
                events="Temporal candidates link supported receivers to departures and allow motion-supported one-touch anchors. Strong transitions allow <=0.25s missing-ball gaps and >=65% observed support. Partial-tracking review-only candidates allow <=0.75s gaps and >=35% support with at least three observed positions; their flight never enters possession totals. Transit is bounded at four seconds. No coordinates are invented. Cuts, confirmed stoppages, contested/third-player transitions and near-position identity splits are rejected. These are unvalidated review candidates, not approved events.",
                actions="Possible shots, tackles, duels and dribbles come from rules over tracked positions, measured in player heights. A tackle is the ball changing kit group at close range; a duel is two opponents about equally close to the ball; a dribble is a player keeping the ball while an opponent comes close and falls away; a shot is a fast ball heading at a detected goalkeeper or at a goal on the estimated pitch map. They need the ball and both players to be tracked, so most real events are missed, and none has been checked against labelled events.",
                shape="Shape uses at least three visible grouped players per frame, weighted by sampled time, in five-second windows. Image width/depth/centroid include camera motion and perspective, and are not pitch distances or formation labels.",
                zones="Left/centre/right thirds and grids refer to the camera image, not attacking/defending pitch zones. Ball paths contain observed positions only; segment IDs split gaps, cuts and jumps.",
                confidence="Detection confidence and heuristic support are not measured accuracy. No named player, home/away identity, completed pass, confirmed shot, xG or calibrated physical metric is inferred."))


def from_tracking(rows, metadata, summary, context=None, pitch=None):
    from workers.video.track_linking import linked_view
    rows, identity_source = linked_view(rows)
    calc = VideoInsights(metadata, summary["sample_fps"], context, pitch)
    groups = {index: list(group) for index, group in groupby(sorted(rows, key=lambda r: r["frame_idx"]), key=lambda r: r["frame_idx"])}
    stride = max(1, round(metadata["fps"]/summary["sample_fps"]))
    for index in range(0, summary["decoded_frames"], stride):
        calc.observe(index, groups.get(index, []))
    result = calc.finish()
    if identity_source == "linked_track_id":
        result["track_identity"] = identity_source
    return result


def ensure_insights(manifest, directory):
    """Backfill only the selected real run. Return whether artifacts changed."""
    import json
    from pathlib import Path
    directory = Path(directory)
    from workers.video.sequence_context import load_context
    context=load_context(directory/'sequence-context.json',manifest)
    current=manifest.get('metrics',{}).get('video_insights') or {}
    if manifest.get("engine") != "yolo_bytetrack":
        return False
    # A calibration stored by an older version has its steadying step redone from its stored frames; no model is needed.
    from workers.video import pitch_keypoints as pk
    stored = (manifest.get("detection_summary") or {}).get("pitch_calibration") or {}
    upgraded = False
    if (stored.get("schema_version") not in (None, pk.VERSION) and (directory/"pitch-calibration.json").is_file()
            and {"width", "height", "fps"} <= manifest.get("metadata", {}).keys()):
        calibration = json.loads((directory/"pitch-calibration.json").read_text(encoding="utf-8"))
        if pk.upgrade(calibration, tuple(manifest["metadata"][key] for key in ("width", "height", "fps"))):
            pk._write_json(directory/"pitch-calibration.json", calibration)
            manifest["detection_summary"]["pitch_calibration"] = {key: calibration[key] for key in pk.SUMMARY_KEYS}
            upgraded = True
    if not upgraded and current.get("schema_version") == VERSION and current.get('sequences',{}).get('context_revision',0)==context['revision']:
        return False
    # Legacy/incomplete artifacts remain readable. Never synthesize analytics
    # from the small preview or assume absent full-tracking observations.
    if (not (directory/"tracking.json").is_file()
            or not {"width", "height", "fps", "duration_seconds"} <= manifest.get("metadata", {}).keys()
            or not {"sample_fps", "decoded_frames"} <= manifest.get("detection_summary", {}).keys()):
        return False
    pitch = None
    if (directory/"pitch-calibration.json").is_file():
        pitch = {int(index): record for index, record in json.loads((directory/"pitch-calibration.json").read_text(encoding="utf-8")).get("frames", {}).items()}
    manifest.setdefault("metrics", {})["video_insights"] = from_tracking(
        json.loads((directory/"tracking.json").read_text(encoding="utf-8")), manifest["metadata"], manifest["detection_summary"],context,pitch)
    import os
    import tempfile
    for name, value in [("metrics.json", manifest["metrics"]), ("manifest.json", manifest)]:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as stream:
            json.dump(value, stream)
            temporary = stream.name
        try:
            os.replace(temporary, directory/name)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return True
