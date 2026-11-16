"""Measure the CPU fallback settings against saved full-resolution detections.

Usage: python scripts/benchmark_cpu_fallback.py SOURCE.mp4 RAW.json [--threads 4] [--frames 30]

RAW.json is a capture from data/qa/tracker-diagnosis/capture_raw.py: the detector's raw rows
for every frame at image size 1280. It is the reference; nothing here is hand-labelled, so the
figures say how closely a cheaper setting reproduces the full setting, not how correct either is.

Reports, for this machine only:
  detector   ms per frame and agreement with the reference at each image size
  pitch      ms per frame for the pitch landmark model
  decode     seconds to read every frame against grabbing the frames in between
  tracker    ms per frame and continuity diagnostics for BoT-SORT and ByteTrack at the balanced stride
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workers.video.accuracy import box_iou
from workers.video.yolo_adapter import model_path, pitch_model_path

PEOPLE = {"person", "player", "goalkeeper", "referee"}
BALLS = {"ball", "sports ball"}


def agreement(reference, rows):
    """People: share of reference boxes found (IoU >= 0.5) and share of found boxes that are in the reference."""
    ref = [r["bbox"] for r in reference if r["object_type"] in PEOPLE and r["confidence"] >= .45]
    got = [r["bbox"] for r in rows if r["object_type"] in PEOPLE and r["confidence"] >= .45]
    used, matched = set(), 0
    for box in ref:
        best = max(((box_iou(box, other), i) for i, other in enumerate(got) if i not in used), default=(0, None))
        if best[0] >= .5:
            used.add(best[1])
            matched += 1
    ref_ball = max((r for r in reference if r["object_type"] in BALLS and r["confidence"] >= .3), key=lambda r: r["confidence"], default=None)
    ball_found = None
    if ref_ball:
        ball_found = any(box_iou(ref_ball["bbox"], r["bbox"]) > .1 for r in rows if r["object_type"] in BALLS)
    return len(ref), matched, len(got), ball_found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--sizes", default="1280,960,640")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from workers.video.detector import FootballDetector
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    total = len(raw["frames"])
    picks = sorted(set(np.linspace(0, total - 1, args.frames).astype(int).tolist()))
    capture, frames = cv2.VideoCapture(str(args.source)), {}
    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        if index in picks:
            frames[index] = frame
    capture.release()
    report = dict(source=args.source.name, threads=args.threads, cpu_capability=torch.backends.cpu.get_cpu_capability(),
                  torch=torch.__version__, sampled_frames=len(frames), detector={}, tracker={})

    for size in map(int, args.sizes.split(",")):
        detector = FootballDetector(model_path(ROOT), device="cpu", image_size=size, ball_confidence=.1,
                                    object_confidence=.45, sample_fps=raw["fps"])
        captured = []
        detector.refiner.refine = lambda frame, rows: captured.append(rows) or []
        detector.detect(next(iter(frames.values())))  # warm-up, not timed
        captured.clear()
        started = time.perf_counter()
        for frame in frames.values():
            detector.detect(frame)
        elapsed = time.perf_counter() - started
        ref_n = hit = got_n = balls = ball_hits = 0
        for index, rows in zip(frames, captured):
            a, b, c, ball = agreement(raw["frames"][index], rows)
            ref_n, hit, got_n = ref_n + a, hit + b, got_n + c
            if ball is not None:
                balls, ball_hits = balls + 1, ball_hits + int(ball)
        report["detector"][size] = dict(ms_per_frame=round(elapsed / len(frames) * 1000, 1),
                                        people_recall_vs_reference=round(hit / max(1, ref_n), 4),
                                        people_precision_vs_reference=round(hit / max(1, got_n), 4),
                                        ball_frames_in_reference=balls, ball_found=ball_hits)
        print(json.dumps({size: report["detector"][size]}), flush=True)

    pitch_weights = pitch_model_path(ROOT)
    if pitch_weights:
        from workers.video.pitch_keypoints import PitchKeypointModel
        model = PitchKeypointModel(pitch_weights, "cpu")
        sample = list(frames.values())[:10]
        model.calibrate(sample[0])
        started = time.perf_counter()
        for frame in sample:
            model.calibrate(frame)
        report["pitch_ms_per_frame"] = round((time.perf_counter() - started) / len(sample) * 1000, 1)

    def decode(stride):
        capture, count = cv2.VideoCapture(str(args.source)), 0
        started = time.perf_counter()
        while True:
            if stride > 1 and count % stride:
                if not capture.grab():
                    break
            else:
                ok, frame = capture.read()
                if not ok:
                    break
                cv2.resize(frame, (1280, 720)).tobytes()  # what the playback video needs from each written frame
            count += 1
        capture.release()
        return round(time.perf_counter() - started, 2)
    report["decode_seconds"] = dict(every_frame=decode(1), grab_between_stride_2=decode(2))

    spec = importlib.util.spec_from_file_location("replay", ROOT / "data/qa/tracker-diagnosis/replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    from workers.video.observation_refiner import ObservationRefiner
    from workers.video.track_linking import AppearanceBank, link_tracks
    stride = 2
    for backend in ("botsort", "bytetrack"):
        os.environ["TRACKER_BACKEND"] = backend
        refiner = ObservationRefiner(replay.NAMES, raw["fps"] / stride, .45)
        capture, rows, bank, spent, used = cv2.VideoCapture(str(args.source)), [], AppearanceBank(), 0.0, 0
        for index, observations in enumerate(raw["frames"]):
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride:
                continue
            started = time.perf_counter()
            emitted = refiner.refine(frame, [dict(r) for r in observations])
            spent, used = spent + time.perf_counter() - started, used + 1
            emitted = [dict(row, frame_idx=index // stride) for row in emitted]
            bank.observe(frame, emitted)
            rows.extend(emitted)
        capture.release()
        linked = link_tracks(rows, bank.descriptors(), raw["fps"] / stride)["summary"]["identities_after"]
        report["tracker"][backend] = dict(ms_per_frame=round(spent / used * 1000, 1), identities_after_linking=linked,
                                          **replay.diagnostics(rows))
        print(json.dumps({backend: report["tracker"][backend]}), flush=True)
    os.environ.pop("TRACKER_BACKEND", None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
