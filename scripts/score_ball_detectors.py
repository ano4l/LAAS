"""Score the saved output of deploy/modal_ball_eval.py on a CPU.

    python scripts/score_ball_detectors.py <folder> [--threshold 0.1]

For each detector run, at several confidence thresholds: the share of labelled-ball frames in which
a box lies on the ball, the share in which its most confident box does (what a frame-by-frame
choice would get right), and how many other boxes it produces per frame. Then the same by ball
size and place at one threshold, and for pairs of runs used together.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.video.ball_benchmark import MATCH_MIN_PX, MATCH_SIZES, SIZE_BANDS  # noqa: E402

DEV = "SNGS-116,SNGS-121,SNGS-126,SNGS-131,SNGS-136,SNGS-140,SNGS-145,SNGS-150,SNGS-191,SNGS-196".split(",")
HELD_OUT = "SNGS-118,SNGS-138,SNGS-193".split(",")
THRESHOLDS = (.05, .1, .2, .3, .5)


def on_ball(box, ball):
    cx, cy = (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2
    reach = max(MATCH_MIN_PX, MATCH_SIZES * max(ball[2] - ball[0], ball[3] - ball[1]))
    return (((box[0] + box[2]) / 2 - cx) ** 2 + ((box[1] + box[3]) / 2 - cy) ** 2) ** .5 <= reach


def frames_of(clips):
    """Every labelled-ball frame: (clip, frame, ball box, at a player?, {run: boxes})."""
    for clip in clips:
        for frame, ball in clip["balls"].items():
            cx, cy = (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2
            at_player = any(b[0] - 2 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2 for b in clip["people"].get(frame, []))
            yield clip, frame, ball, at_player, {name: run.get(frame, []) for name, run in clip["runs"].items()}


def measure(clips, boxes_of, threshold):
    found = top = spare = total = 0
    for clip, frame, ball, at_player, runs in frames_of(clips):
        boxes = [b for b in boxes_of(clip, frame, runs) if b[4] >= threshold]
        total += 1
        hits = [b for b in boxes if on_ball(b, ball)]
        found += bool(hits)
        top += bool(boxes) and on_ball(max(boxes, key=lambda b: b[4]), ball)
        spare += len(boxes) - len(hits)
    return dict(found=found / total, top=top / total, spare=spare / total, frames=total) if total else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--threshold", type=float, default=.1)
    args = parser.parse_args()
    clips = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(args.folder).glob("SNGS-*.json"))}
    names = list(next(iter(clips.values()))["runs"])
    for title, wanted in (("development", DEV), ("held-out", HELD_OUT)):
        group = [clips[name] for name in wanted if name in clips]
        if not group:
            continue
        labelled = sum(len(clip["balls"]) for clip in group)
        print(f"\n== {title}: {len(group)} clips, {labelled} labelled-ball frames of {sum(len(clip['frames']) for clip in group)} sampled")
        stored = Counter()
        for clip, frame, ball, _, _ in frames_of(group):
            rows = clip["stored_pipeline"].get(frame, [])
            stored["found"] += any(on_ball(r, ball) for r in rows)
            stored["chosen_right"] += any(r[5] and on_ball(r, ball) for r in rows)
        print(f"   stored pipeline: ball among its candidates {stored['found'] / labelled:.1%}, chosen ball right {stored['chosen_right'] / labelled:.1%}")
        ms = defaultdict(list)
        for clip in group:
            for name, value in clip["ms_per_frame"].items():
                ms[name].append(value)
        print(f"   {'run':<28}{'ms':>6}  " + "  ".join(f"conf>={t:<4} found/top/other" for t in THRESHOLDS))
        for name in names:
            cells = []
            for threshold in THRESHOLDS:
                m = measure(group, lambda clip, frame, runs, name=name: runs[name], threshold)
                cells.append(f"{m['found']:>9.1%} {m['top']:>6.1%} {m['spare']:>5.1f}   ")
            print(f"   {name:<28}{sum(ms[name]) / len(ms[name]):>6.0f}  " + "".join(cells))
        t = args.threshold
        print(f"   pairs used together at conf >= {t}: found / other boxes per frame")
        for a in names:
            for b in names:
                if a < b:
                    m = measure(group, lambda clip, frame, runs, a=a, b=b: runs[a] + runs[b], t)
                    print(f"     {a + ' + ' + b:<52}{m['found']:>7.1%} {m['spare']:>6.1f}")
        if title == "development":
            print(f"   found at conf >= {t}, by ball size and place")
            for name in names:
                by = defaultdict(lambda: [0, 0])
                for clip, frame, ball, at_player, runs in frames_of(group):
                    size = max(ball[2] - ball[0], ball[3] - ball[1])
                    hit = any(b[4] >= t and on_ball(b, ball) for b in runs[name])
                    for key in (next(label for low, high, label in SIZE_BANDS if low <= size < high), "at a player" if at_player else "in the open"):
                        by[key][0] += hit
                        by[key][1] += 1
                print(f"     {name:<28}" + "  ".join(f"{key}: {hit / n:.0%} of {n}" for key, (hit, n) in sorted(by.items())))


if __name__ == "__main__":
    main()
