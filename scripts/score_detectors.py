"""Score saved detector output against the labelled SoccerNet boxes. CPU only, no model.

    python scripts/score_detectors.py data/qa/detector-eval [--clips SNGS-116,SNGS-121] [--json out.json]

Reads the files `deploy/modal_detector_eval.py` wrote. Each one holds, for every sampled frame of a
clip, the labelled boxes, each detector's raw boxes from confidence 0.05 up, the field outline, and
the boxes the stored pipeline run emitted. A variant here is a detector run plus what is done to
its boxes (a confidence threshold, the field filter, the pipeline's own suppression, joining
stacked pieces). Scoring is `workers/video/detection_diagnosis.py`.

Two scores are given. "players": only labelled players and goalkeepers must be found, and a box on
a referee is an error. "people": referees count too, which measures where the boxes are and
ignores who the detector says they are.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.video import detection_diagnosis as dd
from workers.video.people_boxes import adopt_roles, clean_people, join_stacked_pieces

PEOPLE_KINDS = {"player", "goalkeeper", "referee", "person"}


def rows_of(run, threshold):
    return {int(frame): [dict(bbox=box[:4], confidence=box[4], object_type=box[5]) for box in boxes if box[4] >= threshold and box[5] in PEOPLE_KINDS]
            for frame, boxes in run.items()}


def on_field(clip, rows_by_frame):
    """Drop boxes whose feet are off the grass outline, as the pipeline does before tracking."""
    height, out = clip["size"][1], {}
    for frame, rows in rows_by_frame.items():
        outline = clip["field"].get(str(frame))
        if outline is None:
            out[frame] = rows
            continue
        region = np.array(outline, dtype=np.float32).reshape(-1, 1, 2)
        out[frame] = [r for r in rows if cv2.pointPolygonTest(region, ((r["bbox"][0] + r["bbox"][2]) / 2, r["bbox"][3]), True) >= -height * .015]
    return out


def each_frame(rows_by_frame, step):
    return {frame: step(rows) for frame, rows in rows_by_frame.items()}


def with_roles(people_by_frame, football_by_frame):
    """Boxes from a general person detector, each given the kind the football detector saw there, if it saw one."""
    return {frame: adopt_roles(people, football_by_frame.get(frame, [])) for frame, people in people_by_frame.items()}


def variants(clip):
    runs = clip["runs"]
    football = lambda t, size=1280: rows_of(runs[f"football@{size}"], t)
    out = {}
    stored = {int(frame): [dict(bbox=b[:4], confidence=b[4], object_type=b[5], association=b[6]) for b in boxes] for frame, boxes in clip["stored_pipeline"].items()}
    out["stored pipeline (as shipped)"] = stored
    out["stored pipeline, confident boxes only"] = {f: [r for r in rows if r["confidence"] >= .45] for f, rows in stored.items()}
    for threshold in (.1, .25, .45, .6):
        out[f"football 1280 raw, conf {threshold}"] = football(threshold)
    out["football 1920 raw, conf 0.45"] = football(.45, 1920)
    out["football 1280 conf 0.45 + field + suppression"] = each_frame(on_field(clip, football(.45)), clean_people)
    out["football 1280 conf 0.25 + field + suppression"] = each_frame(on_field(clip, football(.25)), clean_people)
    out["football 1280 conf 0.25 + field + suppression + joined pieces"] = each_frame(on_field(clip, football(.25)), lambda rows: join_stacked_pieces(clean_people(rows)))
    for name in ("coco-yolov8m@1280", "coco-yolo11x@1280", "coco-yolo11x@1920"):
        if name not in runs:
            continue
        for threshold in (.25, .45):
            out[f"{name} raw, conf {threshold}"] = rows_of(runs[name], threshold)
        out[f"{name} conf 0.25 + field"] = on_field(clip, rows_of(runs[name], .25))
        out[f"{name} conf 0.25 + field + suppression"] = each_frame(on_field(clip, rows_of(runs[name], .25)), clean_people)
        out[f"{name} conf 0.25 + field + suppression + football roles"] = with_roles(each_frame(on_field(clip, rows_of(runs[name], .25)), clean_people), football(.25))
    return out


def labels_of(clip):
    return {int(frame): [dict(box=p[:4], role=p[4]) for p in people] for frame, people in clip["labels"].items()}


def score(clips):
    table = {}
    for clip in clips:
        labels = labels_of(clip)
        for name, rows in variants(clip).items():
            entry = table.setdefault(name, dict(players=[], people=[], per_clip={}))
            players, people = dd.diagnose(rows, labels, dd.PLAYERS), dd.diagnose(rows, labels, dd.PEOPLE)
            entry["players"].append(players)
            entry["people"].append(people)
            entry["per_clip"][clip["sequence"]] = dict(recall=players["recall"], precision=players["precision"])
    return {name: dict(players=dd.merge(entry["players"]), people=dd.merge(entry["people"]), per_clip=entry["per_clip"]) for name, entry in table.items()}


def f1(report):
    p, r = report["precision"] or 0, report["recall"] or 0
    return round(2 * p * r / (p + r), 3) if p + r else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--clips", default="")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    wanted = {name.strip() for name in args.clips.split(",") if name.strip()}
    clips = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.folder.glob("SNGS-*.json")) if not wanted or path.stem in wanted]
    if not clips:
        raise SystemExit("No saved detector output found")
    table = score(clips)
    print(f"{len(clips)} clips, {sum(len(c['frames']) for c in clips)} frames: {', '.join(c['sequence'] for c in clips)}")
    print(f"{'variant':<72} {'recall':>6} {'prec':>6} {'F1':>6} | {'split':>5} {'piece':>5} {'miss':>5} | {'ours:piece':>10} {'dup':>5} {'ref':>5} {'off':>5} || people R/P")
    for name, entry in table.items():
        r, people = entry["players"], entry["people"]
        lb, ob = r["labelled_bins"], r["our_bins"]
        print(f"{name:<72} {r['recall']:>6} {r['precision']:>6} {f1(r):>6} | {lb['split']:>5} {lb['piece_only']:>5} {lb['missed']:>5} | {ob['piece']:>10} {ob['duplicate']:>5} {ob['other_kind']:>5} {ob['off_label']:>5} || {people['recall']} / {people['precision']}")
    if args.json:
        args.json.write_text(json.dumps(dict(clips=[c["sequence"] for c in clips], ms_per_frame={c["sequence"]: c["ms_per_frame"] for c in clips}, variants=table), indent=1), encoding="utf-8")
        print("written", args.json)


if __name__ == "__main__":
    main()
