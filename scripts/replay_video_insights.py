"""Recalculate video insights from full saved tracking; never calls inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.video.insights import from_tracking
from scripts.export_video_analytics import export


def replay(root):
    summaries = []
    for path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("engine") != "yolo_bytetrack":
            continue
        rows = json.loads((path.parent/"tracking.json").read_text(encoding="utf-8"))
        from workers.video.sequence_context import load_context
        context=load_context(path.parent/'sequence-context.json',manifest)
        insights = from_tracking(rows, manifest["metadata"], manifest["detection_summary"],context=context)
        manifest["metrics"]["video_insights"] = insights
        path.write_text(json.dumps(manifest), encoding="utf-8")
        (path.parent/"metrics.json").write_text(json.dumps(manifest["metrics"]), encoding="utf-8")
        export(path.parent)
        summaries.append(dict(run_id=manifest["run_id"], filename=manifest["metadata"]["filename"],
            duration_seconds=insights["source_duration_seconds"], analyzed_frames=insights["analyzed_frames"],
            selected_ball_coverage_pct=insights["quality"]["selected_ball_coverage_pct"],
            kit_assignment_coverage_pct=insights["quality"]["kit_assignment_coverage_pct"],
            estimated_control_seconds=insights["control"]["estimated_control_seconds"],
            unknown_control_seconds=insights["control"]["unknown_seconds"],
            control_spells=len(insights["control"]["spells"]), candidate_events=len(insights["candidates"]),
            kit_shape_windows=sum(len(g["shape_timeline"]) for g in insights["kits"]),
            ball_path_observations=insights["ball"]["path_observations"]))
    return dict(provenance="CPU recalculation from full stored real tracking observations; no fresh detector inference or accuracy labels.", runs=summaries)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--summary", type=Path)
    args=parser.parse_args()
    result=replay(args.root)
    if args.summary:
        args.summary.parent.mkdir(parents=True,exist_ok=True)
        args.summary.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
