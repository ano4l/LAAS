"""Read-only hosted smoke check; no upload or inference is submitted."""
import argparse
import json
import math
from pathlib import Path
import urllib.request


def verify(api, key):
    def get(path, authenticated=True, byte_range=False):
        url = path if path.startswith("https://") else api+path
        headers = {"Authorization": "Bearer "+key} if authenticated else {}
        if byte_range:
            headers["Range"] = "bytes=0-1023"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            return response.status, response.read()
    _, listing = get("/library")
    listing = json.loads(listing)
    checks = []
    targets = [r for r in listing["runs"] if r["metadata"]["filename"].endswith("-fresh.mp4")]
    assert len(targets)==5
    for target in targets:
        status, body = get("/runs/"+target["run_id"])
        result = json.loads(body)
        data = result["metrics"]["video_insights"]
        assert data["schema_version"] == "video-insights-5"
        assert data['collective_dynamics']['schema_version']=='soccertrack-dynamics-1'
        assert data['collective_dynamics']['formation']['label'] is None
        dynamics=data['collective_dynamics']
        assert dynamics['coordinate_system']=='camera_image'
        for group in dynamics['teams']:
            assert 0<=group['source_coverage_pct']<=100
            for window in group['timeline']:
                shape=window['snapshot']['ellipse']
                assert shape['point_count']>=3 and all(math.isfinite(shape[k]) for k in ['major_radius_px','minor_radius_px','angle_degrees','area_image_pct'])
                assert shape['major_radius_px']>=shape['minor_radius_px']>=0
                assert window['motion_alignment'] is None or 0<=window['motion_alignment']<=1
        assert data["analyzed_frames"] == result["detection_summary"]["analyzed_frames"]
        assert data["source_duration_seconds"] == result["metadata"]["duration_seconds"]
        assert data["source_video"]["sha256"] == result["metadata"]["sha256"]
        assert abs(data["control"]["unknown_seconds"]+data["control"]["estimated_control_seconds"]-data["source_duration_seconds"]) < .002
        media_status, _ = get(result["files"]["annotated_video"], False, True)
        metrics_status, downloaded = get(result["files"]["metrics_json"], False)
        assert json.loads(downloaded)["video_insights"] == data
        assert media_status == 206
        _, reviewed = get("/runs/"+target["run_id"]+"/review")
        review = json.loads(reviewed)
        _, context_body = get('/runs/'+target['run_id']+'/sequence-context')
        context=json.loads(context_body);sequences=data['sequences']
        assert context['source_sha256']==result['metadata']['sha256'] and context['run_id']==target['run_id']
        assert sequences['context_revision']==context['revision']
        assert abs(sum(sequences[k] for k in ['known_seconds','unknown_seconds','contested_seconds','dead_ball_seconds'])-data['source_duration_seconds'])<.002
        checks.append(dict(run_id=target["run_id"], filename=target["metadata"]["filename"], status=status,
            duration_seconds=data["source_duration_seconds"], analyzed_frames=data["analyzed_frames"],
            source_hash_matches=True, control_seconds=data["control"]["estimated_control_seconds"], unknown_seconds=data["control"]["unknown_seconds"],
            candidates=len(data["candidates"]), shape_windows=sum(len(k["shape_timeline"]) for k in data["kits"]),
            ball_points=data["ball"]["path_observations"], signed_video_status=media_status, signed_metrics_status=metrics_status,
            review_revision=review["revision"], approved_events=review["summary"]["approved_events"],
            context_revision=context['revision'],temporal_known_seconds=sequences['known_seconds'],
            temporal_unknown_seconds=sequences['unknown_seconds'],review_only_candidates=sequences['diagnostics']['review_only_candidates'],
            collective_groups=len(dynamics['teams']),collective_windows=sum(len(g['timeline']) for g in dynamics['teams']),
            activity_ellipses=sum(len(g['activity_ellipses']) for g in dynamics['teams']),named_formation=None))
    _, after = get("/library")
    assert {j["id"] for j in listing["jobs"]} == {j["id"] for j in json.loads(after)["jobs"]}
    _, sample = get("/sample")
    sample = json.loads(sample)["result"]
    assert sample["metadata"]["duration_seconds"] == 30
    assert sample["metrics"]["video_insights"]["analyzed_frames"] == 750
    return dict(api=api, reads_only=True, job_ids_unchanged=True, runs=checks,
                sample=dict(duration_seconds=30, analyzed_frames=750, schema=sample["metrics"]["video_insights"]["schema_version"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api",default="https://ano4l--laas-football-web-web.us-east.modal.run")
    parser.add_argument("--key-file",type=Path,default=Path("data/qa/web-access-key.txt"))
    parser.add_argument("--output",type=Path,default=Path("data/qa/video-insights-2026-10-25/hosted-verification.json"))
    args=parser.parse_args()
    result=verify(args.api,args.key_file.read_text(encoding="utf-8").strip())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
