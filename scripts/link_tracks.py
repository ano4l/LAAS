"""Link fragmented tracks of a stored run offline (CPU only, no neural inference).

Decodes the source video once to read kit colours inside the stored boxes, then joins
tracklets. With --evaluate it first cuts long stored tracks into pieces with artificial
gaps and reports how many cuts the linker repairs. That is a self-consistency check
against the online tracker's own IDs, not accuracy against human identity labels.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import random
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.video.camera_motion import CameraMotion, estimate_warps
from workers.video.track_linking import AppearanceBank, LinkSettings, PEOPLE, link_tracks, write_links


def load(source, run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["metadata"]["sha256"]:
            raise ValueError("Source video hash does not match the stored run")
    return manifest, json.loads((run / "tracking.json").read_text(encoding="utf-8"))


def appearance(source, rows, embedder=None):
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame_idx"]].append(row)
    bank, capture, index = AppearanceBank(embedder=embedder), cv2.VideoCapture(str(source)), 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in by_frame:
                bank.observe(frame, by_frame[index])
            index += 1
    finally:
        capture.release()
    return bank.descriptors(), bank.embeddings()


def fragment(rows, fps, seed=7, min_observations=60):
    """Cut long tracks into pieces separated by removed gaps of 0.2-2.5 seconds."""
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        if row.get("track_id") and row["object_type"] in PEOPLE:
            grouped[row["track_id"]].append(row)
    spans = {key: (items[0]["frame_idx"], items[-1]["frame_idx"]) for key, items in grouped.items()}
    removed, rename, cuts = set(), {}, []
    for key, items in grouped.items():
        if len(items) < min_observations:
            continue
        position, piece = rng.randint(20, 40), 0
        while position < len(items) - 20:
            gap = max(1, round(rng.uniform(.2, 2.5) * fps))
            if position + gap >= len(items) - 10:
                break
            removed.update(id(r) for r in items[position:position + gap])
            for r in items[position + gap:]:
                rename[id(r)] = f"{key}#{piece + 1}"
            cuts.append((f"{key}#{piece}" if piece else key, f"{key}#{piece + 1}"))
            piece += 1
            position += gap + rng.randint(20, 60)
    out = []
    for row in rows:
        if id(row) in removed:
            continue
        out.append(dict(row, track_id=rename.get(id(row), row.get("track_id"))))
    return out, cuts, spans


def evaluate(source, rows, fps, settings, embedder=None, camera=None):
    cut_rows, cuts, spans = fragment(rows, fps)
    descriptors, embeddings = appearance(source, cut_rows, embedder)
    result = link_tracks(cut_rows, descriptors, fps, settings, embeddings, camera)
    truth = set(cuts)
    repaired = skipped = wrong = unknown = 0
    for link in result["links"]:
        pair = (link["from_track"], link["to_track"])
        a, b = (name.split("#")[0] for name in pair)
        if pair in truth:
            repaired += 1
        elif a == b:
            skipped += 1  # same stored track, but an intermediate piece was left out
        elif spans[a][0] <= spans[b][1] and spans[b][0] <= spans[a][1]:
            wrong += 1  # joined two stored tracks that were on screen at the same time
        else:
            unknown += 1  # different stored tracks that never coexisted: may be a genuine repair
    return dict(artificial_cuts=len(cuts), repaired=repaired, same_track_piece_skipped=skipped,
                definitely_wrong=wrong, unverifiable=unknown,
                repaired_share=round(repaired / len(cuts), 4) if cuts else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, help="Directory for track-links.json (default: print summary only)")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--max-gap", type=float, default=LinkSettings.max_gap_seconds)
    p.add_argument("--margin", type=float, default=LinkSettings.ambiguity_margin)
    p.add_argument("--embedding-weights", type=Path, help="Detector weights whose backbone supplies appearance embeddings")
    p.add_argument("--osnet", type=Path, nargs=2, metavar=("WEIGHTS", "OSNET_PY"), help="OSNet ReID checkpoint and torchreid osnet.py")
    p.add_argument("--image-camera", action="store_true", help="Use image-feature camera motion instead of the default player-movement estimate")
    p.add_argument("--min-observations", type=int, default=LinkSettings.min_observations)
    args = p.parse_args()
    settings = LinkSettings(max_gap_seconds=args.max_gap, ambiguity_margin=args.margin, min_observations=args.min_observations)
    manifest, rows = load(args.source, args.run)
    fps = manifest["metadata"]["fps"]  # gaps are measured in source frames
    embedder = None
    if args.osnet:
        from workers.video.reid import OSNetEmbedder
        embedder = OSNetEmbedder(*args.osnet)
    elif args.embedding_weights:
        from workers.video.reid import BackboneEmbedder
        embedder = BackboneEmbedder(args.embedding_weights)
    if embedder:
        settings = replace(settings, **embedder.link_settings)
    descriptors, embeddings = appearance(args.source, rows, embedder)
    # Default: the pipeline's player-movement estimate (measured better for linking than image features).
    camera = CameraMotion.from_warps(estimate_warps(args.source)) if args.image_camera else None
    result = link_tracks(rows, descriptors, fps, settings, embeddings, camera)
    report = dict(source=args.source.name, run_id=manifest["run_id"],
                  **{k: result["summary"][k] for k in ("tracks_before", "identities_after", "links_accepted", "tracklet_ends_left_ambiguous")})
    if args.evaluate:
        report["cut_repair_check"] = evaluate(args.source, rows, fps, settings, embedder, camera)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_links(args.output, result)
    print(json.dumps(report, indent=2))
