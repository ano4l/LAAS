"""Offline linking of fragmented person tracklets into longer temporary identities.

The online tracker ends a track whenever a player is occluded or missed for about a
second, so one player becomes many short track IDs. After the whole clip is known,
this stage joins a tracklet's end to another tracklet's start when timing, position,
size, kit colour and class all agree and no competing candidate is nearly as good.

Linked identities are still anonymous and unreviewed. Colour histograms cannot tell
team-mates apart, so links rely on short gaps and position; a player who leaves the
view and returns later is deliberately left as separate identities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np

from workers.video.camera_motion import CameraMotion

PEOPLE = {"person", "player", "goalkeeper", "referee"}
VERSION = "track-linking-1"


@dataclass(frozen=True)
class LinkSettings:
    max_gap_seconds: float = 3.0
    base_reach_heights: float = 0.4       # allowed miss at zero gap, in player heights
    reach_heights_per_second: float = 1.0  # extra allowed distance per second of gap
    max_height_ratio: float = 1.6
    max_appearance_distance: float = 0.45
    appearance_weight: float = 1.0
    max_embedding_distance: float = 0.9    # cosine distance, 0-2; depends on the embedding model
    centre_embeddings: bool = True         # subtract the clip mean first (needed for detector features)
    embedding_weight: float = 0.5
    ambiguity_margin: float = 0.35         # best link must beat any rival by this much
    velocity_window: int = 5
    min_observations: int = 5              # shorter tracklets are never linked
    team_min_votes: int = 3
    team_min_share: float = 0.7


def _descriptor(frame, box):
    """Kit-colour histogram for shirt and shorts; None when too little non-grass evidence."""
    height, width = frame.shape[:2]
    a, b, c, d = box
    w, h = c - a, d - b
    x1, x2 = max(0, int(a + .2 * w)), min(width, int(c - .2 * w))
    parts = []
    for top, bottom in ((.18, .52), (.52, .8)):
        y1, y2 = max(0, int(b + top * h)), min(height, int(b + bottom * h))
        crop = frame[y1:y2, x1:x2]
        if crop.size < 30:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
        grass = (hsv[:, 0] >= 28) & (hsv[:, 0] <= 95) & (hsv[:, 1] > 45)
        hsv = hsv[~grass]
        if len(hsv) < 8:
            return None
        coloured = hsv[:, 1] > 45
        hist = np.zeros(40)
        # 36 hue/saturation bins for coloured pixels, 4 brightness bins for white/grey/black.
        np.add.at(hist, (hsv[coloured, 0] * 12 // 180) * 3 + np.minimum(2, (hsv[coloured, 1] - 46) // 70), 1)
        np.add.at(hist, 36 + np.minimum(3, hsv[~coloured, 2] // 64), 1)
        parts.append(hist / hist.sum())
    return np.concatenate(parts)


def appearance_distance(a, b):
    """0 for identical kit histograms, 1 for no overlap (mean Bhattacharyya distance of shirt and shorts)."""
    half = len(a) // 2
    overlap = [float(np.sqrt(a[s] * b[s]).sum()) for s in (slice(0, half), slice(half, None))]
    return 1 - sum(overlap) / 2


class AppearanceBank:
    """Accumulates a mean kit-colour descriptor, and optionally a mean learned embedding, per track."""

    def __init__(self, every=5, embedder=None):
        self.every, self.embedder = every, embedder
        self.seen = Counter()
        self.sums, self.counts = {}, Counter()
        self.embedding_sums, self.embedding_counts = {}, Counter()

    def observe(self, frame, rows):
        people = [r for r in rows if r.get("track_id") and r["object_type"] in PEOPLE]
        crops, owners = [], []
        for row in people:
            identity = row["track_id"]
            self.seen[identity] += 1
            if (self.seen[identity] - 1) % self.every:
                continue
            if any(other is not row and _iou(row["bbox"], other["bbox"]) > .2 for other in people):
                continue  # overlapping players contaminate each other's appearance
            feature = _descriptor(frame, row["bbox"])
            if feature is not None:
                self.sums[identity] = self.sums.get(identity, 0) + feature
                self.counts[identity] += 1
            if self.embedder:
                x1, y1, x2, y2 = (max(0, int(v)) for v in row["bbox"])
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] >= 16 and crop.shape[1] >= 8:
                    crops.append(crop)
                    owners.append(identity)
        if crops:
            for identity, vector in zip(owners, self.embedder.embed(crops)):
                self.embedding_sums[identity] = self.embedding_sums.get(identity, 0) + vector
                self.embedding_counts[identity] += 1

    def descriptors(self):
        return {key: self.sums[key] / self.counts[key] for key in self.sums}

    def embeddings(self):
        """Unit-length mean embedding per track."""
        return {key: value / max(1e-9, float(np.linalg.norm(value))) for key, value in self.embedding_sums.items()}


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _foot(box):
    return (box[0] + box[2]) / 2, box[3]


def camera_offsets(rows):
    """Cumulative image shift per analysed frame, from the median movement of tracked people.

    Players run in different directions, so their median frame-to-frame displacement
    approximates a camera pan. Zoom is not modelled.
    """
    by_frame = defaultdict(dict)
    for row in rows:
        if row.get("track_id") and row["object_type"] in PEOPLE:
            by_frame[row["frame_idx"]][row["track_id"]] = _foot(row["bbox"])
    offsets, total, previous = {}, np.zeros(2), None
    for frame in sorted(by_frame):
        if previous is not None:
            shared = by_frame[frame].keys() & by_frame[previous].keys()
            if len(shared) >= 4:
                total = total + np.median([np.subtract(by_frame[frame][k], by_frame[previous][k]) for k in shared], axis=0)
        offsets[frame] = total
        previous = frame
    return offsets


def _tracklets(rows, settings, camera):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("track_id") and row["object_type"] in PEOPLE:
            grouped[row["track_id"]].append(row)
    tracklets = {}
    for identity, items in grouped.items():
        items.sort(key=lambda r: r["frame_idx"])
        # Foot point and box height in stabilised coordinates, so pans and zooms do not look like player movement.
        points = [(r["frame_idx"], camera.stabilise(r["frame_idx"], _foot(r["bbox"])),
                   (r["bbox"][3] - r["bbox"][1]) * camera.scale(r["frame_idx"])) for r in items]
        head, tail = points[:settings.velocity_window], points[-settings.velocity_window:]
        classes = Counter(r["object_type"] for r in items)
        teams = Counter(r["team"] for r in items if r.get("team"))
        tracklets[identity] = dict(
            track_id=identity, first=points[0][0], last=points[-1][0], observations=len(items),
            scene=items[0].get("scene_id", 0), classes=classes, teams=teams,
            start=points[0][1], end=points[-1][1],
            start_height=float(np.median([p[2] for p in head])), end_height=float(np.median([p[2] for p in tail])),
            velocity=(tail[-1][1] - tail[0][1]) / (tail[-1][0] - tail[0][0]) if len(tail) >= 3 else np.zeros(2))
    return tracklets


def _strong(counter, share=.7):
    if not counter:
        return None
    label, votes = counter.most_common(1)[0]
    return label if votes / sum(counter.values()) >= share else None


def _link_evidence(a, b, descriptors, fps, settings, embeddings=None):
    """Evidence for joining tracklet a's end to b's start, or None if a hard rule forbids it."""
    gap = (b["first"] - a["last"]) / fps
    if gap <= 0 or gap > settings.max_gap_seconds or a["scene"] != b["scene"]:
        return None
    if min(a["observations"], b["observations"]) < settings.min_observations:
        return None
    team_a, team_b = _strong(a["teams"]), _strong(b["teams"])
    if team_a and team_b and team_a != team_b:
        return None
    if ("referee" in (_strong(a["classes"]), _strong(b["classes"]))) and _strong(a["classes"]) != _strong(b["classes"]):
        return None
    height = (a["end_height"] + b["start_height"]) / 2
    if height <= 0 or max(a["end_height"], b["start_height"]) / max(1e-6, min(a["end_height"], b["start_height"])) > settings.max_height_ratio:
        return None
    # Velocity is only trusted for one second; beyond that the player may have turned.
    predicted = a["end"] + a["velocity"] * min(gap, 1.0) * fps
    distance = float(np.linalg.norm(predicted - b["start"])) / height
    reach = settings.base_reach_heights + settings.reach_heights_per_second * gap
    if distance > reach:
        return None
    appearance = None
    if a["track_id"] in descriptors and b["track_id"] in descriptors:
        appearance = appearance_distance(descriptors[a["track_id"]], descriptors[b["track_id"]])
        if appearance > settings.max_appearance_distance:
            return None
    cost = distance / reach + settings.appearance_weight * (appearance if appearance is not None else settings.max_appearance_distance)
    embedding = None
    if embeddings:
        if a["track_id"] in embeddings and b["track_id"] in embeddings:
            embedding = 1 - float(embeddings[a["track_id"]] @ embeddings[b["track_id"]])
            if embedding > settings.max_embedding_distance:
                return None
        cost += settings.embedding_weight * (embedding if embedding is not None else settings.max_embedding_distance)
    return dict(from_track=a["track_id"], to_track=b["track_id"], gap_seconds=round(gap, 3),
                distance_heights=round(distance, 3), appearance_distance=None if appearance is None else round(appearance, 4),
                embedding_distance=None if embedding is None else round(embedding, 4), cost=round(cost, 4))


def link_tracks(rows, descriptors, fps, settings=LinkSettings(), embeddings=None, camera=None):
    """Return linked identities, accepted links with their evidence, and a summary.

    `embeddings` are optional unit-length learned appearance vectors per track. Detector
    backbone features are centred on the clip mean first, because they are dominated by
    a component shared by every player; each embedder's `link_settings` says what it needs.
    """
    if embeddings and settings.centre_embeddings:
        mean = np.mean(list(embeddings.values()), axis=0)
        embeddings = {key: (value - mean) / max(1e-9, float(np.linalg.norm(value - mean))) for key, value in embeddings.items()}
    camera = camera or CameraMotion.from_people(rows)
    tracklets = _tracklets(rows, settings, camera)
    order = sorted(tracklets.values(), key=lambda t: (t["first"], t["track_id"]))
    index = {t["track_id"]: i for i, t in enumerate(order)}
    starts_sorted = [t["first"] for t in order]
    candidates = {}
    for a in order:
        # Only tracklets starting inside the gap window can follow a.
        low = np.searchsorted(starts_sorted, a["last"], side="right")
        high = np.searchsorted(starts_sorted, a["last"] + settings.max_gap_seconds * fps, side="right")
        for b in order[low:high]:
            evidence = _link_evidence(a, b, descriptors, fps, settings, embeddings)
            if evidence:
                candidates[index[a["track_id"]], index[b["track_id"]]] = evidence
    # Accept a link only when it beats every rival for its tracklet end and its tracklet
    # start by a clear margin. A rival is an alternative that overlaps in time with the
    # chosen partner; a later non-overlapping fragment may be the same player further
    # along the chain, so it does not block. Each accepted link removes its rivals,
    # which can make further links clear, so repeat until nothing changes.
    def overlap(p, q):
        return order[p]["first"] <= order[q]["last"] and order[q]["first"] <= order[p]["last"]

    remaining, accepted = dict(candidates), []
    while remaining:
        by_end, by_start = defaultdict(list), defaultdict(list)
        for (i, j), evidence in remaining.items():
            by_end[i].append((j, evidence["cost"]))
            by_start[j].append((i, evidence["cost"]))
        clear = []
        for (i, j), evidence in remaining.items():
            rivals = [c for q, c in by_end[i] if q != j and overlap(q, j)] + [c for p, c in by_start[j] if p != i and overlap(p, i)]
            runner_up = min(rivals) if rivals else None
            if runner_up is None or runner_up - evidence["cost"] >= settings.ambiguity_margin:
                clear.append((evidence["cost"], evidence["gap_seconds"], i, j, dict(evidence, runner_up_cost=runner_up)))
        ends, starts = set(), set()
        for _, _, i, j, link in sorted(clear, key=lambda c: c[:4]):  # cheapest, then shortest gap, first
            if i not in ends and j not in starts:
                ends.add(i)
                starts.add(j)
                accepted.append(link)
        if not ends:
            break
        remaining = {(i, j): e for (i, j), e in remaining.items() if i not in ends and j not in starts}
    ambiguous = len({i for i, _ in remaining})
    parent = {key: key for key in tracklets}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for link in accepted:
        parent[find(link["to_track"])] = find(link["from_track"])
    groups = defaultdict(list)
    for t in order:
        groups[find(t["track_id"])].append(t)
    identities, mapping = [], {}
    for number, members in enumerate(sorted(groups.values(), key=lambda m: (m[0]["first"], m[0]["track_id"])), 1):
        linked_id = f"L{number}"
        teams = sum((m["teams"] for m in members), Counter())
        classes = sum((m["classes"] for m in members), Counter())
        team = None
        # An identity seen mostly as a referee gets no kit group, even if a few of its
        # detections were classed as a player and picked up kit votes.
        if sum(teams.values()) >= settings.team_min_votes and classes.most_common(1)[0][0] != "referee":
            team = _strong(teams, settings.team_min_share)
        for m in members:
            mapping[m["track_id"]] = dict(linked_track_id=linked_id, linked_team=team)
        identities.append(dict(linked_track_id=linked_id, track_ids=[m["track_id"] for m in members],
                               object_type=classes.most_common(1)[0][0], team=team, team_votes=dict(teams),
                               first_frame=members[0]["first"], last_frame=members[-1]["last"],
                               first_seen_ms=round(members[0]["first"] / fps * 1000), last_seen_ms=round(members[-1]["last"] / fps * 1000),
                               observations=sum(m["observations"] for m in members)))
    summary = dict(version=VERSION, tracks_before=len(tracklets), identities_after=len(identities),
                   links_accepted=len(accepted), tracklet_ends_left_ambiguous=ambiguous,
                   candidate_links=len(candidates), camera_motion=camera.source, tracks_with_appearance=len(descriptors.keys() & tracklets.keys()),
                   tracks_with_embedding=len((embeddings or {}).keys() & tracklets.keys()),
                   settings=asdict(settings),
                   limitations=["Linked identities are anonymous, unreviewed and not validated against labelled identities.",
                                "Kit colours cannot separate team-mates; links depend on gaps of at most "
                                f"{settings.max_gap_seconds:g} seconds and image position.",
                                "Players who leave the view and return later remain separate identities.",
                                "Identity switches made by the online tracker inside one track are not split."])
    return dict(summary=summary, identities=identities, links=sorted(accepted, key=lambda l: l["cost"]), mapping=mapping)


def linked_view(rows):
    """Rows as analytics should see them: linked identities in place of raw tracker IDs.

    Returns (rows, identity_source). Runs without linking are returned unchanged. The
    raw tracker ID is kept on each row as source_track_id, and a team voted across all
    fragments of an identity replaces a missing or per-fragment team label.
    """
    if not any("linked_track_id" in row for row in rows):
        return rows, "track_id"
    view = []
    for row in rows:
        if row.get("linked_track_id"):
            row = dict(row, track_id=row["linked_track_id"], source_track_id=row.get("track_id"),
                       team=row.get("linked_team") or row.get("team"))
        view.append(row)
    return view, "linked_track_id"


def apply_links(tracking_path: Path, result: dict) -> None:
    """Add linked_track_id/linked_team to each row of a one-row-per-line tracking.json; track_id is kept."""
    temporary = tracking_path.with_suffix(".linking.tmp")
    with tracking_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            text = line.rstrip("\n")
            body = text.rstrip(",")
            if not body.startswith("{"):
                target.write(line)
                continue
            row = json.loads(body)
            row.update(result["mapping"].get(row.get("track_id"), dict(linked_track_id=None, linked_team=None)))
            target.write(json.dumps(row) + text[len(body):] + "\n")
    temporary.replace(tracking_path)


def write_rows(tracking_path: Path, rows) -> None:
    """Write tracking rows in the one-row-per-line layout the pipeline streams."""
    temporary = tracking_path.with_suffix(".rows.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        target.write("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]")
    temporary.replace(tracking_path)


def write_links(directory: Path, result: dict) -> None:
    payload = {key: result[key] for key in ("summary", "identities", "links")}
    (directory / "track-links.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
