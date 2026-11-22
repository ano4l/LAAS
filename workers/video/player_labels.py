"""Names and shirt numbers a person gives to tracked identities, bound to one source video.

Nothing here is recognised automatically. A person watches the footage and says which
tracked identities are one player; that record is stored beside the run and never changes
the automatic analysis. Several identities may be given to one player, because the tracker
splits a player into several identities. Two identities that are on screen in the same
analysed frames cannot be one person, so that is refused.
"""
import json
import os
from collections import defaultdict
from pathlib import Path
import re
import tempfile

from workers.video.event_review import ReviewConflict, text

VERSION = "player-labels-2"  # 2 added lineup_id and team: a named player can point at a line-up entry
MAX_PLAYERS = 60
MAX_TRACKS_PER_PLAYER = 40
MAX_SHARED_FRAMES = 2  # a tracker can briefly hold two boxes on one person where identities hand over
PEOPLE = {"player", "goalkeeper"}


def _source(manifest):
    return manifest.get("run_id"), manifest.get("metadata", {}).get("sha256")


def known_tracks(manifest):
    """{identity id: kit group or None} for the people this run tracked."""
    summary = manifest.get("detection_summary") or {}
    if summary.get("linked_tracks"):
        return {t["linked_track_id"]: t.get("team") for t in summary["linked_tracks"] if t.get("object_type") in PEOPLE}
    return {t["track_id"]: None for t in summary.get("tracks", []) if t.get("object_type") in PEOPLE}


def load_labels(path, manifest):
    path = Path(path)
    run_id, digest = _source(manifest)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("run_id") != run_id or state.get("source_sha256") != digest:
            raise ValueError("Player names source mismatch")
        for player in state.get("players", []):  # names saved before line-ups existed point at none
            player.setdefault("lineup_id", None)
            player.setdefault("team", None)
        return state
    return dict(schema_version=VERSION, run_id=run_id, source_sha256=digest, revision=0, players=[], source="human_entry", overlap_checked=None)


def roster(paths):
    """Names a person has saved on any clip, for picking again on another: [{name, number, clips}], most used first.

    This only saves typing. A name from another clip is not matched to anyone automatically.
    """
    seen = {}
    for path in paths:
        try:
            players = json.loads(Path(path).read_text(encoding="utf-8")).get("players", [])
        except (OSError, ValueError):
            continue
        for player in players:
            if not isinstance(player, dict) or not isinstance(player.get("name"), str):
                continue
            key = (player["name"].casefold(), player.get("number"))
            entry = seen.setdefault(key, dict(name=player["name"], number=player.get("number"), clips=0))
            entry["clips"] += 1
    return sorted(seen.values(), key=lambda entry: (-entry["clips"], entry["name"].casefold(), entry["number"] or ""))[:300]


def _frames_by_track(tracking_path, wanted):
    """{identity id: set of analysed frame indexes it appears in}, or None when the detections are not stored."""
    tracking_path = Path(tracking_path) if tracking_path else None
    if not tracking_path or not tracking_path.is_file():
        return None
    frames = defaultdict(set)
    for row in json.loads(tracking_path.read_text(encoding="utf-8")):
        track = row.get("linked_track_id") or row.get("track_id")
        if track in wanted:
            frames[track].add(row.get("frame_idx"))
    return frames


def save_labels(path, manifest, payload, tracking_path=None, lineups=None):
    """`lineups` is the line-up saved under Review. A player may point at one of its entries, whose name and number are then used."""
    from workers.video.video_roster import lineup_index
    listed = lineup_index(lineups)
    current = load_labels(path, manifest)
    if type(payload.get("revision")) is not int or payload["revision"] != current["revision"]:
        raise ReviewConflict("Player names changed. Reload before saving; your edits are retained.")
    if payload.get("confirmed") is not True:
        raise ValueError("Confirm the names against the footage before saving")
    if manifest.get("engine") != "yolo_bytetrack":
        raise ValueError("Real video required")
    digest = manifest.get("metadata", {}).get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Source video hash required")
    raw = payload.get("players")
    if not isinstance(raw, list) or len(raw) > MAX_PLAYERS:
        raise ValueError("Invalid players")
    known, used, players, linked = known_tracks(manifest), set(), [], set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError("Invalid player")
        lineup_id, team = row.get("lineup_id"), None
        if lineup_id is not None:
            if not isinstance(lineup_id, str) or lineup_id not in listed:
                raise ValueError("That player is not in the line-up saved under Review. Save the line-up first.")
            if lineup_id in linked:
                raise ValueError(f"{listed[lineup_id][1]['name']} is chosen for two named players. Put the tracked identities under one.")
            linked.add(lineup_id)
            team, entry = listed[lineup_id]
            name, number = entry["name"], entry["number"]  # the line-up is the list of record
        else:
            name = text(row.get("name"), "player name", 80)
            number = row.get("number")
        if number is not None and (not isinstance(number, str) or not re.fullmatch(r"\d{1,2}", number)):
            raise ValueError("A shirt number is one or two digits, or left empty")
        tracks = row.get("tracks")
        if not isinstance(tracks, list) or not tracks or len(tracks) > MAX_TRACKS_PER_PLAYER or not all(isinstance(t, str) for t in tracks):
            raise ValueError("A named player needs at least one tracked identity")
        if len(set(tracks)) != len(tracks) or used & set(tracks):
            raise ValueError("A tracked identity can belong to one named player only")
        if not set(tracks) <= known.keys():
            raise ValueError("Tracked identity is not present in this source run")
        kits = {known[t] for t in tracks if known[t]}
        used |= set(tracks)
        players.append(dict(id=f"P{index + 1}", name=name, number=number, tracks=sorted(tracks), kits=sorted(kits), lineup_id=lineup_id, team=team))
    if len({(p["name"].casefold(), p["number"], p["team"]) for p in players}) != len(players):
        raise ValueError("Two players have the same name and number. Put their tracked identities under one player.")
    frames = _frames_by_track(tracking_path, used)
    if frames is not None:
        for player in players:
            for i, a in enumerate(player["tracks"]):
                for b in player["tracks"][i + 1:]:
                    shared = len(frames[a] & frames[b])
                    if shared > MAX_SHARED_FRAMES:
                        raise ValueError(f"{a} and {b} are on screen together in {shared} analysed frames, so they cannot both be {player['name']}")
    state = dict(current, schema_version=VERSION, revision=current["revision"] + 1, players=players, source="human_entry",
                 overlap_checked=frames is not None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(state, stream)
        temporary = stream.name
    try:
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return state
