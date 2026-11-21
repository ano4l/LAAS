"""One list of the people in a video, joining the three places a person can name them.

- The line-up under Review: the team sheet, typed by a person.
- Named tracked identities under Players: which boxes in the video are which person.
- The player typed on an approved event.

The line-up is the list of record. A named tracked player may point at a line-up entry, and an
event's player is matched to the line-up by name. Nothing here recognises anyone: every link was
made by a person. A name that is not in the line-up is listed apart, never merged by guesswork.
"""
from collections import Counter, defaultdict

VERSION = "video-roster-1"
SIDES = ("home", "away")


def lineup_index(lineups):
    """{line-up id: (team side, line-up entry)}."""
    return {entry["id"]: (side, entry) for side in SIDES for entry in (lineups or {}).get(side, [])}


def refresh_labels(labels, lineups):
    """Named tracked players with the name, number and team of the line-up entry they point at.

    The line-up is the list of record, so a name corrected there shows here without saving again.
    A link to an entry that has since been removed keeps its stored name and is marked.
    """
    index = lineup_index(lineups)
    players = []
    for player in labels.get("players", []):
        linked = index.get(player.get("lineup_id"))
        if linked:
            side, entry = linked
            player = dict(player, name=entry["name"], number=entry["number"], team=side, lineup_missing=False)
        else:
            player = dict(player, lineup_id=player.get("lineup_id"), team=player.get("team"), lineup_missing=bool(player.get("lineup_id")))
        players.append(player)
    return dict(labels, players=players)


def video_roster(review, labels):
    """Every person a reviewer has named for this video, with what is known about each from all three places."""
    lineups, summary = review.get("lineups") or {}, review["summary"]
    labels = refresh_labels(labels, lineups)
    tracked = {player["lineup_id"]: player for player in labels["players"] if player.get("lineup_id") and not player["lineup_missing"]}
    events = {row["lineup_id"]: row for row in summary["players"] if row.get("lineup_id")}

    def stats(row):
        return dict(event_count=row["event_count"] if row else 0, counts=row["counts"] if row else {},
                    outcomes=row.get("outcomes", {}) if row else {}, per_90=row.get("per_90") if row else None)

    players = []
    for side in SIDES:
        for entry in lineups.get(side, []):
            label = tracked.get(entry["id"])
            players.append(dict(source="lineup", lineup_id=entry["id"], label_id=label["id"] if label else None, team=side,
                name=entry["name"], number=entry["number"], position=entry["position"], started=entry["started"], minutes=entry["minutes"],
                tracks=label["tracks"] if label else [], kits=label["kits"] if label else [], **stats(events.get(entry["id"]))))
    # A tracked identity given a name that is not in the line-up, or whose line-up entry was removed.
    for label in labels["players"]:
        if not label.get("lineup_id") or label["lineup_missing"]:
            players.append(dict(source="tracked_only", lineup_id=None, label_id=label["id"], team=None, name=label["name"], number=label["number"],
                position=None, started=None, minutes=None, tracks=label["tracks"], kits=label["kits"], lineup_missing=label["lineup_missing"], **stats(None)))
    # A name typed on an approved event that is not in the line-up.
    for row in summary["players"]:
        if not row.get("listed") and row["event_count"]:
            players.append(dict(source="event_only", lineup_id=None, label_id=None, team=row["team"], name=row["player"], number=None,
                position=None, started=None, minutes=None, tracks=[], kits=[], **stats(row)))

    # Which team each kit group seems to be, from named players a person tied to both. Evidence for a person, not a decision.
    votes = defaultdict(Counter)
    for player in players:
        if player["source"] == "lineup":
            for kit in player["kits"]:
                votes[kit][player["team"]] += 1
    count = lambda source: sum(1 for player in players if player["source"] == source)
    return dict(schema_version=VERSION, run_id=review["run_id"], teams=review["teams"], players=players,
        counts=dict(listed=count("lineup"), listed_with_tracks=sum(1 for p in players if p["source"] == "lineup" and p["tracks"]),
                    tracked_only=count("tracked_only"), event_only=count("event_only")),
        kit_team_votes={kit: dict(home=sides["home"], away=sides["away"]) for kit, sides in sorted(votes.items())},
        per_90=summary.get("per_90"), review_revision=review["revision"], labels_revision=labels["revision"],
        note="Every link here was made by a person: a line-up typed under Review, tracked identities named under Players, and players typed on approved events. Nothing is recognised automatically.")
