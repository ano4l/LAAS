"""One roster per video: the line-up, named tracked identities and players typed on events, joined only by links a person made."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api.modal_gateway import create_gateway
from workers.video.event_review import load_review, save_review
from workers.video.player_labels import load_labels, save_labels
from workers.video.video_roster import refresh_labels, video_roster


def manifest():
    linked = [dict(linked_track_id=f"L{n}", track_ids=[f"S0:{n}"], object_type="player", team=team, observations=10, first_seen_ms=0, last_seen_ms=900)
              for n, team in ((1, "kit-A"), (2, "kit-A"), (3, "kit-B"), (4, None))]
    return dict(run_id="7" * 32, engine="yolo_bytetrack", metadata=dict(sha256="b" * 64, duration_seconds=60), detection_summary=dict(linked_tracks=linked))


def event(identity, player, kind="pass", outcome="completed", team="home"):
    return dict(id=identity, type=kind, outcome=outcome, review_status="approved", team=team, timestamp_seconds=5, player=player, notes="")


def review_body(revision=0, **extra):
    lineups = dict(home=[dict(id="h8", name="Ada Obi", number="8", position="MF", started=True, minutes=1),
                         dict(id="h9", name="Bo Lee", number="9")], away=[dict(id="a4", name="Cy Mo", number="4")])
    return dict(dict(revision=revision, teams=dict(home="Reds", away="Blues"), coverage=[], possession_intervals=[], lineups=lineups,
                     events=[event("e1", "Ada Obi"), event("e2", "ada obi", "shot", "goal"), event("e3", "Guest", team="away")]), **extra)


def names(players, revision=0):
    return dict(revision=revision, confirmed=True, players=players)


def saved(tmp_path):
    m = manifest()
    review_path, labels_path = tmp_path / "review.json", tmp_path / "player-labels.json"
    review = save_review(review_path, m, review_body())
    labels = save_labels(labels_path, m, names([dict(lineup_id="h8", tracks=["L1", "L2"]), dict(lineup_id="a4", tracks=["L3"]),
                                               dict(name="Unknown sub", number=None, tracks=["L4"])]), lineups=review["lineups"])
    return m, review_path, labels_path, review, labels


def test_roster_joins_lineup_tracked_identities_and_event_names(tmp_path):
    m, _, _, review, labels = saved(tmp_path)
    # A linked player takes the line-up's name, number and team; a typed name is not needed.
    assert labels["players"][0] == dict(id="P1", name="Ada Obi", number="8", tracks=["L1", "L2"], kits=["kit-A"], lineup_id="h8", team="home")
    roster = video_roster(review, labels)
    by_name = {player["name"]: player for player in roster["players"]}
    ada = by_name["Ada Obi"]
    assert (ada["source"], ada["team"], ada["tracks"], ada["kits"], ada["position"]) == ("lineup", "home", ["L1", "L2"], ["kit-A"], "MF")
    assert ada["event_count"] == 2 and ada["counts"] == dict(**{"pass": 1}, shot=1)   # both spellings are the one listed player
    assert by_name["Bo Lee"]["tracks"] == [] and by_name["Bo Lee"]["event_count"] == 0      # listed, never tracked or tagged
    assert (by_name["Unknown sub"]["source"], by_name["Unknown sub"]["team"], by_name["Unknown sub"]["tracks"]) == ("tracked_only", None, ["L4"])
    assert (by_name["Guest"]["source"], by_name["Guest"]["team"], by_name["Guest"]["event_count"]) == ("event_only", "away", 1)
    assert roster["counts"] == dict(listed=3, listed_with_tracks=2, tracked_only=1, event_only=1)
    # Named players tied to both a kit group and a team are evidence of which kit is which team.
    assert roster["kit_team_votes"] == {"kit-A": dict(home=1, away=0), "kit-B": dict(home=0, away=1)}
    assert roster["teams"] == dict(home="Reds", away="Blues") and roster["review_revision"] == 1 and roster["labels_revision"] == 1


def test_lineup_is_the_list_of_record_and_a_removed_entry_leaves_a_marked_link(tmp_path):
    m, review_path, labels_path, _, _ = saved(tmp_path)
    changed = review_body(revision=1)
    changed["lineups"]["home"][0].update(name="Adaeze Obi", number="18")
    changed["lineups"]["away"] = []
    changed["events"] = []
    review = save_review(review_path, m, changed)
    labels = refresh_labels(load_labels(labels_path, m), review["lineups"])
    assert (labels["players"][0]["name"], labels["players"][0]["number"], labels["players"][0]["lineup_missing"]) == ("Adaeze Obi", "18", False)
    # The away player was removed from the line-up: the name saved with the link is kept, and the link is marked as broken.
    assert (labels["players"][1]["name"], labels["players"][1]["lineup_missing"]) == ("Cy Mo", True)
    roster = video_roster(review, load_labels(labels_path, m))
    cy = next(player for player in roster["players"] if player["name"] == "Cy Mo")
    assert (cy["source"], cy["lineup_missing"], cy["tracks"]) == ("tracked_only", True, ["L3"])
    assert roster["kit_team_votes"] == {"kit-A": dict(home=1, away=0)}


@pytest.mark.parametrize("players, message", [
    ([dict(lineup_id="nobody", tracks=["L1"])], "not in the line-up"),
    ([dict(lineup_id=7, tracks=["L1"])], "not in the line-up"),
    ([dict(lineup_id="h8", tracks=["L1"]), dict(lineup_id="h8", tracks=["L2"])], "chosen for two named players"),
])
def test_invalid_lineup_links_are_refused(tmp_path, players, message):
    m = manifest()
    review = save_review(tmp_path / "review.json", m, review_body())
    with pytest.raises(ValueError, match=message):
        save_labels(tmp_path / "player-labels.json", m, names(players), lineups=review["lineups"])
    # With no line-up saved, any link is refused, and typed names still work.
    with pytest.raises(ValueError, match="not in the line-up"):
        save_labels(tmp_path / "other.json", m, names([dict(lineup_id="h8", tracks=["L1"])]))
    assert save_labels(tmp_path / "other.json", m, names([dict(name="Ada", number=None, tracks=["L1"])]))["players"][0]["lineup_id"] is None


def test_names_saved_before_lineups_existed_still_load(tmp_path):
    m, path = manifest(), tmp_path / "player-labels.json"
    path.write_text(json.dumps(dict(schema_version="player-labels-1", run_id=m["run_id"], source_sha256="b" * 64, revision=3, source="human_entry",
                                    overlap_checked=True, players=[dict(id="P1", name="Ada", number=None, tracks=["L1"], kits=["kit-A"])])))
    loaded = load_labels(path, m)
    assert loaded["players"][0]["lineup_id"] is None and loaded["players"][0]["team"] is None
    roster = video_roster(load_review(tmp_path / "review.json", m), loaded)
    assert [(player["source"], player["name"]) for player in roster["players"]] == [("tracked_only", "Ada")]


def test_gateway_serves_the_roster_and_links_through_the_saved_lineup(tmp_path):
    def forbidden(*args): raise AssertionError("The roster must not invoke GPU")
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    client = TestClient(create_gateway(volume, SimpleNamespace(spawn=forbidden), "k" * 32, tmp_path))
    m, job_id = manifest(), "a" * 32
    (tmp_path / "requests" / job_id / "outputs" / m["run_id"]).mkdir(parents=True)
    (tmp_path / "web-jobs").mkdir(exist_ok=True)
    (tmp_path / "web-jobs" / f"{job_id}.json").write_text(json.dumps(dict(id=job_id, run_id=m["run_id"], status="complete", result=m, created_at="2026-11-22")))
    base, headers = f"/runs/{m['run_id']}", {"Authorization": "Bearer " + "k" * 32}
    assert client.get(base + "/roster").status_code == 401
    assert client.get(base + "/roster", headers=headers).json()["players"] == []
    assert client.post(base + "/review", headers=headers, json=review_body()).status_code == 200
    assert client.post(base + "/players", headers=headers, json=names([dict(lineup_id="nobody", tracks=["L1"])])).status_code == 422
    assert client.post(base + "/players", headers=headers, json=names([dict(lineup_id="h9", tracks=["L1"])])).json()["players"][0]["name"] == "Bo Lee"
    roster = client.get(base + "/roster", headers=headers).json()
    assert next(player for player in roster["players"] if player["name"] == "Bo Lee")["tracks"] == ["L1"]
    assert client.get(base + "/players", headers=headers).json()["players"][0]["team"] == "home"


def test_shirt_number_reads_are_matched_to_the_lineup_for_identities_nobody_has_named(tmp_path):
    from workers.video.video_roster import kit_teams, lineup_number_matches
    m, _, _, review, labels = saved(tmp_path)
    # L1 and L2 are named Ada Obi (home, kit-A); L3 is Cy Mo (away, kit-B); L4 has a typed name. Add two unnamed identities.
    m["detection_summary"]["linked_tracks"] += [dict(linked_track_id="L5", track_ids=["S0:5"], object_type="player", team="kit-A", observations=10, first_seen_ms=0, last_seen_ms=900),
                                                dict(linked_track_id="L6", track_ids=["S0:6"], object_type="player", team="kit-A", observations=10, first_seen_ms=0, last_seen_ms=900)]
    assert kit_teams(labels, review["lineups"]) == {"kit-A": "home", "kit-B": "away"}
    def entry(track, number, count):
        return dict(track_id=track, proposed_number=number if len(number) == 2 else None, confident_reads=count, crops_read=30, reads=[dict(number=number, count=count)])
    numbers = dict(reader="test", proposals=[], reads=[entry("L1", "8", 12), entry("L5", "9", 9), entry("L6", "9", 10), entry("L9", "9", 9)])
    out = lineup_number_matches(numbers, review, labels, m)["lineup"]
    by_track = {match["track_id"]: match for match in out["matches"]}
    # L1 already has a name and L9 is not a tracked player of this run: neither is matched.
    assert set(by_track) == {"L5", "L6"} and out["status"] == "checked" and out["counts"] == dict(match=2)
    assert by_track["L5"]["candidates"] == [dict(team="home", lineup_id="h9", name="Bo Lee", number="9", position=None)]
    assert by_track["L5"]["team_basis"] == "named_players_in_this_kit_group" and by_track["L5"]["kit"] == "kit-A"
    # Two identities reading the same listed player point at each other: probably one player split by the tracker.
    assert by_track["L5"]["same_player_tracks"] == ["L6"] and by_track["L6"]["same_player_tracks"] == ["L5"]
    empty = save_review(tmp_path / "bare.json", m, dict(review_body(), lineups=dict(home=[], away=[])))
    assert lineup_number_matches(numbers, empty, labels, m)["lineup"]["status"] == "no_lineup_numbers"
