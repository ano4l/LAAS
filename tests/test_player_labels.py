import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api.modal_gateway import create_gateway
from workers.video.event_review import ReviewConflict
from workers.video.player_labels import load_labels, save_labels


def manifest():
    linked = [dict(linked_track_id=f"L{n}", track_ids=[f"S0:{n}"], object_type=kind, team=team, observations=10, first_seen_ms=0, last_seen_ms=900)
              for n, kind, team in ((1, "player", "kit-A"), (2, "player", "kit-A"), (3, "goalkeeper", None), (4, "referee", None), (5, "player", "kit-B"))]
    return dict(run_id="9" * 32, engine="yolo_bytetrack", metadata=dict(sha256="a" * 64, duration_seconds=1),
                detection_summary=dict(linked_tracks=linked))


def tracking(path):
    # L1 is seen in frames 0-9, L2 in frames 10-19 (one player, split by the tracker), L5 alongside L1 throughout.
    rows = [dict(frame_idx=i, linked_track_id="L1") for i in range(10)] + [dict(frame_idx=i, linked_track_id="L2") for i in range(10, 20)]
    rows += [dict(frame_idx=i, linked_track_id="L5") for i in range(10)]
    path.write_text(json.dumps(rows))
    return path


def payload(players, revision=0):
    return dict(revision=revision, confirmed=True, players=players)


def test_names_are_saved_per_source_and_one_player_can_hold_several_identities(tmp_path):
    path, m = tmp_path / "player-labels.json", manifest()
    assert load_labels(path, m)["players"] == [] and not path.exists()
    state = save_labels(path, m, payload([dict(name=" Ada Obi ", number="8", tracks=["L2", "L1"]), dict(name="Keeper", number=None, tracks=["L3"])]),
                        tracking(tmp_path / "tracking.json"))
    assert state["revision"] == 1 and state["overlap_checked"] is True and state["source"] == "human_entry"
    assert state["players"][0] == dict(id="P1", name="Ada Obi", number="8", tracks=["L1", "L2"], kits=["kit-A"], lineup_id=None, team=None)
    assert state["players"][1]["kits"] == []
    assert load_labels(path, m) == state
    with pytest.raises(ReviewConflict):
        save_labels(path, m, payload([], revision=0))
    assert save_labels(path, m, payload([], revision=1))["players"] == []
    with pytest.raises(ValueError, match="source mismatch"):
        load_labels(path, dict(m, run_id="8" * 32))


def test_two_identities_on_screen_together_cannot_be_one_player(tmp_path):
    path, m = tmp_path / "player-labels.json", manifest()
    with pytest.raises(ValueError, match="on screen together in 10 analysed frames"):
        save_labels(path, m, payload([dict(name="Ada", number=None, tracks=["L1", "L5"])]), tracking(tmp_path / "tracking.json"))
    assert not path.exists()
    # Without the stored detections the check cannot run, and the record says so.
    assert save_labels(path, m, payload([dict(name="Ada", number=None, tracks=["L1", "L5"])]))["overlap_checked"] is False


@pytest.mark.parametrize("players, message", [
    ([dict(name="", number=None, tracks=["L1"])], "player name"),
    ([dict(name="Ada", number="123", tracks=["L1"])], "shirt number"),
    ([dict(name="Ada", number=8, tracks=["L1"])], "shirt number"),
    ([dict(name="Ada", number=None, tracks=[])], "at least one"),
    ([dict(name="Ada", number=None, tracks=["L4"])], "not present"),  # a referee is not a player
    ([dict(name="Ada", number=None, tracks=["L9"])], "not present"),
    ([dict(name="Ada", number=None, tracks=["L1"]), dict(name="Bo", number=None, tracks=["L1"])], "one named player only"),
    ([dict(name="Ada", number="8", tracks=["L1"]), dict(name="ada", number="8", tracks=["L2"])], "same name and number"),
])
def test_invalid_names_are_refused(tmp_path, players, message):
    with pytest.raises(ValueError, match=message):
        save_labels(tmp_path / "player-labels.json", manifest(), payload(players))


def test_saving_needs_confirmation_and_a_hashed_real_source(tmp_path):
    path, m = tmp_path / "player-labels.json", manifest()
    with pytest.raises(ValueError, match="Confirm"):
        save_labels(path, m, dict(revision=0, players=[]))
    with pytest.raises(ValueError, match="hash"):
        save_labels(path, dict(m, metadata={}), payload([]))


def test_gateway_requires_the_key_saves_names_and_never_starts_inference(tmp_path):
    def forbidden(*args):
        raise AssertionError("Naming players must not start inference")
    client = TestClient(create_gateway(SimpleNamespace(commit=lambda: None, reload=lambda: None), SimpleNamespace(spawn=forbidden), "k" * 32, tmp_path))
    headers = {"Authorization": "Bearer " + "k" * 32}
    m, job_id = manifest(), "8" * 32
    directory = tmp_path / "requests" / job_id / "outputs" / m["run_id"]
    directory.mkdir(parents=True)
    tracking(directory / "tracking.json")
    (tmp_path / "web-jobs" / f"{job_id}.json").write_text(json.dumps(dict(id=job_id, run_id=m["run_id"], status="complete", created_at="2026-11-15", result=m)))
    url = f"/runs/{m['run_id']}/players"
    body = payload([dict(name="Ada", number="8", tracks=["L1", "L2"])])
    assert client.get(url).status_code == 401 and client.post(url, json=body).status_code == 401
    assert client.get(url, headers=headers).json()["revision"] == 0
    assert client.post(url, headers=headers, json=payload([dict(name="Ada", number=None, tracks=["L1", "L5"])])).status_code == 422
    assert client.post(url, headers=headers, json=body).json()["players"][0]["tracks"] == ["L1", "L2"]
    assert client.post(url, headers=headers, json=body).status_code == 409
    assert client.get(url, headers=headers).json()["revision"] == 1
    # The roster lists names saved on any clip, behind the same key.
    assert client.get("/roster").status_code == 401
    assert client.get("/roster", headers=headers).json() == dict(players=[dict(name="Ada", number="8", clips=1)])
    # Suggestions need the key too, read stored detections only, and never join anything.
    assert client.get(url + "/suggestions").status_code == 401
    suggested = client.get(url + "/suggestions", headers=headers).json()
    assert suggested["suggestions"] == [] and suggested["review_status"] == "unreviewed"
