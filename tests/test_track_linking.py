import json

import numpy as np

from workers.video.measurements import from_tracking
from workers.video.track_linking import AppearanceBank, LinkSettings, appearance_distance, apply_links, link_tracks, linked_view


def rows_for(track, frames, x, team=None, kind="player", speed=0.0):
    return [dict(frame_idx=f, track_id=track, object_type=kind, team=team, scene_id=0,
                 bbox=[x + speed * (f - frames[0]), 100, x + speed * (f - frames[0]) + 20, 160]) for f in frames]


def linked(result):
    return {tuple(i["track_ids"]) for i in result["identities"]}


def test_short_gap_at_predicted_position_is_linked_and_team_is_voted_across_fragments():
    rows = rows_for("T1", range(0, 20), 100, team="kit-A", speed=2) + rows_for("T2", range(30, 50), 160, speed=2)
    result = link_tracks(rows, {}, fps=25)
    assert linked(result) == {("T1", "T2")}
    assert result["mapping"]["T2"] == dict(linked_track_id="L1", linked_team="kit-A")
    assert result["links"][0]["gap_seconds"] == 0.44
    assert result["summary"]["tracks_before"] == 2 and result["summary"]["identities_after"] == 1


def test_overlapping_far_long_gap_and_conflicting_tracks_stay_separate():
    base = rows_for("T1", range(0, 20), 100, team="kit-A")
    cases = dict(overlap=rows_for("T2", range(10, 40), 102),
                 far=rows_for("T2", range(25, 45), 400),
                 long_gap=rows_for("T2", range(120, 140), 100),
                 other_team=rows_for("T2", range(25, 45), 100, team="kit-B"),
                 referee=rows_for("T2", range(25, 45), 100, kind="referee"))
    for name, other in cases.items():
        assert linked(link_tracks(base + other, {}, fps=25)) == {("T1",), ("T2",)}, name


def test_two_equally_plausible_successors_are_left_unlinked():
    rows = (rows_for("T1", range(0, 20), 100) + rows_for("T2", range(25, 45), 96) + rows_for("T3", range(25, 45), 104))
    result = link_tracks(rows, {}, fps=25)
    assert len(result["identities"]) == 3
    assert result["summary"]["tracklet_ends_left_ambiguous"] == 1


def test_kit_colour_resolves_an_ambiguous_choice_and_vetoes_a_mismatch():
    red, blue = np.zeros(80), np.zeros(80)
    red[[0, 40]] = 1
    blue[[24, 64]] = 1
    assert appearance_distance(red, red) == 0 and appearance_distance(red, blue) == 1
    rows = (rows_for("T1", range(0, 20), 100) + rows_for("T2", range(25, 45), 96) + rows_for("T3", range(25, 45), 104))
    result = link_tracks(rows, dict(T1=red, T2=blue, T3=red), fps=25)
    assert linked(result) == {("T1", "T3"), ("T2",)}
    result = link_tracks(rows_for("T1", range(0, 20), 100) + rows_for("T2", range(25, 45), 100), dict(T1=red, T2=blue), fps=25)
    assert len(result["identities"]) == 2


def test_chain_of_fragments_becomes_one_identity_and_camera_pan_is_compensated():
    # Six bystanders share a steady pan; the linked player must be compared in pan-compensated position.
    pan = 6
    rows = []
    for k in range(6):
        rows += [dict(frame_idx=f, track_id=f"B{k}", object_type="player", team=None, scene_id=0,
                      bbox=[300 + 60 * k + pan * f, 300, 320 + 60 * k + pan * f, 360]) for f in range(0, 70)]
    for name, frames in (("T1", range(0, 20)), ("T2", range(25, 45)), ("T3", range(50, 70))):
        rows += [dict(frame_idx=f, track_id=name, object_type="player", team=None, scene_id=0,
                      bbox=[100 + pan * f, 100, 120 + pan * f, 160]) for f in frames]
    result = link_tracks(rows, {}, fps=25, settings=LinkSettings())
    assert ("T1", "T2", "T3") in linked(result)
    assert all(len(i["track_ids"]) == 1 for i in result["identities"] if i["track_ids"][0].startswith("B"))


def test_appearance_bank_separates_kits_and_skips_overlapping_players():
    frame = np.zeros((200, 300, 3), np.uint8)
    frame[:, :150] = (0, 0, 255)
    frame[:, 150:] = (255, 0, 0)
    bank = AppearanceBank(every=1)
    bank.observe(frame, [dict(track_id="T1", object_type="player", bbox=[20, 20, 60, 180]),
                         dict(track_id="T2", object_type="player", bbox=[200, 20, 240, 180]),
                         dict(track_id="T3", object_type="player", bbox=[100, 20, 140, 180]),
                         dict(track_id="T4", object_type="player", bbox=[110, 20, 150, 180])])
    found = bank.descriptors()
    assert set(found) == {"T1", "T2"}
    assert appearance_distance(found["T1"], found["T2"]) > .9


def test_apply_links_keeps_original_track_ids_and_valid_json(tmp_path):
    path = tmp_path / "tracking.json"
    rows = [dict(frame_idx=0, track_id="T1", object_type="player"), dict(frame_idx=0, track_id=None, object_type="ball")]
    path.write_text("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]", encoding="utf-8")
    apply_links(path, dict(mapping=dict(T1=dict(linked_track_id="L1", linked_team=None))))
    out = json.loads(path.read_text(encoding="utf-8"))
    assert [r["track_id"] for r in out] == ["T1", None]
    assert [r["linked_track_id"] for r in out] == ["L1", None]


def test_learned_embeddings_choose_between_same_kit_candidates_after_centring():
    rows = (rows_for("T1", range(0, 20), 100) + rows_for("T2", range(25, 45), 96) + rows_for("T3", range(25, 45), 104))
    shared = np.array([10., 0, 0])  # component common to every player, removed by centring
    unit = lambda v: (shared + v) / np.linalg.norm(shared + v)
    embeddings = dict(T1=unit(np.array([0, 1., 0])), T2=unit(np.array([0, -1., 1.])), T3=unit(np.array([0, 1., .1])))
    assert len(link_tracks(rows, {}, fps=25)["identities"]) == 3
    result = link_tracks(rows, {}, fps=25, embeddings=embeddings)
    assert linked(result) == {("T1", "T3"), ("T2",)}
    assert result["links"][0]["embedding_distance"] < .5
    assert result["summary"]["tracks_with_embedding"] == 3


def test_measurements_follow_linked_identities_and_voted_team():
    meta = dict(width=200, height=200, fps=10, frame_count=4, duration_seconds=.4)
    def row(frame, track, linked, team=None):
        return dict(frame_idx=frame, object_type="player", track_id=track, linked_track_id=linked, linked_team="kit-A",
                    team=team, bbox=[10, 10, 20, 40], confidence=.9)
    rows = [row(0, "T1", "L1", "kit-A"), row(1, "T1", "L1"), row(2, "T9", "L1"), row(3, "T9", "L1")]
    view, source = linked_view(rows)
    assert source == "linked_track_id" and {r["track_id"] for r in view} == {"L1"}
    assert [r["source_track_id"] for r in view] == ["T1", "T1", "T9", "T9"] and all(r["team"] == "kit-A" for r in view)
    assert rows[2]["track_id"] == "T9"  # the stored rows are not modified
    measured = from_tracking(rows, meta, dict(sample_fps=10, decoded_frames=4))
    assert measured["track_identity"] == "linked_track_id"
    assert [(t["track_id"], t["observations"]) for t in measured["tracks"]] == [("L1", 4)]
    unlinked = [dict(r) for r in rows]
    for r in unlinked:
        del r["linked_track_id"], r["linked_team"]
    plain = from_tracking(unlinked, meta, dict(sample_fps=10, decoded_frames=4))
    assert "track_identity" not in plain and len(plain["tracks"]) == 2


def test_identity_seen_mostly_as_referee_gets_no_kit_group():
    rows = rows_for("T1", range(0, 20), 100, kind="referee") + rows_for("T1", range(20, 25), 100, team="kit-A")
    assert link_tracks(rows, {}, fps=25)["identities"][0]["team"] is None
