"""One box per person before tracking: suppression across classes, and joining an upper-body box to a legs box."""
from workers.video.people_boxes import clean_people, join_stacked_pieces


def row(box, confidence=.9, kind="player"):
    return dict(bbox=box, confidence=confidence, object_type=kind)


def boxes(rows):
    return sorted(tuple(r["bbox"]) for r in rows)


def test_overlapping_boxes_of_any_class_are_one_person_and_the_most_confident_stays():
    kept = clean_people([row([100, 100, 140, 200], .6, "referee"), row([102, 101, 141, 199], .9), row([400, 100, 440, 200], .5)])
    assert boxes(kept) == [(102, 101, 141, 199), (400, 100, 440, 200)] and kept[0]["object_type"] == "player"


def test_a_piece_inside_a_larger_box_gives_way_even_when_it_is_more_confident():
    whole, piece = row([100, 100, 140, 200], .5), row([104, 100, 138, 150], .9)
    assert boxes(clean_people([whole, piece])) == [(100, 100, 140, 200)]
    assert boxes(clean_people([piece, whole])) == [(100, 100, 140, 200)]


def test_upper_body_and_legs_are_joined_into_one_person_box():
    upper, legs = row([100, 100, 140, 150], .7), row([102, 148, 142, 200], .8, "goalkeeper")
    joined = join_stacked_pieces([upper, legs, row([400, 100, 440, 200])])
    assert boxes(joined) == [(100, 100, 142, 200), (400, 100, 440, 200)]
    made = next(r for r in joined if r.get("joined_from_pieces"))
    assert made["confidence"] == .8 and made["object_type"] == "goalkeeper"   # the more confident piece names it


def test_two_whole_players_one_behind_the_other_are_left_alone():
    near, far = row([100, 150, 140, 250]), row([104, 60, 136, 150])   # each is tall: a whole person, not a piece
    assert boxes(join_stacked_pieces([near, far])) == boxes([near, far])
    # Squat boxes that are side by side, far apart vertically, or would make an impossibly thin person are not joined either.
    assert len(join_stacked_pieces([row([100, 100, 140, 150]), row([160, 150, 200, 200])])) == 2
    assert len(join_stacked_pieces([row([100, 100, 140, 150]), row([100, 220, 140, 270])])) == 2
    assert len(join_stacked_pieces([row([100, 100, 120, 136]), row([100, 136, 120, 172]), row([100, 172, 120, 208])])) == 2   # only one pair joins


def test_a_person_takes_the_kind_the_football_detector_saw_inside_it():
    from workers.video.people_boxes import adopt_roles
    people = [row([100, 100, 140, 200], kind="person"), row([300, 100, 340, 200], kind="person"), row([500, 100, 540, 200], kind="person")]
    hints = [row([102, 100, 138, 150], .5, "player"), row([104, 150, 138, 198], .8, "goalkeeper"),   # two pieces: the better-fitting one names it
             row([300, 100, 340, 200], .6, "referee"), row([480, 100, 520, 200], .9, "referee")]       # the last lies half outside the third person
    out = adopt_roles(people, hints)
    assert [r["object_type"] for r in out] == ["player", "referee", "referee"]
    assert [r["bbox"] for r in out] == [p["bbox"] for p in people]
    assert adopt_roles(people[:1], [], default="person")[0]["object_type"] == "person"


def test_the_pipeline_takes_people_from_the_person_detector_and_roles_from_the_football_weights(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.engine.results import Boxes
    from workers.video.detector import FootballDetector

    class Model:
        def __init__(self, path, *args):
            self.general = "person" in str(path)
            self.names = {0: "person", 32: "sports ball"} if self.general else {0: "ball", 1: "player", 2: "referee"}

        def predict(self, frame, classes=None, **kwargs):
            if self.general:   # whole people, as a general detector gives them
                data = [[100, 40, 140, 200, .9, 0], [198, 58, 238, 192, .8, 0], [250, 100, 256, 106, .6, 32], [20, 20, 26, 26, .05, 32]]
            else:              # the football weights box one player in two pieces, and the referee whole
                data = [[100, 40, 140, 115, .7, 1], [100, 125, 140, 200, .6, 1], [200, 60, 236, 190, .9, 2], [50, 50, 54, 54, .5, 0]]
            data = [d for d in data if classes is None or d[5] in classes]
            return [SimpleNamespace(names=self.names, boxes=Boxes(torch.tensor(data, dtype=torch.float32).reshape(-1, 6), frame.shape[:2]))]

    monkeypatch.setattr(ultralytics, "YOLO", Model)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = FootballDetector(tmp_path / "football.pt", device="cpu", image_size=640, ball_confidence=.1, object_confidence=.45,
                                sample_fps=25, person_weights=tmp_path / "person.pt")
    rows = detector.detect(frame)
    people = sorted((r for r in rows if r["object_type"] not in {"ball", "sports ball"}), key=lambda r: r["bbox"][0])
    assert [(r["bbox"], r["object_type"], r["model_role"]) for r in people] == [([100, 40, 140, 200], "player", "person_detector"), ([198, 58, 238, 192], "referee", "person_detector")]
    assert [r["bbox"] for r in rows if r["object_type"] == "ball"] == [[50, 50, 54, 54]]
    # The general model's ball class comes from the same pass and joins the football weights' ball as a candidate.
    assert [(r["bbox"], r["model_role"]) for r in rows if r["object_type"] == "sports ball"] == [([250, 100, 256, 106], "general_ball_detector")]
    assert detector.quality_counts["general_people"] == 2 and detector.quality_counts["roles_from_football_model"] == 1
    assert detector.quality_counts["general_ball_candidates"] == 1
    # Without a person detector the football weights' own boxes are used, as before.
    plain = FootballDetector(tmp_path / "football.pt", device="cpu", image_size=640, ball_confidence=.1, object_confidence=.45, sample_fps=25)
    assert len([r for r in plain.detect(frame) if r["object_type"] != "ball"]) == 3


def test_each_track_gets_one_role_from_all_its_frames():
    from workers.video.people_boxes import apply_roles, settle_roles

    def frames(track, kinds, kit="kit-A"):
        return [dict(track_id=track, object_type=kind, team=kit, frame_idx=i) for i, kind in enumerate(kinds)]

    rows = (frames("ref", ["player"] * 80 + ["referee"] * 20)              # called a referee in a fifth of its frames
            + frames("odd", ["player"] * 90 + ["referee"] * 10, kit=None)  # fewer calls, but in neither kit group
            + frames("out", ["player"] * 90 + ["referee"] * 10)            # the same few calls on a player in a kit group
            + frames("few", ["player"] * 6 + ["referee"] * 4)              # a high share, but only four frames
            + frames("gk", ["player"] * 85 + ["goalkeeper"] * 15)
            + frames("all", ["player"] * 50)
            + [dict(track_id=None, object_type="referee", team=None, frame_idx=0)])
    roles = settle_roles(rows)
    assert roles == dict(ref="referee", odd="referee", out="player", few="player", gk="goalkeeper", all="player")
    changed = apply_roles(rows, roles)
    assert changed == 80 + 90 + 10 + 4 + 85
    assert {row["object_type"] for row in rows if row["track_id"] == "ref"} == {"referee"}
    assert sum(row["role_hint"] == "referee" for row in rows if row["track_id"] == "ref") == 20   # the frame's own call is kept
    assert rows[-1]["object_type"] == "referee" and "role_hint" not in rows[-1]                    # an untracked box is left alone
    ball = [dict(track_id="ref", object_type="ball")]
    assert apply_roles(ball, roles) == 0 and ball[0]["object_type"] == "ball"
