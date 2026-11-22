"""Every labelled person and every box of ours lands in one bin that names the fault."""
from workers.video.detection_diagnosis import diagnose, diagnose_frame, merge

PERSON = [100, 100, 140, 200]   # 40 wide, 100 tall


def label(box=PERSON, role="player"):
    return dict(box=box, role=role)


def ours(box, kind="player"):
    return dict(bbox=box, object_type=kind, confidence=.9)


def bins(boxes, labels):
    label_bins, our_bins = diagnose_frame(boxes, labels)
    return [kind for kind, _ in label_bins], [kind for kind, _ in our_bins]


def test_a_box_on_the_person_matches_and_a_second_one_is_a_duplicate():
    assert bins([ours([101, 102, 141, 199]), ours([98, 100, 138, 204])], [label()]) == (["matched"], ["matched", "duplicate"])


def test_upper_body_and_legs_boxed_apart_is_a_split_person():
    # Each half overlaps the person by 0.46, under the 0.5 needed to match; together they cover the whole person.
    assert bins([ours([100, 100, 140, 146]), ours([100, 154, 140, 200])], [label()]) == (["split"], ["piece", "piece"])
    # One piece alone covers too little to match, and nothing covers the rest.
    assert bins([ours([100, 100, 140, 145])], [label()]) == (["piece_only"], ["piece"])


def test_a_box_too_big_too_far_off_or_absent_is_named():
    assert bins([ours([80, 60, 180, 240])], [label()]) == (["oversized"], ["oversized"])
    assert bins([ours([120, 130, 160, 230])], [label()]) == (["partial"], ["misplaced"])
    assert bins([ours([500, 100, 540, 200])], [label()]) == (["missed"], ["off_label"])
    assert bins([], [label()]) == (["missed"], [])


def test_a_referee_boxed_as_a_player_is_another_kind_and_a_referee_box_is_not_judged():
    referee = label([300, 100, 340, 200], role="referee")
    assert bins([ours([300, 100, 340, 200])], [label(), referee]) == (["missed"], ["other_kind"])
    assert bins([ours([300, 100, 340, 200], kind="referee")], [referee]) == ([], [])
    # Scored as people rather than players, the referee must be found and a person box counts.
    label_bins, our_bins = diagnose_frame([ours([300, 100, 340, 200], kind="person")], [referee], target={"player", "goalkeeper", "referee"})
    assert [k for k, _ in label_bins] == ["matched"] and [k for k, _ in our_bins] == ["matched"]


def test_counts_rates_and_height_bands_add_up_and_merge_across_clips():
    small = [400, 100, 420, 150]
    frames = {0: [label(), label(small)], 5: [label()]}
    boxes = {0: [ours([100, 100, 140, 200]), ours([700, 0, 720, 40])], 5: [ours([100, 100, 140, 146]), ours([100, 154, 140, 200])]}
    report = diagnose(boxes, frames)
    assert (report["frames"], report["labelled"], report["ours"]) == (2, 3, 4)
    assert report["recall"] == .333 and report["precision"] == .25
    assert report["labelled_bins"] == dict(matched=1, split=1, piece_only=0, oversized=0, partial=0, missed=1)
    assert report["our_bins"] == dict(matched=1, duplicate=0, piece=2, oversized=0, other_kind=0, off_label=1, misplaced=0)
    assert report["recall_by_height"]["60 to 120 px"] == dict(labelled=2, recall=.5, split=1, piece_only=0, missed=0)
    assert report["recall_by_height"]["under 60 px"] == dict(labelled=1, recall=0., split=0, piece_only=0, missed=1)
    both = merge([report, report, None])
    assert both["clips"] == 2 and both["labelled"] == 6 and both["recall"] == .333 and both["labelled_bins"]["split"] == 2
    assert both["recall_by_height"]["60 to 120 px"]["labelled"] == 4 and both["recall_by_height"]["60 to 120 px"]["recall"] == .5
