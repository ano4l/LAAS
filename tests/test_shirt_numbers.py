from workers.video.shirt_numbers import MAX_CROPS_PER_IDENTITY, crop_plan, vote


def row(frame, identity, box, kind="player"):
    return dict(frame_idx=frame, linked_track_id=identity, track_id="raw", object_type=kind, bbox=box)


def test_only_large_unoverlapped_players_are_cropped_and_sampled_sparsely():
    rows = []
    for frame in range(12):
        rows.append(row(frame, "big", [100, 100, 160, 260]))
        rows.append(row(frame, "small", [400, 100, 420, 150]))  # 50 px tall: digits too small
        rows.append(row(frame, "crowded", [600, 100, 660, 260]))
        rows.append(row(frame, "blocker", [620, 100, 680, 260], kind="referee"))
    plan = crop_plan(rows)
    assert sorted(plan) == [0, 3, 6, 9]
    assert {identity for crops in plan.values() for identity, _ in crops} == {"big"}
    (_, box), = plan[0]
    assert box == (100 + .12 * 60, 100 + .12 * 160, 100 + .88 * 60, 100 + .58 * 160)


def test_crops_per_identity_are_capped():
    rows = [row(frame, "big", [100, 100, 160, 260]) for frame in range(3 * MAX_CROPS_PER_IDENTITY + 60)]
    assert sum(len(crops) for crops in crop_plan(rows).values()) == MAX_CROPS_PER_IDENTITY


def proposal(reads, tried=None):
    return vote({"L1": reads}, {"L1": tried or len(reads)})[0]


def test_a_number_is_proposed_only_when_enough_confident_reads_agree():
    agreed = proposal([("28", .95)] * 5 + [("3", .9)])
    assert agreed["proposed_number"] == "28" and agreed["agreement"] == .83 and agreed["confident_reads"] == 6
    assert proposal([("28", .95)] * 3)["proposed_number"] is None  # too few reads
    assert proposal([("28", .95)] * 4 + [("23", .95)] * 4)["proposed_number"] is None  # no agreement
    assert proposal([("28", .5)] * 9)["proposed_number"] is None  # unsure reads do not count
    # One-digit reads were mostly stripes and arms on test footage, so they are counted but never proposed.
    single = proposal([("1", .99)] * 20)
    assert single["proposed_number"] is None and single["reads"] == [dict(number="1", count=20)]
    words = proposal([("FLY", .99), ("123", .99), ("", .99), ("8a", .99)] * 3)
    assert words["proposed_number"] is None and words["confident_reads"] == 0


def test_half_of_a_two_digit_number_does_not_outvote_it():
    out = proposal([("1", .95)] * 6 + [("17", .95)] * 5)
    assert out["proposed_number"] is None  # neither reaches the agreement share, so nothing is proposed
    out = proposal([("1", .95)] * 4 + [("17", .95)] * 9)
    assert out["proposed_number"] == "17"


def test_an_identity_with_no_reads_is_listed_without_a_proposal():
    assert vote({}, {"L2": 7}) == [dict(track_id="L2", crops_read=7, confident_reads=0, reads=[], proposed_number=None, agreement=None)]
