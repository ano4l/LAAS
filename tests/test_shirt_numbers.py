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


# ---- Reads checked against a team sheet ----

from workers.video.shirt_numbers import match_lineup

SHEET = dict(home={"17": dict(lineup_id="h17", name="Ada", number="17", position="MF"), "7": dict(lineup_id="h7", name="Bo", number="7", position="FW"),
                   "1": dict(lineup_id="h1", name="Keeper", number="1", position="GK")},
             away={"7": dict(lineup_id="a7", name="Cy", number="7", position="FW"), "23": dict(lineup_id="a23", name="Di", number="23", position="DF")})
KEEPERS = dict(home={"1"}, away=set())


def reads(track, counts, proposed=None):
    return dict(track_id=track, proposed_number=proposed, confident_reads=sum(counts.values()), crops_read=40,
                reads=[dict(number=number, count=count) for number, count in sorted(counts.items(), key=lambda item: -item[1])])


def matched(counts, team=None, proposed=None, keeper=False):
    out = match_lineup([reads("L1", counts, proposed)], SHEET, {"L1": team} if team else {}, KEEPERS, {"L1"} if keeper else set())
    return out.get("L1")


def test_a_number_on_one_team_only_names_the_player_without_knowing_the_team():
    result = matched({"17": 6, "3": 1})
    assert result["status"] == "match" and result["candidates"] == [dict(team="home", **SHEET["home"]["17"])]
    assert result["team_basis"] == "number_worn_on_one_team_only" and result["one_digit"] is False and result["agreement"] == .86


def test_half_reads_of_the_matched_number_do_not_count_against_it():
    # Without a sheet, six "1" against five "17" proposed nothing. Nobody outfield wears 1 here, and "1" is half of "17".
    result = matched({"1": 6, "17": 5})
    assert result["status"] == "match" and result["number"] == "17" and result["against"] == 0 and result["unconstrained_top_read"] == "1"


def test_a_number_worn_on_both_teams_needs_the_team():
    assert matched({"7": 9})["status"] == "ambiguous_team" and len(matched({"7": 9})["candidates"]) == 2
    known = matched({"7": 9}, team="away")
    assert known["status"] == "match" and known["candidates"][0]["lineup_id"] == "a7" and known["team_basis"] == "named_players_in_this_kit_group"


def test_one_digit_numbers_need_more_reads_and_more_agreement():
    assert matched({"7": 7}, team="home")["status"] == "too_few_reads"
    assert matched({"7": 8, "23": 4}, team="away")["status"] == "too_few_reads"   # eight reads, but a third disagree
    assert matched({"7": 8}, team="home")["one_digit"] is True and matched({"7": 8}, team="home")["status"] == "match"
    assert matched({"17": 3}, team="home")["status"] == "too_few_reads"


def test_reads_that_are_not_on_the_sheet_are_reported_not_matched():
    assert matched({"99": 9}, proposed="99") == dict(status="not_on_sheet", number="99", reads=9, confident_reads=9, team=None, candidates=[])
    assert matched({"99": 9}) is None                      # never proposed, nothing to say
    assert matched({"23": 9}, team="home", proposed="23")["status"] == "not_on_sheet"   # worn by the other team only


def test_an_outfield_identity_does_not_match_a_goalkeepers_number():
    # Stripes and arms read as "1" were the commonest false read; only an identity the detector calls a goalkeeper can match the keeper.
    assert matched({"1": 20}, team="home")["status"] == "keeper_number_on_outfield_player"
    assert matched({"1": 20}, team="home", keeper=True)["candidates"][0]["lineup_id"] == "h1"
