"""Shirt-number proposals: read digits on player crops, then vote across each tracked identity.

Experimental and offline. A general scene-text model reads a crop of each player's torso
on frames where the player is large enough and not overlapped. A single read is weak
evidence, so nothing is proposed from one frame: an identity gets a proposed number only
when enough confident reads agree. A proposal is a question for a person, never a label;
the number may belong to a player standing behind, or be a sponsor's digits.

The reader is passed in, so the voting can be tested without a model. `ParseqReader`
wraps PARSeq (Apache-2.0 code); it was trained on general scene text, not on shirts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np

VERSION = "shirt-numbers-1"
PEOPLE = {"player", "goalkeeper"}
MIN_BOX_HEIGHT_PX = 70  # smaller players have digits only a few pixels tall
MAX_OVERLAP = .1  # another person's box overlapping more than this may lend its number
SAMPLE_EVERY = 3  # analysed frames between crops of one identity
MAX_CROPS_PER_IDENTITY = 80
MIN_READ_CONFIDENCE = .85  # the least certain character of a read
MIN_READS = 4  # confident reads an identity needs before anything is proposed
MIN_SHARE = .6  # of those reads, the share that must agree
# On the first test footage every one-digit proposal that was checked was false (shirt stripes and arms read as "1",
# a front view read as "2"), while every two-digit proposal checked was right. So one-digit numbers are not proposed.
PROPOSE_ONE_DIGIT = False
TORSO = (.12, .58, .12, .88)  # top, bottom, left, right of the person box
LIMITATIONS = ["Experimental. A proposal is a question for a person watching the footage, not a recognised player.",
               "The reader was trained on general scene text, not on shirts. On ten labelled SoccerNet test clips, 27 of 29 two-digit proposals that could be checked were right; one-digit ones were right 4 times in 7.",
               "Only large, unoverlapped player crops are read, so most players in a wide view get no proposal.",
               "A read can come from a sponsor's digits or from another player close behind.",
               "Front and back are not told apart; a player facing the camera usually shows no number.",
               "One-digit numbers are not proposed: on test footage those reads were mostly shirt stripes or arms, not numbers."]


def _iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def crop_plan(rows):
    """{frame_idx: [(identity, torso box), ...]} for the crops worth reading."""
    by_frame, seen, taken = defaultdict(list), Counter(), Counter()
    for row in rows:
        if row.get("object_type") in PEOPLE | {"referee", "person"} and row.get("bbox"):
            by_frame[row["frame_idx"]].append(row)
    plan = defaultdict(list)
    for frame in sorted(by_frame):
        people = by_frame[frame]
        for row in people:
            identity = row.get("linked_track_id") or row.get("track_id")
            if not identity or row["object_type"] not in PEOPLE:
                continue
            seen[identity] += 1
            a, b, c, d = row["bbox"]
            if (seen[identity] - 1) % SAMPLE_EVERY or taken[identity] >= MAX_CROPS_PER_IDENTITY or d - b < MIN_BOX_HEIGHT_PX:
                continue
            if any(other is not row and _iou(row["bbox"], other["bbox"]) > MAX_OVERLAP for other in people):
                continue
            top, bottom, left, right = TORSO
            plan[frame].append((identity, (a + left * (c - a), b + top * (d - b), a + right * (c - a), b + bottom * (d - b))))
            taken[identity] += 1
    return plan


def vote(reads, crops_tried):
    """Proposals from `reads` ({identity: [(text, confidence), ...]}) and `crops_tried` ({identity: count})."""
    proposals = []
    for identity in sorted(crops_tried):
        confident = [text for text, confidence in reads.get(identity, []) if re.fullmatch(r"\d{1,2}", text) and confidence >= MIN_READ_CONFIDENCE]
        counts = Counter(confident)
        entry = dict(track_id=identity, crops_read=crops_tried[identity], confident_reads=len(confident),
                     reads=[dict(number=number, count=count) for number, count in counts.most_common(4)], proposed_number=None, agreement=None)
        if len(confident) >= MIN_READS:
            number, count = counts.most_common(1)[0]
            # A one-digit read is often half of a two-digit number, so "1" does not outvote "10" or "17" by count alone.
            longer = [(n, c) for n, c in counts.items() if len(n) == 2 and number in (n[0], n[1]) and len(number) == 1]
            if longer and max(c for _, c in longer) >= MIN_READS and sum(c for _, c in longer) >= count / 2:
                number, count = max(longer, key=lambda item: item[1])
            if count >= MIN_READS and count / len(confident) >= MIN_SHARE and (len(number) == 2 or PROPOSE_ONE_DIGIT):
                entry.update(proposed_number=number, agreement=round(count / len(confident), 2))
        proposals.append(entry)
    return proposals


LINEUP_VERSION = "shirt-number-lineup-1"
# A one-digit read is weak on its own (stripes and arms read as "1"), so on a team sheet it needs more reads and more agreement.
LINEUP_ONE_DIGIT_MIN_READS = 8
LINEUP_ONE_DIGIT_MIN_SHARE = .75
LINEUP_LIMITATIONS = [
    "A match says only that the reader's digits are a number on the saved line-up. It is a question for a person watching the footage, not an identification.",
    "The line-up narrows the choice; it does not make a misread right. A sponsor's digits or a player standing behind can still match someone on the sheet.",
    "Matching against a line-up has not been scored against labelled footage. One-digit matches in particular are unmeasured.",
    "Which team an identity is on comes from its kit group and from players a person has already named. Where that is not known, a number worn on both teams cannot be resolved.",
]


def match_lineup(proposals, sheet, team_of=None, keepers=None, keeper_tracks=None):
    """What each identity's reads say once only numbers on the team sheet can be right.

    `proposals` are the entries `vote` returns (their `reads` hold the most frequent confident reads).
    `sheet` is {team: {number: player}}; `team_of` is {identity: team} where a person's naming has
    established it; `keepers` is {team: set of numbers worn by goalkeepers} and `keeper_tracks` the
    identities the detector calls goalkeepers. Returns {identity: result}. Nothing is decided here.
    """
    team_of, keepers, keeper_tracks = team_of or {}, keepers or {}, set(keeper_tracks or ())
    results = {}
    for entry in proposals:
        track, reads = entry["track_id"], [(read["number"], read["count"]) for read in entry.get("reads", [])]
        confident = entry.get("confident_reads") or 0
        if not reads:
            continue
        teams = [team_of[track]] if team_of.get(track) in sheet else list(sheet)
        allowed = {number for team in teams for number in sheet[team]}
        on_sheet = [(number, count) for number, count in reads if number in allowed]
        top_number = reads[0][0]
        if not on_sheet:
            if entry.get("proposed_number"):
                results[track] = dict(status="not_on_sheet", number=entry["proposed_number"], reads=reads[0][1], confident_reads=confident,
                                      team=team_of.get(track), candidates=[])
            continue
        number, count = max(on_sheet, key=lambda item: (item[1], len(item[0])))
        # Half of a two-digit number: "1" does not beat a "17" that is also on the sheet and read often enough.
        if len(number) == 1:
            longer = [(n, c) for n, c in on_sheet if len(n) == 2 and number in n and c >= MIN_READS and c >= count / 2]
            if longer:
                number, count = max(longer, key=lambda item: item[1])
        # Reads that are one digit of the chosen number are the same shirt seen partly; they do not count against it.
        partial = sum(c for n, c in reads if n != number and len(n) == 1 and len(number) == 2 and n in number)
        against = max(0, confident - count - partial)
        share = count / (count + against) if count + against else 0.
        one_digit = len(number) == 1
        enough = (count >= LINEUP_ONE_DIGIT_MIN_READS and share >= LINEUP_ONE_DIGIT_MIN_SHARE) if one_digit else (count >= MIN_READS and share >= MIN_SHARE)
        candidates = [dict(team=team, **sheet[team][number]) for team in teams if number in sheet[team]]
        # The detector's goalkeeper class is the only cue for who is in goal: an outfield identity does not match a keeper's number.
        if candidates and track not in keeper_tracks:
            outfield = [c for c in candidates if number not in keepers.get(c["team"], ())]
            if not outfield:
                results[track] = dict(status="keeper_number_on_outfield_player", number=number, reads=count, confident_reads=confident,
                                      team=team_of.get(track), candidates=[])
                continue
            candidates = outfield
        status = "too_few_reads" if not enough else "match" if len(candidates) == 1 else "ambiguous_team"
        results[track] = dict(status=status, number=number, reads=count, against=against, agreement=round(share, 2), confident_reads=confident,
                              one_digit=one_digit, team=candidates[0]["team"] if len(candidates) == 1 else team_of.get(track),
                              team_basis="named_players_in_this_kit_group" if team_of.get(track) in sheet else "number_worn_on_one_team_only" if len(candidates) == 1 else None,
                              candidates=candidates, unconstrained_top_read=top_number if top_number != number else None)
    return results


def read_run(video, rows, reader, crop_sink=None):
    """Read planned crops from `video` with `reader` and vote. `crop_sink(identity, frame, crop, text, confidence)` is optional."""
    import cv2
    plan = crop_plan(rows)
    reads, tried = defaultdict(list), Counter()
    capture = cv2.VideoCapture(str(video))
    try:
        frame_idx, wanted = 0, set(plan)
        last = max(wanted, default=-1)
        while frame_idx <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx in wanted:
                crops, owners = [], []
                for identity, (a, b, c, d) in plan[frame_idx]:
                    crop = frame[max(0, int(b)):int(d), max(0, int(a)):int(c)]
                    if crop.shape[0] >= 16 and crop.shape[1] >= 12:
                        crops.append(crop)
                        owners.append(identity)
                for identity, crop, (text, confidence) in zip(owners, crops, reader.read(crops) if crops else []):
                    tried[identity] += 1
                    reads[identity].append((text, confidence))
                    if crop_sink:
                        crop_sink(identity, frame_idx, crop, text, confidence)
            frame_idx += 1
    finally:
        capture.release()
    proposals = vote(reads, tried)
    return dict(schema_version=VERSION, review_status="unreviewed", accuracy_evaluated=False, reader=getattr(reader, "name", "unknown"),
                parameters=dict(min_box_height_px=MIN_BOX_HEIGHT_PX, max_overlap=MAX_OVERLAP, sample_every_analysed_frames=SAMPLE_EVERY,
                                max_crops_per_identity=MAX_CROPS_PER_IDENTITY, min_read_confidence=MIN_READ_CONFIDENCE, min_reads=MIN_READS,
                                min_share=MIN_SHARE, propose_one_digit=PROPOSE_ONE_DIGIT, torso=list(TORSO)),
                identities_read=len(proposals), identities_proposed=sum(p["proposed_number"] is not None for p in proposals),
                crops_read=int(sum(tried.values())), proposals=proposals, limitations=LIMITATIONS)


class ParseqReader:
    """PARSeq scene-text recogniser loaded from a local checkout of baudm/parseq (pretrained weights from its release)."""

    def __init__(self, repository, device="cpu", batch=128):
        import torch
        self.torch = torch
        self.net = torch.hub.load(str(repository), "parseq", pretrained=True, source="local").eval().to(device)
        self.device, self.batch, self.name = device, batch, "parseq-base-pretrained"

    def read(self, crops):
        """(text, confidence of the least certain character) per BGR crop."""
        import cv2
        out = []
        for start in range(0, len(crops), self.batch):
            images = np.stack([cv2.cvtColor(cv2.resize(crop, (128, 32), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) for crop in crops[start:start + self.batch]])
            tensor = (self.torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float() / 255 - .5) / .5
            with self.torch.no_grad():
                texts, confidences = self.net.tokenizer.decode(self.net(tensor).softmax(-1))
            out += [(text, float(confidence.min()) if len(confidence) else 0.) for text, confidence in zip(texts, confidences)]
        return out
