# Shirt-number reading: first test (2026-11-15)

Experimental. Not part of the analysis worker and not deployed.

## What was run

`deploy/modal_shirt_numbers.py` ran once on Modal as its own short-lived app, over the 25 finished clips in hosted storage, without writing to the volume. `workers/video/shirt_numbers.py` crops each player's torso, reads it with PARSeq, and votes per tracked identity.

- **Reader:** PARSeq base, pretrained weights from the `baudm/parseq` v1.0.0 release, code pinned to commit `1902db04`. The code is Apache-2.0. The weights were trained on public scene-text datasets whose terms were not checked. It was trained on signs and labels, not shirts.
- **Crops:** players at least 70 px tall, not overlapped by another person by more than 10%, every third analysed frame, at most 80 per identity.
- **A read counts** when it is one or two digits and its least certain character is at least 0.85.
- **A proposal needs** at least 4 counted reads on one identity with at least 60% agreeing.

## Result

| | |
|---|---|
| Clips | 25 |
| Clips with any player large enough to crop | 19 |
| Crops read | 13,563 |
| Reads that counted | 587 |
| Clips with a proposal | 5 (three of them are the same 17 s footage analysed three times) |
| Proposals | 26: 16 two-digit, 10 one-digit |

Checked by eye on six sample crops per identity, for the 10 proposals in three clips:

- **Two-digit proposals: 6 of 6 right** (19, 14, 11, 14, 15, 28). The number is plainly visible on the back.
- **One-digit proposals: 4 of 4 wrong.** Three were "1" read from a striped shirt or an arm, one was "2" read from a front view with no number.

The other 16 proposals were not looked at. No clip has labelled shirt numbers, so there is no accuracy figure.

## What changed because of it

- One-digit numbers are counted but never proposed (`PROPOSE_ONE_DIGIT = False`). Players numbered 1 to 9 therefore get no proposal.
- In the wide broadcast clips that make up most of the library, players are too small: the 30 s 1080p clips gave 0 to 15 counted reads each and no proposals. All proposals came from two closer-view clips.

## Not done

- No legibility check before reading, which is what the published pipelines use to reject fronts, stripes and sponsors.
- Nothing is stored for the hosted clips. Running the script with `--save` writes `shirt-numbers.json` beside a run, and the Players tab then shows each proposal with a "Use" button. That display was type-checked but not seen in a browser.
- Two tracks of one clip were both read as 14. A shared number could feed the "possibly the same player" suggestions; that is not built.
