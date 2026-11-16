# Evaluation against SoccerNet Game State Reconstruction (2026-11-15)

The first check of this app's player figures against labelled data. Until now the pitch positions, distances, speeds, merge suggestions and shirt-number proposals had only plausibility checks.

## What was run

- **Data:** 10 of the 49 labelled clips in the `test` split of [SoccerNet SN-GSR-2025](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) (public, not gated, tagged GPL-3.0): SNGS-116, 121, 126, 131, 136, 140, 145, 150, 191 and 196, taken evenly from the sorted list. Each is 30 s of main-camera broadcast at 1920 x 1080, 25 fps, from Swiss Super League matches. Every frame labels each person's box, role, team side, shirt number when visible, and foot position on the pitch in metres. The labels were made with an annotated camera per frame; they carry their own annotation error.
- **Pipeline:** the same analysis the worker runs (`run_real_analysis`: detector, BoT-SORT tracking, offline linking, pitch calibration with steadying), then the shirt-number reader with one-digit proposals allowed so they could be scored too. The frames were re-encoded to H.264 (CRF 14) first, because the pipeline reads video.
- **Scoring:** `workers/video/gsr_benchmark.py`, run by `deploy/modal_gsr_eval.py` as its own short-lived Modal app with its own volume (`laas-eval`). The analysis worker and the volume with users' clips were not touched. Only the ten clips' bytes were read from the 8.8 GB archive (about 1.9 GB).
- **Matching:** our players and goalkeepers are matched to labelled ones frame by frame on box overlap (intersection over union of at least 0.5). An identity of ours "follows" a labelled player when at least half its observations, and at least 5, match that player.

The detector's own training data is not documented, so whether it saw any of these matches is unknown. The pitch landmark model was trained on Bundesliga frames, not on this league.

## Results

| Clip | Action | Detection recall | Precision | Frames calibrated | Position error, median | Calibration alone, median | Distance, ours / labelled (per player median) | Identities per followed player |
|---|---|---|---|---|---|---|---|---|
| SNGS-116 | Corner | 24% | 20% | 74% | 5.3 m | 5.4 m | 1.64 | 1.19 |
| SNGS-121 | Yellow card | 39% | 23% | 50% | 8.0 m | 6.9 m | 0.91 | 1.43 |
| SNGS-126 | Clearance | 54% | 32% | 87% | 8.2 m | 7.6 m | 1.60 | 1.87 |
| SNGS-131 | Direct free-kick | 55% | 34% | 67% | 9.3 m | 8.6 m | 1.55 | 2.06 |
| SNGS-136 | Shots on target | 36% | 29% | 40% | 12.6 m | 12.6 m | n/a | 1.81 |
| SNGS-140 | Corner | 18% | 20% | 77% | 6.9 m | 7.2 m | 1.14 | 1.56 |
| SNGS-145 | Corner | 19% | 31% | 67% | 9.9 m | 10.3 m | n/a | 1.55 |
| SNGS-150 | Substitution | 59% | 42% | 41% | 9.4 m | 9.2 m | 0.66 | 2.22 |
| SNGS-191 | Offside | 76% | 67% | 90% | 20.5 m | 20.9 m | 1.98 | 2.76 |
| SNGS-196 | Clearance | 85% | 78% | 86% | 6.1 m | 6.1 m | 1.41 | 2.19 |
| **All** | | **47%** | **37%** | **68%** | | | **1.5** | **1.92** |

"Calibration alone" pushes the labelled box's foot point through our mapping, so it measures the calibration without our detector.

### Pitch positions: wrong by several metres

- Over 31,608 matched boxes with a position, the median error per clip was 5 to 21 m. Only 3% were within 2 m and 20% within 5 m.
- Calibration alone gives the same numbers, so the error comes from the pitch calibration, not from our boxes.
- The landmark checks do not catch it. On SNGS-191 a frame passed every check (10 landmarks agreeing to 0.31 m) while the model had placed a whole penalty area in midfield, where there is none. Six hallucinated landmarks that agree with each other fit one consistent, wrong mapping. The checks test whether the landmarks agree with each other, not whether they are real.
- 282 of 4,665 frames with positions had the ends or touchlines swapped relative to the rest of their clip.

### Distance and speed: overstated

Over the same frames and the same labelled players, with the profile's own method (half-second averaging and hops):

| | Ours | Labelled boxes through our mapping | Labelled positions |
|---|---|---|---|
| Distance (m) over 357 to 361 player-seconds | 1,136 | 1,116 | 714 |
| Average speed | 3.2 m/s | 3.1 m/s | 2.0 m/s |
| Time walking / jogging / running / fast running | 31 / 41 / 15 / 13% | 32 / 41 / 14 / 13% | 62 / 31 / 6 / 2% |
| Fastest half-second, median player | 6.2 m/s | 6.2 m/s | 3.8 m/s |

Distances came out about 1.6 times the labelled ones (per-player median 1.5, 10th to 90th percentile 1.1 to 2.1), and time at fast-running speed was about seven times too high. The calibration is the cause: the labelled boxes give the same inflation. Averaging positions over 1 to 2 s instead of 0.5 s did not fix it consistently: it helped on some clips and made others worse.

### Identities and kit groups

- Detection is the weakest link on this footage. On closer views the detector often returns a player as two boxes (upper body and legs), which neither matches the labelled box nor gives a foot point. Recall was 18% to 85% by clip.
- 2,835 identities against 214 labelled players. 315 identities follow a player; each followed player is split into 1.9 of them on average.
- Kit groups agreed with the labelled team side for 306 of 315 identities that follow a player (97%).

### Merge suggestions

The profile lists "possibly the same player" pairs. 9,563 suggestions were made across the ten clips, mostly between fragments that do not follow any labelled player, so only 224 could be checked.

| Suggestions | Checked | Same player |
|---|---|---|
| All | 224 | 43 (19%) |
| Close fit: distance at most 20% of what one player could cover in the gap | 40 | 19 (48%) |
| No rival identity fits the same gap | 2 | 1 |
| Gap at most 1 s | 10 | 4 |

How tight the fit was predicted a right suggestion; the number of rivals did not, and an unrivalled suggestion was rare (58 of 9,563). Of 69 true continuations between our identities, 32 were suggested; 21 more had gaps over the 10 s limit.

### Shirt numbers

| Proposals | Right | Wrong | No labelled number to compare |
|---|---|---|---|
| Two digits | 27 | 2 | 19 |
| One digit (not proposed in the app) | 4 | 3 | 3 |

Two-digit proposals were right 27 times in 29. One-digit reads stay unproposed: 4 in 7 is not good enough. Wrong ones included 10 for 34 and 11 for 17.

## What changed because of it

- **Merge suggestions** (`merge-suggestions-2`) are ranked by how tight the fit is, and each is marked `close_fit` when its distance is at most 20% of what one player could cover in the gap. The profile shows close fits by default and the rest behind a button, and says how often each kind was right here.
- **Wording.** The pitch figures, the shape view, the profile's method note and the shirt-number reader no longer say they are unchecked; they give these results.
- **Bridging rejected frames was tried and dropped.** The idea was to give a frame that fails calibration a mapping interpolated from calibrated frames at most 0.5 s either side, kept only if at least two of its own landmarks agreed. Across the ten clips it recovered 5 of 7,500 frames, because rejected frames come in long stretches rather than single dropouts, and the recovered positions were no better (9 to 10 m off). With calibration this far off where it is accepted, measuring more frames would add more wrong positions.

## Checking each mapping against the image's lines (added the same day)

`workers/video/pitch_lines.py` draws the pitch's markings (touchlines, goal lines, halfway line, penalty and goal areas, centre circle, penalty arcs) through a frame's mapping and measures the share of drawn points that land on a white line pixel within 1.5% of the image height (16 px at 1080p). Line pixels are thin bright streaks on or next to large grass areas.

**On the ten SoccerNet clips** (5,092 calibrated frames, `--lines-only` in the evaluation script):

- A clip's median score was 0.41 to 0.61, and at most 15% of a clip's frames reached 0.75.
- Within these clips the score only partly predicts a frame's error: a good frame outscored a frame more than 5 m off 69% of the time. Keeping only well-supported frames still left a median error of about 5 m, because the calibration is wrong almost everywhere on this footage, not now and then.
- Snapping the mapping onto the nearest lines and refitting made the drawn lines match much better (score up from about 0.2 to 0.6 at a tight tolerance) but not the positions: median error 7.1 m before, 6.8 m after, with 32% of frames better and 24% worse. It is kept in `pitch_lines.refine` for study and not used.

**On the stored clips in the hosted library** (read only, every fifth calibrated frame): two clips scored 0.94 and 0.99; on the one drawn here (a Bundesliga clip from the landmark model's example set) the drawn lines sit on the real ones. The Bundesliga test clips (dfl_test_1, 10 and 11, the landmark model's own domain) scored 0.58 to 0.64, and drawing their mappings shows why: on dfl_test_10 the near touchline is drawn about 60 px from the real one and the halfway line, centre circle and penalty area are visibly shifted. 23 of the 25 stored clips scored below the clip rule.

**What was built.** First as a filter, then, at the user's choice, as a warning:

- Withholding was tried first: a frame under 75% support was rejected, and a clip where under half its frames passed got no pitch positions. On three further test clips not used to set these numbers (SNGS-118, 138 and 193) the full pipeline then rejected every clip (0.2% to 8% of frames passed). Because it would also have removed the pitch figures from 23 of the 25 stored clips, it was changed to a warning.
- As built, every fifth calibrated frame is checked in the worker (`pitch_keypoints.check_lines`) and stores `line_support`; nothing is rejected or changed. The clip's calibration summary gets a `line_check`: `lines_match`, or `lines_do_not_match` when under half of the checked frames reach 75% support, or `not_run`. Calibration version `pitch-keypoints-4`.
- When the lines do not match, the Shape tab and the player profile say so above the pitch figures: positions are likely several metres off, and the pitch map, heat maps, distances and speeds are rough.
- The check costs about 50 ms per checked 1080p frame on a laptop CPU, roughly 8 s on a 30 s clip at one frame in five.
- Stored runs were calibrated before the check existed, so they say it was not run and show no warning. `backfill_lines` in `deploy/modal_app.py` checks them from their stored video and calibration on a CPU, without the landmark model; it needs the worker redeployed first.

## What this does not show

- Ten clips from one league, all from the model's non-training domain. The Bundesliga clips the landmark model was trained on may do much better; there are no labelled clips of that footage here.
- The labelled positions have their own error. It is small next to the differences found here.
- Merge suggestions: only 224 of 9,563 could be checked, and those are the identities that follow a player well, which is not a random sample.

## Next steps this points to

1. **Check the calibration against the image.** Done the same day (section above). It warns where figures are likely wrong, but does not make any more accurate.
2. **A better landmark or line model**, or one fine-tuned on SoccerNet calibration data, whose terms would need checking.
3. **A detector that keeps a player in one box** on closer views; split boxes are the main cause of fragmented identities here.

Command: `python -m modal run deploy/modal_gsr_eval.py --out-dir <folder> --count 10`, and `--rescore-only` to score stored runs again without a GPU.
