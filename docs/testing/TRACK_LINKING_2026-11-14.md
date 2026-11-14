# Offline track linking, 14 November 2026

`workers/video/track_linking.py` joins fragmented person tracks after the whole clip has been processed. Each tracking row gains `linked_track_id` and `linked_team`; the tracker's own `track_id` is unchanged. The run directory gains `track-links.json` (identities, accepted links and their evidence), and the manifest gains `detection_summary.track_linking` and `detection_summary.linked_tracks`.

## Method

A tracklet's end is joined to another tracklet's start only when all of these hold:

- same scene, gap of at most 3 seconds, both tracklets at least 5 observations;
- the start lies within `0.4 + 1.0 × gap seconds` player heights of the position predicted from the end, after subtracting camera pan estimated from the median movement of all tracked people;
- box heights within a factor of 1.6;
- kit-colour histograms (shirt and shorts, grass masked) within 0.45, when both tracklets have one;
- no conflicting kit group, and no referee/non-referee conflict;
- no rival that overlaps the chosen partner in time comes within 0.35 cost of it.

Team is then voted across all fragments of a linked identity (at least 3 votes, 70% agreement).

## Check on the five stored fresh runs

There are no human identity labels for these clips. The check below cuts stored tracks of 60+ observations into pieces separated by removed gaps of 0.2–2.5 seconds and counts how many cuts the linker rejoins. It measures agreement with the online tracker's own IDs, not identity accuracy. Command: `node scripts/python.mjs scripts/link_tracks.py SOURCE RUN_DIR --evaluate`.

| Clip | Tracks before → after | Artificial cuts | Rejoined | Joined two tracks that coexisted | Unverifiable links |
|---|---|---|---|---|---|
| dfl_test_1 | 45 → 42 | 214 | 161 (75%) | 4 | 3 |
| dfl_test_10 | 112 → 89 | 212 | 126 (59%) | 9 | 18 |
| dfl_test_11 | 64 → 56 | 224 | 171 (76%) | 4 | 6 |
| untitled_design | 374 → 332 | 74 | 44 (59%) | 2 | 42 |
| input_vid | 514 → 433 | 51 | 18 (35%) | 4 | 76 |

"Tracks before → after" is the effect on the real stored tracks. The other columns come from the artificially cut copy.

## Learned embeddings from the detector backbone (opt-in)

`workers/video/reid.py` embeds person crops with layers 2 and 4 of the football detector's own backbone (no extra weights), centred on the clip mean. Set `TRACK_EMBEDDINGS=detector` to use them; the default is off. On `dfl_test_10`, 36.5% of different same-kit players still fall within the distance that keeps 90% of true pairs, against 52.8% for kit-colour histograms. Uncentred or deeper-layer features were no better than colour.

Same cut-repair check, colour only → with embeddings (CPU, about twice the linking time):

| Clip | Rejoined | Joined two tracks that coexisted | Real tracks after |
|---|---|---|---|
| dfl_test_1 | 161 → 162 | 4 → 6 | 42 → 42 |
| dfl_test_10 | 126 → 133 | 9 → 5 | 89 → 92 |
| dfl_test_11 | 171 → 173 | 4 → 3 | 56 → 56 |
| untitled_design | 44 → 45 | 2 → 1 | 332 → 338 |
| input_vid | 18 → 27 | 4 → 5 | 433 → 433 |

The gain is marginal (520 → 540 rejoined, 23 → 20 wrong joins in total) and does not reduce real track counts. A detector backbone is trained to find players, not to tell them apart. A ReID model trained on sports identities is still needed; none is installed, and the commonly used sports OSNet checkpoint has no stated licence.

## Sports-trained OSNet ReID (local test only)

`OSNetEmbedder` in `workers/video/reid.py` loads OSNet x1.0 with the SportsMOT-trained checkpoint `sports_model.pth.tar-60` (30,393,613 bytes, SHA-256 `8d5b2fd8…9c5fbd`) taken from the MIT-licensed `sjc042/gta-link` repository, with the model definition from torchreid's `osnet.py`. Both sit under the ignored `data/qa/reid-test/`. The checkpoint's own licence and the SportsMOT dataset terms are unverified, so this is not cleared for the hosted product and is not wired into the pipeline. It is loaded with PyTorch's restricted `weights_only` loader. Run it with `scripts/link_tracks.py SOURCE RUN_DIR --osnet WEIGHTS OSNET_PY --evaluate`.

Separation on `dfl_test_10` (share of different same-kit players that fall within the distance keeping 90% of same-player pairs): kit colour 52.8%, detector backbone 36.5%, sports OSNet 8.9%.

Cut-repair check, 3-second gap limit, colour only → sports OSNet (CPU):

| Clip | Rejoined | Joined two tracks that coexisted | Real tracks after |
|---|---|---|---|
| dfl_test_1 | 161 → 168 | 4 → 4 | 42 → 41 |
| dfl_test_10 | 126 → 149 | 9 → 11 | 89 → 90 |
| dfl_test_11 | 171 → 180 | 4 → 3 | 56 → 57 |
| untitled_design | 44 → 49 | 2 → 4 | 332 → 338 |
| input_vid | 18 → 24 | 4 → 4 | 433 → 435 |

Total rejoined 520 → 570 with wrong joins 23 → 26. Real track counts do not fall. Raising the gap limit to 8 seconds made results worse on every clip (427 rejoined, 52 wrong joins), because the position rule loosens with the gap. The fragments that remain are either too short to link (median 4–6 observations on two clips) or separated by long absences, and an 8.9% same-kit confusion rate is too high to merge on appearance alone among roughly ten same-kit candidates.

## Limitations

- Kit colours cannot tell team-mates apart, so links rest on short gaps and image position. Learned appearance embeddings (ReID) are the next step and are not implemented.
- The reduction on real tracks is modest (8–21%). Players who leave the view and return stay separate identities.
- `untitled_design` and `input_vid` consist mostly of very short tracks; most of their links cannot be checked and the wrong-join rate there is unknown.
- Identity switches inside one online track are not split.
- Camera zoom is not compensated. Nothing was run on GPU or on Modal.
- The annotated video still paints original tracker IDs, and `track-links.json` is not in the Modal artifact list.

## Live tracker fix: expanded-box matching

Two clips (`untitled_design`, `input_vid`) produced hundreds of tracks only 4–6 observations long. Replaying the detector's raw output through the tracker with one setting changed at a time showed the cause: the tracker matches by box overlap, and for players a few dozen pixels tall under a fast-moving camera the overlap between the predicted and detected box collapses, so the track is dropped and a new ID starts. Disabling camera compensation, lowering the new-track threshold, a longer lost-track buffer and ByteTrack did not help; looser matching did.

`workers/video/expanded_iou.py` grows both boxes by half their size on each side before measuring overlap (the Deep-EIoU idea). Only matching is affected; reported boxes are unchanged. It is on by default (`TRACK_IOU_EXPANSION=0.5`; `0` restores the previous behaviour).

Replay of freshly captured raw detections (same model weights, CPU, 1280 px), before → after:

| Clip | Tracks | Median track length | Detections with no track | Identities after offline linking | Jumps over one player height |
|---|---|---|---|---|---|
| dfl_test_1 | 45 → 39 | 293 → 681 | 74 → 52 | 42 → 38 | 0 → 0 |
| dfl_test_10 | 111 → 91 | 55 → 68 | 247 → 211 | 88 → 77 | 0 → 0 |
| dfl_test_11 | 63 → 59 | 80 → 110 | 123 → 116 | 55 → 53 | 0 → 0 |
| untitled_design | 393 → 281 | 4 → 6 | 1184 → 795 | 353 → 255 | 3 → 8 |
| input_vid | 518 → 358 | 6 → 9 | 1943 → 1277 | 435 → 310 | 0 → 6 |

The last column counts adjacent-frame jumps inside one track larger than a player height after camera compensation; it is a warning sign of identity switches, not a count of them. Turning score fusion off or raising the match threshold as well cut tracks further (down to 225 and 179 on the two hard clips) but raised these jumps to 43–64, so those settings were not adopted.

These are continuity diagnostics without human identity labels. Roughly 22 players and officials are on screen in each clip, so 255–310 identities on the two hard clips is still far from solved. Not run on GPU or Modal. Harness and captured detections: `data/qa/tracker-diagnosis/` (ignored by git).

## Analytics and dashboard use linked identities

Measurements and insights are now computed after linking, from rows where the linked identity replaces the tracker ID and a team voted across all fragments fills missing team labels (`linked_view` in `track_linking.py`). Such outputs carry `track_identity: "linked_track_id"`; runs without linking are computed exactly as before. The dashboard's track explorer lists linked identities with their fragment count and kit group and states how many tracker tracks they were joined from.

Checked on a local replay of `dfl_test_10` through the full pipeline (stored detections, CPU): 91 tracker tracks, 77 linked identities, 74 player identities in measurements, shown correctly in the Analytics view of the locally served app with no console errors. The hosted app was not updated.
