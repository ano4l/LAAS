# Football analysis research and implementation

Research date: 24 October 2026. Target use: upload a complete clip, inspect detections, review football events, and obtain traceable analytics. Intended production camera: fixed wide-angle full pitch. The five available Kaggle clips include broadcast camera movement and are development examples, not a representative fixed-camera accuracy benchmark.

## Decisions from the research

| Problem observed in this system | Primary evidence | Decision implemented | Remaining evidence needed |
| --- | --- | --- | --- |
| Hundreds of short person tracks; detector confidence filtering removed weak boxes before tracking | [ByteTrack paper](https://arxiv.org/abs/2110.06864) associates lower-score detections with established tracks | Pass person boxes down to 0.10 into association. Established tracks can recover low-score detections; unmatched low-score detections cannot start reported tracks. New-track threshold 0.50, high-score threshold 0.45 | Labelled identity sequences, IDF1/HOTA and false positive checks |
| Broadcast panning breaks image-space motion assumptions | [BoT-SORT paper](https://arxiv.org/abs/2206.14651), [official implementation](https://github.com/NirAharon/BoT-SORT), [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) | Enable installed BoT-SORT with sparse optical-flow camera compensation. Preserve ByteTrack as a configurable alternative. Re-identification is **disabled**; this is not the complete paper configuration | Compare both on held-out fixed-camera and broadcast sequences, including runtime and identity switches |
| Tiny balls disappear when the full image is resized | [SAHI paper](https://arxiv.org/abs/2202.06934), [official code](https://github.com/Small-Object-Detection/SAHI) establish slicing as a small-object strategy | Add a bounded, recent-ball search crop to the existing secondary detector, up to 8fps; retain full-frame fallback around 2fps. Crop predictions are translated back into source pixels. Exhaustive tiling remains optional | Independent ball labels; precision/recall by apparent size, blur, occlusion and false alarms |
| Ball prediction can lock onto the wrong location; airborne balls can lie above green pitch pixels | [TrackNet](https://arxiv.org/abs/1907.03698) and [TrackNetV4](https://arxiv.org/abs/2409.14543) motivate temporal evidence | Keep global reacquisition, expire stale search positions, never turn crop prediction into a detection. Retain aerial/outside-ground candidates with a selection penalty instead of deleting them | A football-trained temporal ball model; racket-sport weights are not validated football weights |
| Colour clustering splits one kit by brightness or selects rare colours | [SoccerNet game-state reconstruction](https://github.com/SoccerNet/sn-gamestate) and [PRTReID](https://arxiv.org/abs/2401.09942) separate team, role, appearance and identity | Reduce LAB lightness weight; initialise from supported colour populations after multiple tracks/frames. Publish anonymous scene-local kit groups and fit diagnostics | Reviewed team/role labels; learned football appearance features, jersey OCR and roster mapping |
| Pixels and camera movement were unsuitable for metres/speed | [SoccerNet calibration](https://github.com/SoccerNet/sn-calibration), [task definition](https://www.soccer-net.org/tasks/camera-calibration), [OpenCV homography documentation](https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html) | Add explicit fixed-camera landmark calibration, two held-out landmarks, source binding and background-motion rejection. Only accepted observed player foot-point transitions produce physical measurements | Actual pitch dimensions and six or more landmarks for a source video, plus external position/speed validation |
| Ball proximity was insufficient evidence of completed passes, shots or possession | [SoccerNet ball-action task](https://www.soccer-net.org/tasks/ball-action-spotting) treats event timing/classification as a separate learned task | Add a timestamped human event review API and dashboard. Only saved approved records affect football-event summaries. Model proximity candidates stay unreviewed | Annotated event sequences and an independently evaluated action model; spotting alone does not establish pass completion |
| A plausible-looking dashboard could hide missing observations | [HOTA paper](https://arxiv.org/abs/2009.07736) separates detection and association performance; [SoccerNet GSR](https://github.com/SoccerNet/sn-gamestate) evaluates pitch position and identity together | Show observed coverage, unknown outcomes, reviewed duration and provenance. Do not equate fewer tracks or more ball boxes with accuracy | Match-separated training/tuning/test data, consistent labels and reproducible threshold selection |
| xG/xT cannot be recovered from raw box counts | [Hudl/StatsBomb open data](https://github.com/hudl/open-data), [socceraction xT documentation](https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/xT.html) use structured event data and trained models | Keep xG/xT unavailable. Establish event and calibration layers first; no arbitrary probability formula | Reviewed shot/action locations, outcomes, model training and probability calibration on held-out matches |

These are engineering decisions informed by the sources. We have not reproduced their published benchmark scores. No pretrained ReID, temporal ball or action-spotting model was added in this change.

## What the code now does

### Person detection and association

`workers/video/observation_refiner.py` accepts both `TRACKER_BACKEND=botsort` and `bytetrack`. Confirmed tracks can use lower-confidence observations without opening a stream of new low-confidence tracks. Existing person NMS, pitch-region checks and scene-cut resets remain. The manifest records the actual tracker configuration, thresholds and low-confidence recovery counts. The legacy `engine=yolo_bytetrack` wire value remains for older clients; the dashboard displays the actual tracker field.

BoT-SORT is the default for the current moving-camera examples. ByteTrack remains useful to benchmark on the intended fixed-camera input. Camera compensation adds CPU work and does not supply player identity across cuts or long occlusions. Sparse optical flow can itself fail on blur, crowds, cuts or weak background texture.

### Ball search and uncertainty

`workers/video/detector.py` uses primary ball predictions first. When no primary candidate reaches 0.30, it can run the existing COCO secondary ball detector. A position no older than 0.35 seconds selects a 192–640 pixel search window, predicted using recent observed velocity. Global search has scheduling priority when global and crop intervals coincide. The next global search remains possible even while crop search is active.

The crop only selects where the network looks; it is never exported as an observed ball location. Each secondary box records its model role and search region. Missing/ambiguous positions stay missing. A plausible green-ground region remains useful context, but aerial balls are retained and marked rather than forcibly discarded. This can also retain false alarms, which is why labelled precision matters.

### Reviewable football analytics

`workers/video/event_review.py`, both API implementations, and `apps/web/src/ReviewPanel.tsx` add source-bound persisted review records. The browser seeks to event time, supports draft/approved status, team names, optional player labels, reviewed coverage, and possession intervals. No GPU inference is needed for review or recalculation.

Supported event types: pass, shot, tackle, interception, dribble and duel. A goal is a shot outcome. Summaries expose recorded passes, known pass outcomes/completion, shots/on-target/goals, won tackles, interceptions, successful dribbles and won duels, plus manually attributed player event counts. These are counts of approved annotations, not assertions that every event in the clip was found.

Pass completion uses `completed / (completed + incomplete)`. Unknown outcomes are excluded and their coverage is shown. Possession uses explicitly approved, non-overlapping time intervals; the home/away share denominator contains only known attributed seconds. Unknown seconds and coverage against the entire source remain visible. Reviewed coverage is the union of user-marked intervals; isolated events do not imply full-clip review.

Drafts never enter authoritative summaries. A stale revision returns HTTP 409 instead of overwriting another save. Invalid/out-of-range times, duplicate IDs, invalid outcomes and overlapping approved possession intervals are rejected. Records use atomic replacement and are bound to the run and source SHA256. The hosted API requires the existing workspace access key. Individual analyst accounts/audit identities are still outside this shared-workspace prototype.

Also fixed a proximity-candidate error found while inspecting real results: a player reappearing after a short ball gap could be proposed as passing to their own temporary track. Transfer candidates now require distinct track IDs. They still require review and never count automatically as completed passes.

### Physical-measurement foundation

`scripts/calibrate_tracking.py` is a local CLI foundation, not a dashboard calibration tool. It requires a known fixed camera, measured pitch dimensions, source SHA256, at least six landmarks and successful held-out/reprojection checks. A background-motion check can reject the whole calibration. Unverifiable stability also rejects it.

Accepted output covers observed player foot-point transitions only. Gaps longer than 0.2 seconds and steps above 15m/s are excluded. Temporary track identity and box-footpoint error remain; accumulated detection jitter can bias distance. A homography also does not correct wide-angle lens distortion: target cameras may need measured intrinsics/undistortion and re-annotation in the corrected coordinate system, supported by [OpenCV calibration tools](https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html). That correction is not implemented here. The output is not verified full-match workload, peak speed, injury risk, or ball-ground position. No real clip in this report has been supplied with a verified calibration, so none receives claimed metre/speed statistics.

## Local comparison on all five supplied clips

Total: 138.337 seconds, 3,512 source frames. All five revised exports were decoded to verify the complete source frame count. Source hashes and 100% analysed-frame coverage were checked.

The comparison reuses **the same original primary model observations**, plus the secondary ball observations already computed during the preceding refinement pass. It performs new filtering, optical-flow association, colour grouping, measurements and encoding. It does **not** run the new crop detector over every frame of all five videos. This isolates postprocessing changes and avoids exceeding the existing cloud GPU budget.

| Clip | Duration / frames | Previous refinement temporary player tracks | New BoT-SORT tracks | New mean visible player detections | Selected-ball candidate frame coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| input_vid | 31.370s / 753 | 704 | 476 | 14.15 | 31.34% |
| untitled_design | 16.967s / 509 | 448 | 336 | 19.86 | 22.40% |
| dfl_test_1 | 30s / 750 | 50 | 46 | 19.43 | 30.80% |
| dfl_test_10 | 30s / 750 | 160 | 107 | 22.30 | 57.07% |
| dfl_test_11 | 30s / 750 | 48 | 49 | 20.28 | 40.67% |

The last clip regresses in temporary track count. The first two still have severe fragmentation. Lower counts can indicate continuity but can also hide merged identities, so these are diagnostics, **not measured accuracy gains**. Candidate coverage is not ball recall or possession coverage. Kit groups remain unverified.

An additional association ablation used original primary observations only, holding the new geometry/colour pipeline constant:

| Clip | Low-score ByteTrack tracks | Low-score BoT-SORT tracks | ByteTrack CPU processing seconds | BoT-SORT CPU processing seconds |
| --- | ---: | ---: | ---: | ---: |
| input_vid | 554 | 476 | 11.78 | 21.09 |
| untitled_design | 343 | 336 | 14.36 | 28.73 |
| dfl_test_1 | 48 | 46 | 18.00 | 34.45 |
| dfl_test_10 | 132 | 107 | 17.72 | 43.16 |
| dfl_test_11 | 46 | 49 | 20.06 | 32.68 |

These timings include local decoding and refinement, exclude neural inference/encoding, and are hardware/load dependent. BoT-SORT takes roughly 1.6–2.4 times as long in this ablation. It wins this proxy in four of five clips, not universally.

### Fresh neural-inference smoke test

Both actual models ran on original frames 118–127 of `input_vid.mp4` on local CPU: primary inference at 1280, secondary at 640. Four secondary searches ran, three in recent-ball crops. Three retained secondary observations occurred at source frames 118, 121 and 127; the latter two had crop provenance. All coordinates were mapped back inside the original 848×480 image. The test also recovered 18 low-confidence person observations through association. Ten frames demonstrate execution and provenance, not precision/recall or complete trajectory quality.

Machine-readable evidence is under `data/qa/research-2026-10-24`: `ablation.json`, `fresh-roi-smoke.json`, `comparison.csv`, `comparison.json`, `verification.json` and per-run output directories. Full refined videos and CSV/JSON analytics are bundled in `research-videos-and-analytics.zip`.

## Validation and deployment boundaries

All 47 automated tests passed. A subsequent focused 11-test run also passed after tightening global-search scheduling and ensuring a higher ball-confidence setting cannot suppress low-score person recovery. Checks cover low-score recovery without low-score track creation; ball crop bounds/staleness; duplicate/missed ball scoring; streaming beyond the old truncation cap; review approval, denominators, source binding, validation, authorization, atomic persistence and stale-save conflicts; calibration held-out geometry and camera rejection. UI evaluation covers saved versus draft state, video seeking, persistence/errors, per-run unsaved edits and responsive layout.

The hosted API was checked against all five published `-research.mp4` runs: authenticated review reads returned 200, missing-key requests returned 401, and signed video range reads returned 206. An empty review (no invented football observations) was saved and retrieved at revision 1; repeating the stale revision returned 409. The existing sample still reports its complete 30 seconds/750 frames. The updated worker was deployed without submitting new GPU inference.

The frontend production build passed. Independent same-provider UI evaluation passed both attempts, covering restoration of per-run records and unfinished forms, saved-only JSON export, controlled 409 retention and explicit discard/reload, 375px layout without page overflow and scoped accessibility checks. A final follow-up verified that tab-close warnings cover unsaved work on background runs and survive navigating away from the review panel; warnings clear after all drafts are clean. QA drafts were removed, without adding fake approved events to real footage.

Deployment surfaced existing npm build-tool advisories. Compatible lockfile updates changed two root and twelve frontend packages; both npm audit runs then reported zero known vulnerabilities. The frontend rebuilt successfully with Vite 7.3.7. This is dependency-audit evidence, not a general security certification.

The frontend is deployed at https://laas-football.vercel.app (`dpl_6jZeg5vxJBGa4enZnQsUghGBavRe`), with API and worker on Modal. Production browser checks loaded the research-run Library, saved-review panel and complete 30-second video at readyState 4 / 1280×720 without page overflow. The published JavaScript contains the intended hosted API and no workspace access key. All five revised clips appear as `*-research.mp4`; opening stored results does not submit inference.

See the current entry in `PROGRESS.md` for final test count, browser verdict and deployment verification. The new worker can be deployed without invoking it. The daily GPU attempt cap was already exhausted by the earlier five real runs; this pass does not increase or reset it. New CUDA execution of crop search and the updated tracker remains a subsequent real upload acceptance check when the normal quota permits it. Local fresh inference, full replay processing, cloud API checks and browser checks are different evidence levels.

## Next accuracy work, in order

1. **Create an independent evaluation set.** Label ball boxes/empty frames, player identity sequences, team affiliation and event timestamps on separate matches. Include far-side tiny balls, occlusion, replays, cuts, night footage and moving/fixed cameras. Keep all clips from one match in one split. The existing ball review/scoring tooling is ready; predictions must not become ground truth.
2. **Train a dedicated football ball detector.** Use full-resolution crops and hard negatives (heads, socks, lines, logos, spectators). Compare existing detector, cropped detector and a football-trained temporal model on precision/recall by ball size and false alarms per minute. TrackNet-style motion is a research direction, not a pretrained football capability we have installed.
3. **Resolve identities and teams.** Evaluate learned role/team/ReID representations, jersey OCR and roster constraints. Use labelled HOTA/IDF1 and identity-switch evidence before linking workloads to named players. Inspect model/data licences and weight availability first.
4. **Calibrate the intended fixed camera.** Supply measured dimensions/landmarks; quantify lens distortion for wide-angle input and compare projected points/movement against independent reference measurements. For broadcast footage use per-view calibration rather than applying one fixed homography through pans/zooms.
5. **Train event proposals from the saved review records.** Split by match and score event timing/class at explicit tolerances. Confirm team/player/outcome separately; a detected pass timestamp alone cannot prove completion. Keep review overrides.
6. **Add structured value models.** Extend reviewed events with calibrated start/end locations, attack direction, body part and context before fitting xG/xT. Evaluate probability calibration and downstream attribution. Do not infer injury risk from these short video clips.
7. **Scale only after accuracy measurement.** Long videos need chunk checkpoints, track stitching, durable queues, media storage and cancellation; live analysis also needs bounded latency/backpressure. Current hosted admission remains 120 seconds/250MiB. No 45-minute or live performance claim follows from this five-clip experiment.

### Concrete blockers

The main blockers are independent labels, representative fixed-camera footage, measured pitch references, suitable football model weights/training and compute acceptance under the current budget. Some [SoccerNet datasets](https://www.soccer-net.org/data) require an account and dataset-specific agreement/NDA. No agreement was accepted and no gated dataset was downloaded during this work. The public [SportsMOT paper](https://arxiv.org/abs/2304.05170) and SoccerNet benchmarks are useful evaluation references; their publication is not blanket permission to republish footage or pretrained models. The GSR repository declares GPL-3.0; no wholesale dependency/code import from it was performed.

## Source map

- Association: [ByteTrack authors' code](https://github.com/FoundationVision/ByteTrack), [paper](https://arxiv.org/abs/2110.06864); [BoT-SORT authors' code](https://github.com/NirAharon/BoT-SORT), [paper](https://arxiv.org/abs/2206.14651).
- Small-object/temporal ball research: [SAHI](https://github.com/Small-Object-Detection/SAHI), [paper](https://arxiv.org/abs/2202.06934); [TrackNet](https://arxiv.org/abs/1907.03698); [TrackNetV4](https://arxiv.org/abs/2409.14543).
- Football reconstruction: [SoccerNet GSR](https://github.com/SoccerNet/sn-gamestate), [PRTReID](https://arxiv.org/abs/2401.09942), [calibration devkit](https://github.com/SoccerNet/sn-calibration).
- Evaluation and events: [HOTA](https://arxiv.org/abs/2009.07736), [SportsMOT](https://arxiv.org/abs/2304.05170), [SoccerNet ball-action spotting](https://www.soccer-net.org/tasks/ball-action-spotting).
- Structured analytics: [Hudl/StatsBomb open event data](https://github.com/hudl/open-data), [socceraction xT](https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/xT.html).
