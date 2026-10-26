# Accuracy before hosting

## Research update, 24 October 2026

The [research report](docs/research/FOOTBALL_SYSTEM_RESEARCH_2026-10-24.md) connects primary papers and official implementations to concrete changes: restored low-score person association, configurable BoT-SORT camera compensation, bounded ball crop search with global reacquisition, revised colour grouping, persisted human event review/aggregation and an explicit fixed-camera calibration CLI. Fewer temporary IDs are a diagnostic, not identity accuracy. No independently labelled precision/recall or HOTA score is claimed.

Five full clips were reprocessed from existing real primary/secondary observations and verified at all 3,512 exported frames. New temporary player tracks: 476, 336, 46, 107 and 49. The fifth clip has one more track than the preceding refinement. Fresh local neural inference on ten source frames separately exercised the crop detector (three ROI searches). The full five-clip comparison did not run fresh crop inference. Reviewed-event metrics require saved approved annotations; raw detections never become approved events automatically. Actual physical measurements still require source-specific measured landmarks, dimensions and successful fixed-camera checks; see [calibration instructions](docs/research/FIXED_CAMERA_CALIBRATION.md).

Target camera: fixed wide-angle full-pitch footage (user confirmed). The existing short broadcast example is only a plumbing/smoke benchmark, not a representative accuracy evaluation. Preserve original source resolution; test full-frame inference against overlapping crops on footage from the intended fixed camera. Camera calibration and player identity continuity are separate validation tasks. No fixed-camera footage has been evaluated in this pass.

## Repository reference footage

The user identified `probablyabdullah/Football-Tracking-with-YOLOv5-Bytetrack` as the test source. Its notebook at commit `44ce3a32a82c5f1ef34944608f19071e408187a7` downloads the DFL Bundesliga Kaggle competition clips and selects `clips/08fd33_4.mp4` (cells 15, 20 and 23). The Git repository itself contains the notebook and screenshot links; it does not bundle the MP4. The workspace already has a file with that name at `references/Soccer-Analytics/input_videos/08fd33_4.mp4`: 30 seconds, 750 frames, 25fps, 1920x1080. Source SHA256 is recorded in prepared labels and new run manifests. The notebook supplies no checksum for independently verifying that local asset against its original download.

Prepared 50 original-resolution unreviewed reference images and `review.html` under `data/qa/repo-reference-review`. This is broadcast footage, consistent with the repository's stated TV-broadcast input, and remains separate from future fixed full-pitch evaluation. A full every-frame 1280px run was submitted through the real upload API as job `f61be244d05c4377987d89a9feb736a9`; its persisted status can be checked in the app or at `/analysis-jobs/f61be244d05c4377987d89a9feb736a9`. Submission is not evidence of successful completion or detection accuracy.

The accurate profile analyses every source frame at 1280 pixels. This improves available detail and temporal sampling, but is not proof of higher precision or recall. Local CPU processing is slower. Ball detections remain candidates; no missing position is fabricated. Person track IDs are temporary, not verified player identities. Camera cuts, occlusions and similar shirts still cause identity switches.

## Review and measure

Prepare original-resolution review images and an offline annotation page:

```powershell
node scripts/python.mjs scripts/evaluate_ball.py prepare path/to/video.mp4 data/qa/review-set --frames 50
```

Open `data/qa/review-set/review.html`. Use full-resolution zoom for tiny balls. Draw tight boxes around all visible match balls, then mark each frame reviewed. Reviewed empty frames measure false alarms. Leave ambiguous frames unreviewed; exclusions are reported. Download labels.json and replace the prepared unreviewed file with it. The page holds edits in memory only; download before closing. Do not copy model predictions into the labels.

Run the same source video and score its run directory:

```powershell
node scripts/python.mjs scripts/analyze_video.py path/to/video.mp4 --profile accurate
node scripts/python.mjs scripts/evaluate_ball.py score data/qa/review-set/labels.json data/storage/outputs/RUN_ID
```

Scoring requires matching source SHA256 and image dimensions. At default IoU 0.5 and confidence 0.1, it reports ball-box precision, recall, F1, false positives, missed boxes and per-frame errors. Duplicate boxes count as false positives. Labels without detections count as misses, including unsampled frames; compare configurations on identical frame sets. This is fixed-threshold box evaluation, not mAP or a tracking identity benchmark. Reviewed-frame selection and label consistency affect results. The labels hash and model hash are recorded in the report.

Compare optional crops with `--ball-tile-size 640`. Crops preserve small-object detail but can add false alarms and substantial runtime. Keep them off by default until measured. Do not tune confidence on your final test set.

## Evidence still required

Use several representative matches from the intended camera type, with day/night footage, motion blur, distant balls, occlusion, camera cuts, replays and no-ball frames. Keep entire matches separate between training, tuning and final evaluation. Uniformly sampled review frames are a starting point; add difficult sequences separately and report their results separately.

No reviewed target dataset or specialised ball weights have been supplied. Maximum achievable accuracy cannot yet be established. Next model changes should be driven by reviewed misses and false alarms: fine-tune a dedicated ball model on training matches, evaluate on held-out matches, and validate player identity continuity separately. Possession, passes, speed and tactical metrics need their own labelled validation and camera calibration before being enabled.

## Local smoke benchmark, 20 October 2026

On the existing five-second, 1920x1080, 25fps broadcast clip, accurate mode decoded and analysed all 125 frames at 1280px, producing 78 ball boxes across 70 frames in 84.81 CPU seconds. Run: `9be7573eef4d48e894d2fea06f25416d`. It produced 98 temporary person track IDs, showing that identity fragmentation is unresolved; these are not 98 distinct players.

Experimental 640px crops with preview sampling analysed 25 frames, producing candidates in 17 frames in 70.19 CPU seconds. Run: `e145d0b096164a0788eeb49b042e9174`. These different sample counts are not an accuracy comparison. Neither run has reviewed ground truth. Crops remain disabled by default. Both runs are under `data/qa/accuracy-runs`.

Validation: 13 automated tests passed, production frontend build passed, and the offline review page was checked in a browser for drawing, review state, navigation and download action. Prepared 25 unreviewed reference images in `data/qa/ball-review-20261020`. Do not use the browser QA drawing as a real label; the page was cleared after testing.
# Refinement update — 24 October 2026

The production worker now applies human-box NMS/fragment containment before ByteTrack, a 0.45 human confidence threshold, a fail-open plausible-field filter, hard-cut identity resets and conservative observed-ball continuity selection. Optional secondary sports-ball inference runs at up to 2fps at 640px when primary confidence is below 0.3; the deployed Modal image includes the existing COCO yolov8m weights for this purpose. These heuristics can trade recall for fewer false positives, and geometry can miss valid out-of-play/aerial balls.

New metrics include anonymous per-scene jersey-colour groups, supported near-ball observation seconds/share, image-relative shape and review candidates. They do not establish actual team identity, possession, completed passes, distance, speed or xG. Five complete Kaggle clips were refined from their existing primary detection files plus new sampled secondary inference. Source hashes and all 3,512 exported frames were checked. The full comparison, provenance and remaining limitations are in [the refinement report](docs/testing/ACCURACY_REFINEMENTS_2026-10-24.md). No precision/recall or identity-accuracy score is claimed from fewer boxes or fewer tracks.
