# Automatic football pitch calibration (homography / camera) and the physical metrics it unlocks — state as of Nov 2026

Conventions: **[V]** = number or fact read from a primary source (paper, official README, LICENSE file, GitHub API) during this research. **[S]** = from a secondary summary or search snippet, not checked against the primary. **[C]** = author/vendor claim with no independent check. Nothing here was run on an L4; all L4 statements are inferences.

Metric note: SoccerNet "Acc@5" and "JaC@5" are the same family of metric (share of pitch segments reprojected within 5 px); final score = completeness x Acc@5/JaC@5.

## 1. Top methods of the SoccerNet Camera Calibration and GSR challenges 2023-2026; public code/weights; accuracy and fps

### Takeaway
The only standalone SoccerNet calibration challenge with a published leaderboard in this period is 2023 (winner Sportlight, final score 0.55); 2024 and 2025 folded calibration into Game State Reconstruction (GSR). The best open, runnable single-frame calibrator is PnLCalib (successor to NBJW, same author, actively maintained to Oct 2026; code GPL-2.0-or-later, weights CC BY 4.0), but it is slow per frame (164-439 ms on a 2080 Ti), so a 45-minute half needs keyframe calibration plus tracking/propagation, not per-frame calibration.

### Cited Findings

**SoccerNet Camera Calibration 2023 leaderboard [V]**
- Sportlight 0.55 final (Acc@5 73.22, completeness 75.59); Spiideo 0.53 (52.95 / 99.96); SAIVA_Calibration 0.53 (60.33 / 87.22); BPP 0.50 (69.12 / 72.54); ikapetan 0.43 (53.78 / 79.71); NASK 0.41 (53.01 / 77.81); MA & JT 0.41 (58.61 / 69.34); baseline 0.08 (13.54 / 61.54) — [SoccerNet 2023 Challenges Results](https://arxiv.org/html/2309.06006)
- Sportlight method: HRNetV2-w48 heatmaps for keypoints (line/ellipse intersections, tangent points, homography-projected points) plus lines; OpenCV calibration on point subsets; voting by reprojection error. No code link given in the results paper — [SoccerNet 2023 Challenges Results](https://arxiv.org/html/2309.06006)
- Spiideo method: DeepLabV3 segmentation into 66 pitch-part classes, then camera optimisation by differentiable rendering (SoftRasterizer, AdamW) from 20 k-means initial cameras; code public — [SoccerNet 2023 Challenges Results](https://arxiv.org/html/2309.06006); repo licence MIT, last push Aug 2023 — [Spiideo/soccersegcal](https://github.com/Spiideo/soccersegcal)
- NASK: Mask2Former (Swin-S) segmentation of lines/conics/goal parts, intersections, RANSAC homography, PnP refine — [SoccerNet 2023 Challenges Results](https://arxiv.org/html/2309.06006)
- Official SoccerNet baseline: JaC@5 11.7%, completeness 68%, final 7.96% — [SoccerNet/sn-calibration README](https://github.com/SoccerNet/sn-calibration)

**2024 and 2025: no standalone calibration task**
- The 2024 results paper lists ball action spotting, dense video captioning, multi-view foul recognition and GSR; no calibration leaderboard [S] — [SoccerNet 2024 Challenges Results](https://arxiv.org/html/2409.10587v1)
- 2025 tasks: ball action spotting, monocular depth, multi-view foul recognition, GSR [S] — [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182)

**GSR 2024 leaderboard (GS-HOTA / GS-DetA / GS-AssA) [V from winner's paper]**
- Constructor Tech 63.81 / 49.52 / 82.23; UPCxMobius 43.15 / 30.46 / 61.16; JAM 34.40 / 19.38 / 61.08; baseline 23.36 / 9.80 / 55.69 — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357)
- Constructor Tech calibration: 74 pitch keypoints (pitch-line x grass-line intersections), keypoint net at 480x270 input; a SegFormer-based network regresses camera position, pan, tilt, roll, FOV directly (trained on ~22k real images adapted from TVCalib annotations plus 40k synthetic images from a modified Google Research Football); brute-force refinement of parameters against keypoints; Savitzky-Golay temporal smoothing with corrections capped at +/-2 degrees and +/-2 m and a 15-frame (0.5 s) look-ahead delay — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357)
- Constructor Tech speed claim: up to 80 FPS on an RTX 3080 Ti laptop GPU with DeepStream + TensorRT FP16 [C]; code availability and licence not stated; the paper is internally inconsistent on the keypoint backbone (ResNet-18 in the body, ResNet-50 segmentation in Appendix B) — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357)

**GSR 2025 leaderboard (GS-HOTA) [V]**
- KIST-GSR 63.90; Constructor.Tech 63.81; lianyou 62.76; Playbox & MIXI 61.64; tyler_durden 57.08; SJTU 50.06; eidos.ai 46.24; UTokyo Football Lab 40.13; hjkim 36.90; baseline 29.01; 14 teams, 66 submissions — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Calibration components reported by teams: KIST-GSR used a multi-frame keypoint detection model; lianyou used radial-distortion correction + PnLCalib, reduced the camera to pan/tilt/roll/focal and smoothed over time; Playbox & MIXI extended BroadTrack (parameter-based, optical-flow propagation, initialised from several stable frames); the Leipzig team (13th) used U-Net pitch segmentation to pick reliable start frames, tracked NBJW keypoints with Lucas-Kanade and re-fit a RANSAC homography each frame, reporting up to 4% gain over the NBJW baseline — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- The report names NBJW, PnLCalib and BroadTrack as the commonly used calibration pipelines; it gives no code links for any GSR team [S/V] — [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182)

**PnLCalib / NBJW (Gutiérrez-Pérez & Agudo) — benchmark numbers [V, from the PnLCalib paper]**
- SN22-test-center (JaC@5 / CR / FS): PnLCalib SV+PnL 80.6 / 98.7 / 79.5; SV keypoint pipeline without PnL refinement 74.4 / 99.1 / 73.8; TVCalib 54.8 / 100 / 54.8; TVCalib(tau) 57.6 / 93.7 / 53.9 — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- SN23-test (multi-view, the 2023 challenge distribution): MV+PnL 78.7 / 78.4 / 61.8; MV without PnL 73.1 / 80.1 / 58.6 — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- WC14-test as full calibration: SV+PnL JaC@5 85.9, CR 100, FS 85.9; SV 77.6; TVCalib 39.9 — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- WC14 homography: SV+PnL IoU_part 97.0 mean / 98.2 median, IoU_whole 93.4 / 95.5, projection error 0.60 m mean / 0.42 m median. Without PnL: IoU_whole 92.4 / 94.8, projection 0.65 / 0.44 m — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- TS-WorldCup homography: SV+PnL IoU_part 98.6 / 98.9, IoU_whole 96.3 / 96.8, projection error 0.23 m mean / 0.20 m median; without PnL IoU_whole 94.6 / 95.8, projection 0.28 / 0.24 m — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- Comparators on WC14 IoU_whole mean: Nie 91.6, Chu (KpSFR) 91.2, Maglo 92.0, Shi 93.1, Oo 92.9; projection error mean: Nie 0.84 m, Chu 0.81 m, Maglo 0.74 m, Oo 0.65 m — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- Speed on SN23-test (RTX 2080 Ti for the network, Xeon Silver 4214 for optimisation): 164 ms/frame (1 RANSAC config, 1 keypoint subset, no PnL; FS 52.7), 178 ms (+PnL; FS 54.1), 273 ms (6 configs; FS 57.7), 439 ms (6 configs, 3 subsets, PnL; FS 61.8) — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- Architecture: modified HRNetV2-w48, separate networks for keypoints and line extremities, output at half input resolution; up to 30 line-intersection keypoints plus line-ellipse, tangent and central-axis/circle points. The auxiliary keypoint sets raise completeness on SN22-test-center from 86.1 to 98.1 (FS 63.9 to 74.4) — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- Camera model: full-perspective pinhole, zero skew, principal point at centre; K estimated first if at least 6 keypoints, otherwise K, R, t optimised jointly; homography path uses ground-plane keypoints only with RANSAC grid search + DLT + LM — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- PnLCalib repo: code GPL-2.0-or-later; weights CC BY 4.0; README notes base models were trained on SoccerNet-Calibration under a research agreement that has since expired and that weights contain no footage; weights SV_kp, SV_lines, WC14/TSWC/WorldPose fine-tunes, MV_kp, MV_lines in release v1.0.0; video inference via `inference.py --input_type video --pnl_refine`; March 2026 update added lens-distortion optimisation and the alpha weighting; accepted to Computer Vision and Image Understanding — [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib)
- PnLCalib repo: 113 stars, last push 2026-10-26 (GitHub API, queried 2026-11-14) — [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib)
- NBJW repo: LICENSE file is GPL v2; weights SV_kp, SV_lines, SV_FT_WC14_*, SV_FT_TSWC_*, MV_* in release v1.0.0; no speed figure and no weight licence stated in README; last push 2024-10-25 — [mguti97/No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles)

**TVCalib**
- Repo licence MIT, last push 2024-04-27 (GitHub API) — [MM4SPA/tvcalib](https://github.com/MM4SPA/tvcalib)
- On SoccerNet-gamestate video (HD): TVCalib JaC@5 19.88, JaC@10 50.42, mean reprojection error 12.4 px, CR 99.93% — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- TVCalib remains the default calibration in the sn-gamestate baseline; PnLCalib and NBJW were added as drop-in options (`pnlcalib` / `nbjw_calib` in `soccernet.yaml`) in July 2024 — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)

**BroadTrack (EVS, WACV 2025)**
- SoccerNet-gamestate test (49 sequences x 750 frames): BroadTrack JaC@5 56.88, JaC@10 79.79, MRE 5.02 px, CR 100%; NBJW 37.14 / 68.24 / 10.28 px / 93.67%; TVCalib 19.88 / 50.42 / 12.4 px / 99.93% — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- Uses NBJW keypoints + TVCalib line segmentation + RTMDet player masks + pyramidal Lucas-Kanade flow, optimised by Levenberg-Marquardt with Cauchy-robust losses and a tripod constraint — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- Speed: 16 fps at 1920x1080 on a server with two RTX 4090s, unoptimised — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- Licence: custom EVS agreement, "personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes" — [evs-broadcast/BroadTrack LICENSE](https://github.com/evs-broadcast/BroadTrack/blob/main/LICENSE)
- Input is a folder of `%06d.jpg` frames, run via Docker — [evs-broadcast/BroadTrack README](https://github.com/evs-broadcast/BroadTrack)

**2025-2026 successors**
- SoccerMaster (vision foundation model, arXiv 2512.11016) reports FS 70.1 vs 67.6 for PnLCalib on SN22-test-center [S] — [SoccerMaster](https://arxiv.org/pdf/2512.11016). Note this PnLCalib figure (67.6) differs from the 79.5 in the PnLCalib paper; the evaluation settings were not checked.
- Broadcast2Pitch (Oo et al., WACV 2026) targets GSR from unconstrained soccer video; contents could not be retrieved (HTTP 403) — [Broadcast2Pitch](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)

### Inferences
- For an L4 (roughly 2080 Ti-class or somewhat slower for HRNet-w48 FP32), PnLCalib at 164-439 ms/frame means a 45-minute half at 25 fps (67,500 frames) would take roughly 3-8 GPU-hours if calibrated every frame. Calibrating 1-2 keyframes per second and propagating in between (see section 3) brings this to minutes. This is an estimate, not a measurement.
- The licensing picture constrains choice more than accuracy does: PnLCalib/NBJW code is GPL-2 (copyleft; fine as a separate service, problematic if linked into closed-source distributed code), BroadTrack is non-commercial research only, TVCalib and Spiideo soccersegcal are MIT. For a commercial product, PnLCalib weights (CC BY 4.0) with a clean re-implementation of the solver, or TVCalib/soccersegcal, are the safer routes. The PnLCalib README's own caveat about the expired SoccerNet research agreement is a provenance flag worth a legal look.
- The large gap between single-image benchmark scores (JaC@5 ~74-80 on SN22-center) and video scores (NBJW JaC@5 37 on SoccerNet-gamestate) says single-frame methods degrade noticeably on real continuous footage; temporal modelling is where the 2025 GSR field gained most.
- Sportlight's challenge method and NBJW share the same design (HRNetV2-w48 heatmaps, geometric keypoints, voting on reprojection error); NBJW/PnLCalib is effectively the open implementation of that family.

### Gaps
- No SoccerNet 2026 challenge results paper was found by search; whether a 2026 calibration or GSR leaderboard exists is unconfirmed.
- KpSFR: the guessed repo path returned 404; licence, weights and speed were not verified. Its accuracy appears only as "Chu" in the PnLCalib comparison table.
- TVCalib per-frame speed, and any fps for NBJW, were not found in primary sources.
- No public code was found for Sportlight, SAIVA, Constructor Tech or KIST-GSR.
- No published L4 timings for any of these methods.

## 2. Roboflow sports pitch keypoint detection (YOLOv8-pose, 32 keypoints, ViewTransformer)

### Takeaway
Roboflow's model is a convenient, fast homography-only baseline trained on Bundesliga broadcast frames; it has no published benchmark on SoccerNet / WC14 / TS-WorldCup, no temporal handling, and its YOLOv8 weights carry AGPL-3.0 obligations when self-hosted. Treat it as a quick start or a fine-tuning template, not as a research-grade calibrator.

### Cited Findings
- Model: YOLOv8x-pose fine-tuned for 500 epochs, 640 px input (images stretched to 640x640, which the author found better than letterboxing), mosaic disabled, trained on an A100 with batch 48; hosted as `football-field-detection-f07vi/14` — [Roboflow blog](https://roboflow.com/blog/camera-calibration-sports-computer-vision)
- 32 pitch landmarks, all emitted every frame; invisible ones get low confidence. Pipeline: inference at 0.3, keep keypoints with confidence > 0.5, `cv2.findHomography` from kept points to `SoccerPitchConfiguration` vertices, then `cv2.perspectiveTransform` on player foot points; needs at least 4 visible points — [Roboflow blog](https://roboflow.com/blog/camera-calibration-sports-computer-vision)
- Training data: frames sampled once per second from the DFL Bundesliga Data Shootout Kaggle videos, manually annotated; dataset size not stated; no numeric accuracy given, no smoothing advice given — [Roboflow blog](https://roboflow.com/blog/camera-calibration-sports-computer-vision)
- Licensing per Roboflow: YOLOv8 is AGPL-3.0; use via Roboflow cloud includes a commercial licence; self-hosting requires an Enterprise add-on or open-sourcing the application under AGPL-3.0 — [Roboflow blog](https://roboflow.com/blog/camera-calibration-sports-computer-vision)
- `roboflow/sports` code is MIT (5.5k stars, last push Aug 2026); the README lists camera calibration as an open challenge and does not state dataset licences — [roboflow/sports](https://github.com/roboflow/sports)
- A third-party blog reports mAP 0.99 for a 32-keypoint YOLOv8 pitch model trained on "hundreds of images" [S, unverified, and in-distribution only] — [zzsi blog](https://zzsi.github.io/blog/posts/20240915-soccer-tracking/index.html)

### Inferences
- A model trained on a few hundred Bundesliga main-camera frames should be expected to generalise poorly to amateur footage (faded or missing lines, different camera heights, wide-angle lens distortion, non-standard pitch dimensions). No source tested this; it follows from the training distribution and from the absence of any distortion model.
- A plain homography from 4+ points with no RANSAC inlier check, no line evidence and no temporal prior will jitter and occasionally flip in midfield views. The Roboflow pipeline as published has no failure detection.
- The useful part to reuse is the recipe: fine-tune a pose model on your own annotated frames from the target camera type. For fixed elevated tactical cameras, a few hundred annotated frames per venue type is plausibly enough because the view varies little. To avoid AGPL, the same recipe can be run on a permissively licensed keypoint/pose architecture.

### Gaps
- No head-to-head of the Roboflow model against NBJW/PnLCalib on any public benchmark was found.
- Dataset size and licence of `football-field-detection-f07vi` not confirmed.
- No evidence on amateur or wide-angle generalisation in either direction.

## 3. Homography vs full camera model; few visible lines; temporal filtering; shot cuts and replays

### Takeaway
A homography is sufficient for ground-plane positions (player feet), which is all that speed, distance, heatmaps, formations and pitch control need. A full camera model (pan/tilt/roll/focal + radial distortion, ideally with a fixed-position/tripod constraint) is what makes midfield views and temporal consistency tractable, because it has far fewer free parameters per frame. Radial distortion is the single biggest accuracy term on broadcast video.

### Cited Findings
- BroadTrack ablation on SoccerNet-gamestate: adding one radial distortion coefficient k1 raises JaC@5 from 42.09 to 54.10 and cuts mean reprojection error from 5.74 to 5.04 px; optical flow and tripod constraint add a further ~1-3 points (to 56.88) and the authors argue their main benefit is smoothness, which single-frame metrics do not capture — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- BroadTrack camera model: single focal length, square pixels, centred principal point, k1 only; tripod model with a fixed pan/tilt pivot T for the whole game and a focal point that moves along the optical axis with zoom — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- BroadTrack can (re)initialise from only two point correspondences once the tripod position is known (focal from Gedikli's equations, pan/tilt refined iteratively, roll zero); with more points it uses RANSAC — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- BroadTrack confidence = Jaccard index between the segmented pitch-marking mask and the projected pitch template; re-initialise when confidence < 0.5; on SoccerNet-gamestate this fired 33 times over 49 sequences with tracking lost for 15 frames on average; optical flow is discarded when the previous estimate is unreliable; flow points off the pitch or near detected players are removed — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- BroadTrack does not describe shot-cut handling; its long-run test is a single 20-minute main-camera Bundesliga clip assessed only visually — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- Two-point re-initialisation on SN23 single images: BroadTrack JaC@5 75.25, CR 69.8% vs NBJW 73.7 / 77.5% — similar accuracy, lower completeness — [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- PnLCalib: if at least 6 keypoints are visible, intrinsics are estimated first; otherwise K, R, t are jointly optimised; adding line-based PnL refinement lifts JaC@5 by 5-8 points (SN22-center 74.4 to 80.6; WC14 77.6 to 85.9) while points-only refinement gives no gain — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- PnLCalib's extra keypoints on the centre circle and central axis exist specifically to raise completeness in views with few line intersections (CR 86.1 to 98.1 on SN22-center) — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- Temporal smoothing in the 2024 GSR winner: Savitzky-Golay on camera parameters, corrections capped at +/-2 degrees / +/-2 m, 0.5 s look-ahead — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357)
- 2025 GSR teams smoothed reduced camera parameters (pan/tilt/roll/focal) rather than homography entries, or tracked keypoints with Lucas-Kanade from reliable start frames and re-fit RANSAC homographies — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)

### Inferences (integration advice)
- Recommended architecture for one L4: (1) shot-boundary detection first (cheap histogram/SSIM difference or PySceneDetect-style content detector) and classify each shot as main-wide / close-up / replay / other; only calibrate main-wide shots. (2) Within a shot, run the keypoint+line network at 1-2 Hz, solve camera per keyframe, (3) propagate between keyframes with sparse Lucas-Kanade flow on pitch pixels with player boxes masked out (the detector already supplies them), (4) smooth pan/tilt/focal with a zero-phase filter (Savitzky-Golay or RTS smoother) offline, never smooth raw 3x3 homography entries element-wise.
- For fixed elevated/tactical cameras: calibrate once per camera (median over many frames of automatic keypoints, plus a k1 term for wide lenses), then re-check every N seconds for drift. This is the easy case and should give the most trustworthy metres. Wide-angle action cameras need distortion handled before any homography; a homography alone on a fisheye-like image will be wrong by metres near the edges.
- For broadcast: solve the camera position once per match (robust median of per-frame solutions, since a tripod camera does not translate), then fit only pan/tilt/focal(+k1) per frame. This turns a midfield view with two or three visible features into a solvable problem, as BroadTrack's two-point initialisation shows.
- Failure detection / refuse-to-output-metres rules (thresholds below are starting points to tune on own data, not sourced values except where noted): (a) reprojection confidence — IoU/Jaccard between detected line mask and projected template below 0.5 (BroadTrack's sourced threshold) means flag and re-initialise; (b) fewer than 4 inlier ground-plane keypoints with no valid temporal prior means no output; (c) keypoints nearly collinear or covering a small image area (ill-conditioned homography) means no output; (d) physically implausible camera (focal length jump, camera position far from the match median, roll beyond a few degrees) means reject; (e) frame-to-frame jump in projected position of static pitch points; (f) downstream sanity: player speed above ~12 m/s or accelerations beyond human limits mark that segment as uncalibrated rather than clipping silently; (g) unknown pitch dimensions (amateur pitches vary widely in length and width) — if the pitch size is not measured or supplied, report pitch-relative coordinates and label metres as assuming a 105 x 68 m pitch.
- Report coverage honestly: emit a per-frame calibration state (calibrated / propagated / interpolated / invalid) and compute distance only over valid spans, stating the fraction of match time covered.
- Replays are the main contamination risk for physical metrics (double-counted distance, slow-motion speeds). Detect via shot cuts plus broadcast logo-transition detection or by camera-pose discontinuity relative to the main-camera position; exclude non-main-camera shots from metrics by default.

### Gaps
- No open-source, permissively licensed soccer camera tracker equivalent to BroadTrack was found; BroadTrack itself is non-commercial.
- No source quantified homography-vs-camera-model error specifically for amateur or low elevated cameras.
- No sourced, validated thresholds for refusing output other than BroadTrack's 0.5 confidence; the other thresholds above are engineering suggestions.
- Replay/shot-type detection tools were not researched in depth.

## 4. Datasets and licences

### Takeaway
SoccerNet-calibration is the large training set (22,816 images) but is access-controlled and its licence terms were not confirmed here; WC14 and TS-WorldCup are small homography benchmarks drawn from World Cup broadcast footage with unclear redistribution rights. For a commercial product, the footage-derived datasets are the legal weak point, not the code.

### Cited Findings
- SoccerNet-Calibration 2023: 22,816 images from SoccerNet broadcast videos, multi-view in nature (main and other cameras); test set 3,141 images [S for the first, V for the second] — [PnLCalib paper](https://arxiv.org/pdf/2404.08401), [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- WorldCup 2014 (WC14): 209 training images from ten games, 186 test images from ten other games, manually annotated homographies [S] — [PnLCalib paper](https://arxiv.org/pdf/2404.08401)
- TS-WorldCup: 3,812 images from 43 videos of the 2014 and 2018 World Cups with field-marking annotations [S] — [PnLCalib paper](https://arxiv.org/pdf/2404.08401)
- SoccerNet-GSR: 200 sequences of 30 s, 9.37 million line points for pitch localisation plus athlete positions/identities; test split used by BroadTrack is 49 sequences of 750 frames from three Swiss league games [S / V] — [SoccerNet GSR paper](https://arxiv.org/pdf/2404.11335), [BroadTrack paper](https://arxiv.org/html/2412.01721v1)
- GS-HOTA localisation similarity uses a Gaussian kernel with a 5-metre tolerance in pitch coordinates [S] — [SoccerNet GSR paper](https://arxiv.org/pdf/2404.11335)
- sn-gamestate code is GPL-3.0; sn-calibration has no licence detected by GitHub (GitHub API, 2026-11-14) — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate), [SoccerNet/sn-calibration](https://github.com/SoccerNet/sn-calibration)
- The PnLCalib README states its base models were trained on SoccerNet-Calibration "under a research agreement that has since expired" — [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib)
- WorldPose is used as an additional benchmark/fine-tune set by PnLCalib (SV_WP weights) — [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib)
- Open tracking data usable for validating downstream metrics code (not calibration): SkillCorner open data, 10 matches of broadcast tracking, repo MIT; Metrica sample data (repo has no licence file detected) — [SkillCorner/opendata](https://github.com/SkillCorner/opendata), [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)

### Inferences
- A repository with no licence file (sn-calibration, Metrica sample data) defaults to all rights reserved; treat as research/evaluation only until terms are confirmed in writing.
- The cleanest commercial path is to annotate own footage (especially amateur and tactical-camera views, which none of the public sets cover) and fine-tune; use the public sets for benchmarking only.

### Gaps
- Exact licence/NDA terms for SoccerNet-calibration, SoccerNet-GSR, WC14, TS-WorldCup were not found in any fetched source.
- CSD dataset: nothing retrieved (size, content, licence all unverified).
- No public calibration dataset of amateur or low-elevation footage was found.

## 5. From calibrated positions to physical and tactical metrics: practices and open-source code

### Takeaway
Use kloppy as the interchange format and take metrics from maintained libraries (floodlight, databallpy, unravelsports, mplsoccer) rather than writing them from scratch; the hard part for monocular-video data is not the formulas but smoothing and gap handling, since position noise of even 0.5-1 m at 25 Hz produces large fictitious speed and distance.

### Cited Findings
- kloppy: BSD-3-Clause, 572 stars, last push 2026-11-06 — [PySport/kloppy](https://github.com/PySport/kloppy)
- floodlight: MIT, last push 2026-05; provides space-control models, metabolic power and equivalent distance, centroids, distances/velocities/accelerations, spatial transforms, loaders for 10+ providers — [floodlight-sports/floodlight](https://github.com/floodlight-sports/floodlight)
- databallpy: MIT, last push 2026-11-03; tracking/event synchronisation (Needleman-Wunsch), data filtering by tracking quality, velocity/acceleration, pressure, Voronoi space occupation — [Alek050/databallpy](https://github.com/Alek050/databallpy)
- databallpy `get_pitch_control` is a Gaussian player-influence model passed through a sigmoid on a 106 x 68 default grid, i.e. not Spearman's physics-based model [S] — [databallpy docs](https://databallpy.readthedocs.io/en/latest/api_docs/tracking_data_api.html)
- unravelsports: MPL-2.0, last push 2026-01; graph conversion for GNNs, Pressing Intensity, and EFPI formation and position identification — [UnravelSports/unravelsports](https://github.com/UnravelSports/unravelsports)
- LaurieOnTracking (Friends of Tracking): MIT, last push 2022-11 (unmaintained but the reference Python implementation of Spearman-style pitch control used in the tutorials) — [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
- A faster Spearman pitch-control notebook exists (`FastSpearmanPitchControl.ipynb`) [S, code not inspected] — [eddwebster/Metrica-pitch-control](https://nbviewer.org/github/eddwebster/Metrica-pitch-control)
- Spearman's model is physics-based with a control-rate/time-to-intercept formulation; a 2025 paper proposes a neighbour-based alternative for pitch ownership [S] — [A Neighbor-based Approach to Pitch Ownership Models](https://arxiv.org/pdf/2501.05870)
- mplsoccer: MIT, last push 2026-10; pitch drawing, heatmaps, Voronoi — [andrewRowlinson/mplsoccer](https://github.com/andrewRowlinson/mplsoccer)
- roboflow/sports: MIT; radar/minimap drawing utilities over a pitch configuration — [roboflow/sports](https://github.com/roboflow/sports)
- SkillCorner's own open-data documentation says some points are erroneous and that speed/acceleration smoothing and control should be applied to the raw broadcast-tracking data [S] — [SkillCorner/opendata](https://github.com/SkillCorner/opendata)

### Inferences (best practice, not individually sourced)
- Pipeline order: project bbox bottom-centre to pitch, then reject frames flagged invalid by calibration, then resample to a fixed rate (10-25 Hz), then smooth positions (Savitzky-Golay or Kalman/RTS over a ~0.5-1 s window), then differentiate for speed, then cap speed (~12 m/s) and acceleration, then integrate distance. Never integrate distance from unsmoothed positions; jitter adds distance monotonically.
- Bbox bottom-centre is a biased foot estimate when legs are occluded or the player is airborne/sliding; at a typical broadcast scale this is tens of centimetres to over a metre on the pitch. It is a floor on accuracy independent of calibration quality.
- Sprint and high-speed-running counts are threshold metrics and are far more sensitive to noise than total distance. Report them with wider caveats, and use minimum-duration (dwell) rules.
- Per-player totals require identity continuity. With broadcast footage, report per-player distance only as "distance while visible" and scale-ups to 90 minutes as estimates.
- Formation detection is comparatively robust to metre-level noise because it uses time-averaged relative positions (EFPI-style template assignment), but it needs most outfield players visible, so it is realistic for tactical/wide cameras and only intermittently for broadcast.
- Pitch control/Voronoi from broadcast is biased wherever off-screen players would matter (see section 7); mask the surface to the visible frustum or mark it as partial.
- Radar rendering: mplsoccer or roboflow/sports draw utilities are sufficient; render calibration state in the radar (grey out when invalid) so users see when positions are not trustworthy.

### Gaps
- Whether floodlight or unravelsports ship a Spearman pitch-control implementation was not confirmed (floodlight advertises "space control models" without naming them in the README excerpt read).
- No source was found that prescribes specific filter parameters for vision-derived tracking; values above are common practice.
- No open-source library was found that propagates calibration uncertainty into error bounds on speed/distance.

## 6. Validation: how positional and speed accuracy are validated; what is realistic from monocular broadcast video

### Takeaway
The only peer-reviewed head-to-head found (Crang et al., 2026) shows commercial broadcast-tracking vendors ranging from position RMSE 1.68 m to 16.39 m and total-distance bias from -21.8% to +24.3% against TRACAB Gen 5. So roughly 1.5-2 m position RMSE and ~0.3-0.5 m/s speed RMSE is the realistic best case for a good monocular system, on a tactical feed, for players who are actually in view; calibration alone contributes ~0.2-0.6 m of that on clean broadcast frames.

### Cited Findings
- Crang et al., "Concurrent validity of computer-vision artificial intelligence player tracking software using broadcast footage" (Journal of Sports Analytics vol. 12, 2026; arXiv 2508.19477): one 2022 World Cup match; tactical, programme and camera-1 feeds; three commercial providers vs TRACAB Gen 5; position RMSE 1.68-16.39 m; speed RMSE 0.34-2.38 m/s; total match distance mean bias from -1745 m (-21.8%) to +1945 m (+24.3%) across providers — [arXiv abstract 2508.19477](https://export.arxiv.org/abs/2508.19477)
- Same study's conclusions: "fair precision when players are detected by the software"; providers should use the tactical feed to maximise player detection; 720p and 1080p are both suitable — [arXiv abstract 2508.19477](https://export.arxiv.org/abs/2508.19477)
- TRACAB reference accuracy cited in that context: position RMSE 0.08 m, speed RMSE 0.08 m/s [S] — [Sheffield Hallam repository record](https://shura.shu.ac.uk/37394/)
- TRACAB Gen4/Gen5 were themselves validated against VICON motion capture with 20 players in a stadium (Linke et al., PLOS ONE 2020) and judged valid as long as players are tracked correctly [S] — [Linke et al. 2020](https://doi.org/10.1371/journal.pone.0230179)
- A 2025 Science and Medicine in Football study simulating constrained visibility found physical metrics more vulnerable to missing data than tactical analysis [S] — [Broadcast analytics evaluation record](https://www.sponet.de/sponet/Record/4096118?lng=en)
- Driblab white paper claims >98% correlation of broadcast-derived total distance with optical tracking [C; vendor source; correlation, not bias or limits of agreement] — [Driblab report](https://www.driblab.com/blog/driblab-physical-data-reliability-report)
- SkillCorner compared outputs from 10 camera positions against the Premier League's official tracking system at one stadium and reports consistent outputs, without metre or speed error figures in the retrieved summary [C/S] — [SkillCorner article](https://skillcorner.com/articles/does-camera-position-and-capture-style-affect-tracking-quality)
- A request for a metre-level MAE on the SkillCorner open data went unanswered publicly [S] — [SkillCorner/opendata issue 1](https://github.com/SkillCorner/opendata/issues/1)
- Calibration-only pitch error on benchmark frames: mean projection error 0.60 m (median 0.42 m) on WC14 and 0.23 m (median 0.20 m) on TS-WorldCup for PnLCalib with dataset-specific fine-tuning — [PnLCalib paper](https://arxiv.org/html/2404.08401)
- Research benchmarks do not report metres directly for the full pipeline: GS-HOTA uses a 5 m Gaussian tolerance, and the 2024 GSR winner reports no localisation error in metres — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357)

### Inferences
- Validation protocol to copy: time-sync and spatially align own output with a reference (optical tracking, or GPS/LPS vests), then report position RMSE, speed RMSE, and mean bias with limits of agreement for total distance and for each speed band, separately for "player detected" vs "interpolated/extrapolated" samples and per feed type. Correlation alone (the Driblab style) hides bias and should not be the headline.
- Cheap validation options without vendor data: (a) run the pipeline on footage of a match where open tracking data exists; (b) for amateur footage, have a few players wear GPS watches/vests and compare total distance and peak speed (GPS itself carries error of a few percent on distance and more at high speed, so it is a sanity reference, not ground truth); (c) static checks — project known pitch landmarks not used in calibration (penalty spots, corner flags, a measured cone grid) and measure residuals in metres; (d) have a person walk/run a measured line at known pace.
- The benchmark projection errors (0.2-0.6 m) are for fine-tuned, in-distribution professional broadcast frames with standard pitches. Amateur footage with faded lines, lower camera height (which amplifies far-side error because of grazing viewing angle) and unknown pitch dimensions should be assumed several times worse until measured.
- Given a 20+% spread in total distance among commercial vendors, any product claim tighter than roughly +/-5-10% on total distance from monocular broadcast needs own validation evidence behind it. For amateur/single-camera output, labelling metrics as estimates with a stated validation basis is the defensible position.

### Gaps
- The full Crang et al. paper was not retrieved (the HTML version returned 404); per-provider, per-feed breakdowns and high-speed-distance bias are not available here.
- No peer-reviewed validation of any open-source pipeline's metres/speed error was found.
- No source quantifies error as a function of camera height or distance from camera.

## 7. Off-screen players: imputation approaches

### Takeaway
Off-screen imputation exists in research (Graph Imputer, TranSPORTmer, MIDAS) but learned models need full-pitch tracking data to train, which is mostly proprietary; a July 2026 training-free baseline with MIT code roughly halves pitch-control error but individual positions remain 3-9 m off even for short occlusions. Practical for team-level spatial surfaces with caveats; not for per-player physical metrics.

### Cited Findings
- "Training-Free Off-Screen Player Imputation for Broadcast-Based Spatial Football Analytics" (arXiv 2607.11548): online, training-free; best baseline is role-anchored centroid voting; evaluated by simulating a broadcast viewport (36-60 m wide) over three Metrica open matches — [arXiv 2607.11548](https://arxiv.org/abs/2607.11548)
- Its results: ignoring off-screen players gives hidden-zone pitch-control error of 25.1-26.9 percentage points; the best method cuts this to 12.2-13.8 points; median position error 3.3-8.9 m for occlusions of 9.6 s or less, but 50-57% of hidden-player observations are longer than that — [arXiv 2607.11548](https://arxiv.org/abs/2607.11548)
- Its code: `nowayfootball/offscreen-impute`, MIT, last push 2026-10-09 (GitHub API) — [offscreen-impute](https://github.com/nowayfootball/offscreen-impute)
- Graph Imputer (Omidshafiei et al., 2021): graph networks + variational autoencoder, bidirectional, trained on 105 full matches of proprietary tracking data [S] — [arXiv 2106.04219](https://arxiv.org/abs/2106.04219)
- TranSPORTmer (ACCV 2024): one transformer for forecasting, imputation, inference and state classification of multi-agent sports trajectories [S] — [TranSPORTmer](https://openaccess.thecvf.com/content/ACCV2024/papers/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.pdf)
- MIDAS: Set-Transformer imputer that jointly predicts position, velocity and acceleration with a derivative-accumulating self-ensemble for physical consistency [S] — [arXiv 2408.10878](https://arxiv.org/html/2408.10878v3)
- SoccerNet-GSR evaluates visible players only; off-screen players are out of scope [S] — [SoccerNet GSR paper](https://arxiv.org/pdf/2404.11335)

### Inferences
- With 3-9 m median error, imputed positions must not feed distance/speed/sprint metrics. They are acceptable as a labelled, visually distinct "estimated" layer for pitch-control or formation views.
- The 2607.11548 evaluation is simulated from clean tracking data with perfect identities and roles; real broadcast pipelines with ID switches will do worse. It is a single preprint, not peer-reviewed as far as found.
- For elevated wide/tactical cameras that see the whole pitch, imputation is unnecessary; prioritise that footage type for any metric that needs all 22 players.

### Gaps
- Code/weights availability for Graph Imputer, TranSPORTmer and MIDAS was not confirmed.
- No evaluation of any imputer on real (non-simulated) broadcast-derived tracks with ground truth was found.
