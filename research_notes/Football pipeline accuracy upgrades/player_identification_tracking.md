# Player identification, re-identification, team assignment and multi-object tracking for football video (state as of Nov 2026)

Reader notes
- "Verified" below means I read it on the primary page (repo README, arXiv HTML) during this research session. "Self-reported" means the number comes from the authors' own paper and I found no independent reproduction.
- Items I know only from background knowledge and could not re-verify in this session are confined to the Inferences / Gaps subsections and labelled as such.
- No source gave L4-specific timings. All runtime figures are on other GPUs (A100, RTX 5090, RTX 3080Ti laptop) and are flagged.

---

## 1. Top methods on SoccerNet Tracking and SoccerNet Game State Reconstruction (GSR) 2023-2026; public code; long-term association

### Takeaway
Every strong GSR entry in 2024 and 2025 is tracking-by-detection plus an OFFLINE tracklet split-and-merge stage driven by ReID similarity and jersey-number/team constraints, followed by majority voting of identity attributes over each tracklet; that offline stage, not the online tracker, is what produces high association scores (GS-AssA 79-85). None of the top-4 2025 teams or the 2024 winner released code that I could find; the reusable open pieces are the building blocks they used (Deep-EIoU, GTA-Link, TrackLab, PARSeq, CLIP-ReID).

### Cited Findings
GSR 2025 leaderboard (verified, challenge report)
- 2025 GSR leaderboard (GS-HOTA / GS-DetA / GS-AssA): KIST-GSR 63.90 / 51.36 / 79.55 (winner); Constructor.Tech 63.81 / 49.52 / 82.23; lianyou 62.76 / 46.17 / 85.33; Playbox & MIXI 61.64 / 45.04 / 84.36; tyler_durden 57.08 / 41.52 / 78.48; SJTU 50.06 / 32.98 / 75.99; eidos.ai 46.24 / 31.06 / 68.85; UTokyo Football Lab 40.13 / 21.33 / 75.51; hjkim 36.90 / 17.56 / 77.58; Baseline 29.01 / 13.71 / 61.41 — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 14 teams, 66 submissions; 12 teams beat the baseline; the top score rose only marginally over 2024 (63.81 -> 63.90) while the rest of the leaderboard improved a lot — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Note that GS-DetA (localisation + correct identity attributes) is the bottleneck for everyone (best 51.36) while GS-AssA is already 80-85: association is largely solved by offline merging; correct attributes (jersey number, role, team) and pitch localisation are what limit the score — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Winner KIST-GSR pipeline: YOLO-X detection; Deep-EIoU tracker with OSNet ReID embeddings; tracklets refined with split-and-merge (splits triggered by inconsistency of identity predictions inside a tracklet; merges by ReID similarity under spatio-temporal and identity constraints); multi-frame keypoint model for pitch localisation; role, jersey number and colour predicted by LLaMA-3.2-Vision with instruction prompts (autoregressive generation); teams from jersey-colour clustering, left/right from cluster mean x-position; majority voting inside tracklets; filtering of unreliable trajectories — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- The KIST team's follow-up paper "Broadcast2Pitch: Game State Reconstruction from Unconstrained Soccer Videos" (Oo et al.) was a WACV 2026 oral and claims state of the art on SoccerNet-GSR using keypoint + line detection for homography plus ReID and tracklet refinement — [WACV 2026 listing](https://wacv.thecvf.com/virtual/2026/oral/1201); [paper PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)
- lianyou (3rd, highest GS-AssA 85.33): YOLOv12 person detection; custom RT-DETR for field lines; PnLCalib with radial-distortion correction and temporal smoothing; tracker extends IOF-Tracker with Farneback optical flow feeding the Kalman prediction; CLIP-ReIdent ReID; a 6-layer Transformer aggregates tracklet features; K-means for team; fine-tuned CLIP ViT-L/14 for jersey numbers — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Playbox & MIXI (4th, GS-AssA 84.36), the entry closest to an "all open-source parts" recipe: built on TrackLab; RF-DETR detector; CLIP-ReID; BoT-SORT + Global Tracklet Association (GTA); jersey numbers via Gaussian outlier rejection + ViTPose torso crop + PARSeq OCR (i.e. the Koshkina & Elder recipe); BroadTrack-based calibration with optical-flow propagation; goalkeepers assigned heuristically from position with greedy hill-climbing that penalises invalid configurations; interpolation + Savitzky-Golay smoothing — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- eidos.ai (7th): Qwen2-VL-Instruct with function calling; the VLM is shown a grid of a track's crops and may call a "zoom" function over frame ranges before answering the jersey number; same approach for referee role; clustering modified to balance player tracks — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- UTokyo (8th): DeblurGAN-v2 deblurring, fine-tuned YOLOv8x, DeepLabV3-MobileNetV3 silhouettes giving a 10-d colour-area vector appended to ReID/role/team inputs, GTA-Link for tracklet refinement; reported jersey numbers as their main weakness — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- hjkim (9th): motion deblurring, optical-flow camera-motion compensation, detector fine-tuned on SoccerNet-GSR, and an explicit hard constraint of at most 24 identities (22 players + 2 referees) — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- The 2025 report gives no code links for GSR entries — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)

GSR 2024 winner (verified, team paper)
- Constructor Tech won GSR 2024 with GS-HOTA 63.81 (GS-DetA 49.52, GS-AssA 82.23) vs 43.15 (UPCxMobius) and 34.40 (JAM); baseline 23.36 — [From Broadcast to Minimap (arXiv 2504.06357)](https://arxiv.org/html/2504.06357v1)
- Pipeline: fine-tuned YOLOv5m at 1920x1080; SegFormer camera-parameter estimator refined by a ResNet-18 keypoint model, Savitzky-Golay smoothing; a DeepSORT-style tracker that associates in PITCH (metric) coordinates, not pixels; ResNet-50 ReID trained with triplet loss on 370 identities; orientation (ResNet-18) and team embeddings constrain association — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1)
- Their offline tracklet post-processing, in order: (1) split tracklets so each has a single jersey number and a single team ID; (2) merge tracklets with the same jersey number if they do not overlap in time, are physically reachable (speed bound) and share team ID; (3) merge remaining tracklets by ReID similarity (mean-vector cosine distance AND pairwise cosine-similarity matrix) under the same constraints; (4) linear interpolation of gaps. Authors claim a "90% reduction" in tracklet count (no supporting table) — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1)
- Jersey number model: small modified ResNet on 32x32 crops with two heads (first digit 1-9 or none; second digit 0-9), no digit boxes needed; accuracy not reported — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1)
- Team/role: OSNet "TeamID" embedding trained on 111 kit classes; athletes within 30 m of pitch centre clustered into 3 groups (two teams + referees); goalkeeper clusters taken from athletes inside each penalty area; assignment by minimum cosine distance; left/right by majority vote of mean x — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1)
- Runtime: DeepStream + FP16 TensorRT, raw tracking stage "up to 80 FPS" on an RTX 3080Ti laptop GPU (self-reported); no code release stated — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1)

GSR task / baseline (verified)
- SoccerNet-GSR is 200 fully annotated 30-second broadcast clips; GS-HOTA extends HOTA with pitch-coordinate matching and identity attributes (jersey number, role, team). If a number is visible in at least one frame, the whole tracklet is annotated with it — [SoccerNet GSR paper, arXiv 2404.11335](https://arxiv.org/pdf/2404.11335)
- Official baseline (sn-gamestate, built on TrackLab): YOLO detector (repo now references YOLOv11), PRTReID / BPBreID ReID, StrongSORT (tracker-state files are named "prtreid-strongsort"), MMOCR for jersey numbers, team-affiliation clustering, TVCalib calibration with PnLCalib and "No Bells, Just Whistles" as selectable alternatives. Repo licence GPL-3.0. A 2026-05-01 update fixed a broken GS-HOTA evaluation — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)

2026 edition
- A "SoccerNet 2026 challenges results" document is cited by the SoccerTrack v2 paper, but I could not retrieve its content or a 2026 GSR leaderboard — [SoccerTrack v2, arXiv 2508.01802](https://arxiv.org/html/2508.01802)

### Inferences
- The consistent winning recipe is: any decent detector -> motion+appearance online tracker (Deep-EIoU / BoT-SORT-ReID) -> offline SPLIT (by identity/appearance inconsistency) -> offline MERGE (jersey number first, then ReID cosine, gated by time-overlap, reachability and team) -> majority vote of team/role/number per merged track -> interpolation + smoothing. This maps directly onto our problem (hundreds of IDs for 22 players): the fix is an offline global stage, not a better online tracker alone.
- Constructor Tech's choice to associate in pitch coordinates is noteworthy for our pipeline since we already do camera compensation: with a homography, gating merges by metric reachability (e.g. max ~9-10 m/s) is far stronger than pixel-space gating under camera pan/zoom.
- GS-AssA 80-85 at 30 s clips does not prove identity stability across 45 minutes; SoccerNet-GSR clips are 30 s. Long-horizon identity will lean much more heavily on jersey number + team + roster constraints than on ReID embeddings.

### Gaps
- SoccerNet 2026 challenge results (GSR and any tracking task): not found; the report writer should treat 2025 as the latest verified leaderboard.
- SoccerNet Tracking challenge 2023 winners: not researched in this session within the tool budget (from background knowledge the 2023 tracking challenge was won by offline/global-association approaches and the task was then superseded by GSR in 2024, but I did not re-verify this and cite no number).
- Whether Broadcast2Pitch (KIST) has released code/weights: not verified.
- No public code located for KIST-GSR, Constructor.Tech, lianyou or Playbox & MIXI.

---

## 2. Trackers for SportsMOT / SoccerNet-tracking: which gives the best identity continuity, speed, and with which ReID weights

### Takeaway
Among open, runnable options the practical ordering for identity continuity is: Deep-EIoU (or a BoxMOT appearance tracker such as OccluBoost/HybridSORT/StrongSORT) with a sports-trained OSNet, followed by GTA-Link offline refinement (SportsMOT HOTA 77.2 -> 81.0, AssA 67.7 -> 74.5, IDF1 79.8 -> 86.5). Mask-propagation trackers (McByte, SAM2/SAM3-assisted Deep-EIoU, McByte++) report the highest numbers (up to 87.2 HOTA / 93.6 IDF1) but are much slower and their code availability/licensing is unconfirmed.

### Cited Findings
SportsMOT test set (compiled table in a 2026 paper; all numbers self-reported by respective authors)
- HOTA / AssA / IDF1 / MOTA: ByteTrack 64.1 / 52.3 / 71.4 / 95.9; OC-SORT 73.7 / 61.5 / 74.0 / 96.5; MotionTrack 74.0 / 61.7 / 74.0 / 96.6; DiffMOT 76.2 / 65.1 / 76.1 / 97.1; Deep-EIoU 77.2 / 67.7 / 79.8 / 96.3; GTA (Deep-EIoU + GTA-Link) 81.0 / 74.5 / 86.5 / 96.3; NOOUGAT 85.6 / 83.0 / 92.3 / 95.9; SAM 2-Deep-EIoU 85.5 / 81.7 / 91.2 / 97.3; SAM 3-Deep-EIoU 87.2 / 84.2 / 93.6 / 98.1 — [Selective Mask Propagation for MOT, arXiv 2606.13033](https://arxiv.org/html/2606.13033v3)
- The takeaway from that table: MOTA (detection) is ~96-98 for all methods; the entire spread is in AssA (52 -> 84), i.e. association quality is the only thing that differs — [arXiv 2606.13033](https://arxiv.org/html/2606.13033v3)
- Selective mask propagation method: a lightweight base tracker (Deep-EIoU) runs every frame; a VOS model (SAM 2 / SAM 3) is invoked only on temporal windows where the Hungarian assignment margin is small (ambiguous), and overrides the base tracker only when confident. For sports it adds global track association using jersey numbers (OCR on pose-guided torso crops, citing Koshkina & Elder), team identity and appearance for off-screen re-identification. SAM 3 cost 0.105-0.114 s per frame on an RTX 5090 on DanceTrack val (i.e. ~9-10 FPS on a much faster GPU than an L4). Paper licence CC BY 4.0; single author (A. Holmberg); no code link found — [arXiv 2606.13033](https://arxiv.org/html/2606.13033v3)

Deep-EIoU (verified repo)
- Paper-reported 77.2 HOTA on SportsMOT and 85.4 HOTA on SoccerNet-Tracking — [search summary of Deep-EIoU paper via arXiv 2606.13033 / related listings](https://arxiv.org/html/2606.13033v3)
- Repo ships a YOLOX detector checkpoint (best_ckpt.pth.tar) and a sports OSNet ReID checkpoint (sports_model.pth.tar-60) via Google Drive; built on ByteTrack, Torchreid and BoT-SORT code; workflow reads precomputed per-sequence detection and embedding .npy files, with tools/demo.py for raw video and tools/sport_interpolation.py for gap interpolation. NO LICENCE FILE is stated on the repo page (legally "all rights reserved" by default — flag) — [hsiangwei0903/Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU)

GTA-Link (verified repo + paper)
- Offline, tracker-agnostic post-processor: a tracklet SPLITTER (DBSCAN over per-detection ReID embeddings, to cut tracklets containing ID switches) and a CONNECTOR (iteratively merges tracklet pairs by averaged cosine distance, with spatial constraints) — [sjc042/gta-link](https://github.com/sjc042/gta-link)
- Inputs: MOT-format txt per sequence from any tracker + a ReID checkpoint; default ReID is OSNet trained on SportsMOT (sports_model.pth.tar-60, same file name as Deep-EIoU's), built on torchreid 1.4.0. Default params: --min_len 100, --eps 0.6, --min_samples 10, --max_k 3, --spatial_factor 1.0, --merge_dist_thres 0.4. Tested on Python 3.8 / PyTorch 2.3 / CUDA 11.8. Licence MIT — [sjc042/gta-link](https://github.com/sjc042/gta-link)
- Self-reported: SportsMOT HOTA 81.04 (state of the art at the time); on SoccerNet it raised several trackers' HOTA, from 79.41 to 83.11; ACCV 2024 Workshops; arXiv 2411.08216 — [GTA paper, CVF open access](https://openaccess.thecvf.com/content/ACCV2024W/MLCSA2024/html/Sun_GTA_Global_Tracklet_Association_for_Multi-Object_Tracking_in_Sports_ACCVW_2024_paper.html)
- GTA-Link was used in the 2025 GSR challenge by the 4th-place (with BoT-SORT) and 8th-place teams — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)

McByte and McByte++ (verified papers; code status unconfirmed)
- McByte: ByteTrack + temporally propagated segmentation mask (SAM ViT-B to initialise, Cutie to propagate) used as an association cue only in ambiguous cases; ORB-based camera motion compensation; training-free with fixed parameters. SportsMOT test: HOTA 76.9, IDF1 77.5, MOTA 97.2 (vs DiffMOT 76.2 / 76.1, OC-SORT 73.7 / 74.0, ByteTrack 64.1 / 71.4). SoccerNet-tracking 2022 test (oracle detections): McByte HOTA 85.0, IDF1 79.9; OC-SORT 82.0 / 76.3; ByteTrack 72.1 / 75.3; C-BIoU 89.2 / 86.1 with per-dataset tuning — [McByte paper "No Train Yet Gain", arXiv 2506.01373](https://arxiv.org/html/2506.01373v1)
- McByte speed: about 3-5 FPS on a single NVIDIA A100 — [arXiv 2506.01373](https://arxiv.org/html/2506.01373v1)
- McByte++ (Aug 2026, Stanczyk, Yoon, Bremond): training-free; lightweight mask propagation, conditional camera motion compensation, online ReID; claims up to +3.0 HOTA and +6.1 IDF1 over McByte online on SoccerNet-tracking and SportsMOT, further gains with offline global association, and "up to an order-of-magnitude" speed increase. Code announced at github.com/tstanczyk95/McBytePlusPlus ("will be made available"); no code licence stated — [arXiv 2608.15688](https://arxiv.org/abs/2608.15688)
- A search summary of the McByte++ full text states the speed-up comes from switching to the EdgeTAM model (up to ten-fold faster than McByte) and that evaluation covers SportsMOT, SoccerNet-tracking 2022 and 2023 (secondary summary; not verified on the abstract page) — [arXiv 2608.15688 PDF](https://arxiv.org/pdf/2608.15688)

BoxMOT (verified README)
- Licence AGPL-3.0; each integrated detector/segmentor keeps its upstream licence — [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
- Trackers available: OccluBoost, BoTSORT, BoostTrack, StrongSORT, DeepOCSORT, ByteTrack, HybridSORT, OC-SORT, SFSORT — [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
- BoxMOT's own SportsMOT VAL results (HOTA / MOTA / IDF1): OccluBoost 83.17 / 97.48 / 89.36; HybridSORT 81.14 / 98.07 / 81.88; StrongSORT 79.80 / 97.31 / 80.27; DeepOCSORT 79.51 / 97.94 / 79.59; BoTSORT 76.93 / 98.11 / 78.30; OC-SORT 76.34 / 96.60 / 75.64; BoostTrack 76.32 / 97.08 / 77.82; SFSORT 75.73 / 98.39 / 72.99; ByteTrack 67.93 / 97.25 / 76.90 — [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
- On BoxMOT's SportsMOT val table, ByteTrack is ~9 HOTA below BoT-SORT and ~15 HOTA below OccluBoost; IDF1 gap ByteTrack -> OccluBoost is 12.5 points — [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
- Python API takes custom detections directly: `tracker.update(dets, frame)` with dets as an N x 6 array [x1, y1, x2, y2, conf, cls] and a BGR frame; track IDs in output column 4; install `pip install boxmot "ultralytics>=8.4.58"`; example CLI uses the `lmbn_n_duke` ReID model — [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
- An older third-party mirror of the README lists MOT17-ablation speeds of 46 FPS for BoT-SORT and 25 FPS for BoostTrack (hardware unstated; may be outdated) — [boxmot mirror](https://sourceforge.net/projects/boxmot.mirror/)

### Inferences
- For our pipeline the single highest-leverage, lowest-risk change is to turn ON appearance ReID in the online tracker (we currently run BoT-SORT with no ReID) and add an offline GTA-style split/merge pass. Evidence: BoxMOT's table shows appearance-using trackers 9-15 HOTA above ByteTrack on SportsMOT val, and GTA-Link adds ~+4 HOTA / +7 AssA / +7 IDF1 on top of Deep-EIoU.
- GTA-Link is MIT and tracker-agnostic (MOT txt in, MOT txt out), so it can be bolted onto our existing Ultralytics BoT-SORT output without replacing the tracker; the only new requirement is storing a ReID embedding per detection.
- Licensing reality check: our stack is already AGPL-encumbered via Ultralytics YOLO 8.4 (background knowledge: Ultralytics code and weights are AGPL-3.0 unless an enterprise licence is bought), so adding BoxMOT (AGPL-3.0) does not worsen the position; but if the product is offered as a closed-source network service, both are a problem and Deep-EIoU (no licence) is also not clean. A clean-licence path would be: RF-DETR (background knowledge: Apache-2.0 for base sizes) or YOLOX (Apache-2.0) detector + own BoT-SORT/Deep-EIoU-style association re-implementation + torchreid OSNet (MIT code) + GTA-Link (MIT). Verify each licence before relying on this.
- Mask-propagation methods (SAM2/SAM3/Cutie/EdgeTAM) are attractive for occlusion-heavy moments (corners, celebrations) but at McByte's 3-5 FPS on an A100, a 45-minute half at 25 fps (67,500 frames) would take roughly 4-6 hours on an A100 and longer on an L4; the "selective" variants (invoke VOS only on ambiguous windows) are the only version plausible for full halves. This is an estimate from the cited FPS, not a measurement.
- Extended-IoU (Deep-EIoU's box expansion) matters specifically for football because broadcast players are small, fast, and the camera pans; our sparse-optical-flow camera compensation addresses the same failure mode, so the incremental gain from EIoU on top of good CMC is likely smaller than the gain from adding ReID. Not benchmarked.

### Gaps
- No verified numbers found in this session for SAMURAI, SAM2MOT, CAMELTrack, HybridSORT (original paper), Deep OC-SORT (original paper) or DiffMOT on SoccerNet-tracking specifically; only SportsMOT rows above. TrackLab's news notes CAMELTrack was released 2025-05-02 ([TrackLab](https://github.com/TrackingLaboratory/tracklab)) but I did not retrieve its numbers.
- NOOUGAT (85.6 HOTA on SportsMOT): only seen as a table row; method, code and licence not researched.
- No L4 FPS figures exist in any source I read. Expect an L4 to be materially slower than an A100/RTX 5090.
- BoxMOT's README did not list which ReID weights are bundled or their licences; I did not verify whether sports-specific ReID weights ship with it. The sports OSNet checkpoint (sports_model.pth.tar-60) is distributed by Deep-EIoU via Google Drive with no explicit licence; SportsMOT, the dataset it was trained on, has its own terms that I did not verify.
- OccluBoost appears only in BoxMOT's own table; I found no independent paper or third-party reproduction of its 83.17 SportsMOT-val HOTA.
- CLIP-ReID vs OSNet vs PRTReID head-to-head on football: no direct comparison found. Evidence is only circumstantial (2025 GSR 3rd and 4th used CLIP-based ReID; winner used OSNet).

---

## 3. Jersey number recognition: methods, accuracy, code, tracklet aggregation

### Takeaway
Two open recipes dominate: Koshkina & Elder's multi-stage pipeline (ReID-outlier filtering -> legibility classifier -> ViTPose torso crop -> fine-tuned PARSeq -> tracklet consolidation; 87.45% test / 79.31% challenge tracklet accuracy on SoccerNet, but CC BY-NC) and Grad's single-stage uncertainty-aware ViT (up to 88.27% test / 85.62% challenge). VLMs are used by some 2025 GSR entries (LLaMA-3.2-Vision, Qwen2-VL with zoom tool) but there is no published evidence they beat fine-tuned specialists, and jersey numbers remain the weakest GSR component.

### Cited Findings
- SoccerNet jersey-number tracklet-level accuracy (test / challenge): Gerke 2015 32.57 / 35.79; Vats 2021 46.73 / 49.88; Li 2018 47.85 / 50.60; Vats 2022 52.91 / 58.45; Balaji 2023 68.53 / 73.77; Koshkina et al. 2024 87.45 / 79.31; Grad ViT-S (SoccerNet) 82.74 / –; Grad ViT-B (SoccerNet) 86.37 / 83.52; Grad ViT-B (200M pretraining data) 85.46 / 85.62; Grad ViT-B (200M + SoccerNet fine-tune) 88.27 / 85.41 — [Single-Stage Uncertainty-Aware Jersey Number Recognition in Soccer (CVPRW 2025)](https://lukaszgrad.github.io/jnr)
- Grad's method: single stage on the whole player crop (no torso/number detector); digit-compositional head (position embeddings + shared digit classifier) which "consistently outperform[s] independent classifiers"; Dirichlet-based uncertainty; tracklet aggregation by confidence filtering — e.g. 1-second time bins keeping the lowest-uncertainty prediction per bin; code at github.com/lukaszgrad/uncertainty-jnr (code/weights licence not stated on the project page; needs checking) — [project page](https://lukaszgrad.github.io/jnr)
- Koshkina & Elder pipeline stages for SoccerNet tracklets: (1) occlusion/outlier removal using ReID features (Centroid-ReID) and a fitted Gaussian; (2) legibility classifier; (3) pose-guided RoI (torso) cropping with ViTPose; (4) scene text recognition with PARSeq fine-tuned on SoccerNet; (5) tracklet prediction consolidation. Released weights: PARSeq (original, hockey-fine-tuned, SoccerNet-fine-tuned), legibility classifiers (hockey, SoccerNet), plus pointers to Centroid-ReID and ViTPose weights. Requires PyTorch 1.9 and per-repo conda envs — [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)
- LICENCE FLAG: the Koshkina repo is CC BY-NC 3.0 (non-commercial); the hockey dataset is not public — [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)
- SoccerNet jersey task definition: short player tracklets, predict the number; the number may be visible in only a very small subset of frames; non-visible numbers get a sentinel label — [SoccerNet/sn-jersey](https://github.com/SoccerNet/sn-jersey)
- 2025 GSR entries' jersey approaches: LLaMA-3.2-Vision instruction prompting (winner); fine-tuned CLIP ViT-L/14 (3rd); Gaussian outlier rejection + ViTPose torso + PARSeq (4th); Qwen2-VL with a zoom function over a grid of track crops (7th); several teams (e.g. 8th) cite jersey numbers as their main weakness — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 2024 GSR winner used a tiny two-head ResNet on 32x32 crops (first digit / second digit) trained without digit boxes, and then used the number as the PRIMARY merge key for tracklets — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1)
- The official baseline uses MMOCR for jersey numbers, and the baseline scores only 29.01 GS-HOTA in 2025 — [sn-gamestate](https://github.com/SoccerNet/sn-gamestate); [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Generic zero-shot VLMs are weak on soccer understanding benchmarks: one table on the SoccerNet test set reports GPT-4o at 0.39 and Qwen2.5-VL-32B at 0.40 accuracy (general soccer VQA, NOT jersey-number-specific) — [MSUE: Multi-Modal Soccer Understanding Expert](https://www.alphaxiv.org/abs/2606.12106.md)
- Roboflow's basketball player-identification write-up uses RF-DETR to detect players and jersey numbers, SAM2 for tracking and SigLIP+UMAP+K-means for teams (basketball, close-up numbers — a different regime from wide football shots) — [Roboflow blog](https://blog.roboflow.com/identify-basketball-players/)

### Inferences
- Recommended aggregation for our pipeline (synthesised from the three cited designs): per detection, run a cheap legibility gate; on legible crops only, run the recogniser; store (number, confidence/uncertainty) per frame; per tracklet, take a confidence-weighted vote with time-binning (one vote per ~1 s) so that one long clear view does not get outvoted by many blurry frames and one burst of identical frames does not dominate; emit "unknown" unless the top number wins by a margin. Then use numbers as hard merge keys across tracklets within a team (Constructor Tech step 2) and as SPLIT triggers when a tracklet contains two confident different numbers (step 1 / KIST's split rule).
- For a tactical (wide, high) camera, numbers are often illegible (players 40-80 px tall). Legibility gating matters more than the recogniser choice; expect only a small fraction of frames per player to yield a number, which is exactly why tracklet merging must come first so the sparse readings propagate to a long track.
- Cost: the legibility classifier + PARSeq are small models; ViTPose over every crop is the expensive part of the Koshkina recipe. Grad's single-stage ViT avoids pose entirely, which is attractive on one L4. Running a VLM per tracklet (a grid of the N most legible crops, as eidos.ai did) is affordable for 22-60 merged tracks per clip but not per detection. These are engineering judgements, not benchmarked.
- For commercial use, the Koshkina weights/code (CC BY-NC) cannot be used as-is; the recipe can be re-implemented with PARSeq (background knowledge: baudm/parseq is Apache-2.0 — verify) fine-tuned on own data, but the SoccerNet jersey dataset used for fine-tuning has its own research-oriented terms (not verified here).

### Gaps
- No published head-to-head of small VLMs (Florence-2, SmolVLM, Qwen2.5-VL small variants) on the SoccerNet Jersey Number benchmark was found; VLM usage in GSR 2025 is reported without per-component accuracy.
- Licence of lukaszgrad/uncertainty-jnr code and whether trained weights (esp. the 200M-pretrained one) are released: not verified.
- SoccerNet jersey-number challenge official leaderboards for 2023/2024 winners (team names, scores) not retrieved; the 79.31-85.62 "challenge" numbers above are from Grad's comparison table.
- Model size / throughput of any jersey recogniser on an L4: no source.
- Licence terms of the SoccerNet datasets (tracking, GSR, jersey): not stated on the repo pages I read. Background knowledge (unverified this session): SoccerNet video data requires signing an NDA and is for research use; treat as research-only until confirmed.

---

## 4. Team / role classification (team A/B, goalkeeper, referee)

### Takeaway
Top systems all use unsupervised clustering of appearance embeddings per match (K-means on colour or learned kit embeddings), then per-tracklet majority voting, with goalkeepers and referees handled by position heuristics plus a separate cluster rather than by the detector class alone. The Roboflow SigLIP+UMAP+KMeans recipe is the common community default; the strongest reported variant is a kit-specific embedding (OSNet trained on kit classes) clustered into 3 groups using only centrally-located athletes.

### Cited Findings
- 2025 winner: jersey-colour clustering for teams; left/right side from cluster-wise mean x-position; majority vote inside tracklets — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 2025 3rd: 6-layer Transformer aggregates tracklet features, then K-means for team assignment — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 2025 4th: goalkeepers assigned heuristically from positioning with greedy hill-climbing that penalises invalid configurations (e.g. two goalkeepers on one team) — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 2025 8th: segmentation silhouettes (DeepLabV3-MobileNetV3) used to compute a 10-d colour-area vector that excludes background pixels, appended to ReID/role/team inputs — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 2025 7th: clustering "modified to balance player tracks" (i.e. enforce roughly equal team sizes), and a VLM for referee role — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 2024 winner: OSNet TeamID embedding trained on 111 kit classes; cluster only athletes within 30 m of pitch centre into 3 groups (team A, team B, referees) to get clean prototypes; define goalkeeper clusters from athletes inside each penalty area; assign every athlete by minimum cosine distance to cluster centres — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1)
- PRTReID (Somers et al.): one model producing ReID features with two extra objectives for team affiliation and role classification (a role head on top of the BPBReID-derived model), trained on SoccerNet-GSR / SoccerNet-Tracking through the sn-gamestate dataloader. LICENCE FLAG: Hippocratic License 3.0 (HL3-LAW-MEDIA-MIL-SOC-SV), a non-standard ethical-use licence. The repo page says pretrained-weight download instructions "will be added later" and reports no metrics — [VlSomers/prtreid](https://github.com/VlSomers/prtreid)
- Roboflow-style team classifier as implemented in community soccer toolkits: SigLIP embeddings of player crops -> UMAP dimensionality reduction -> K-means — [Football-PitchVision](https://github.com/DesusLove/Football-PitchVision); [wunderscout on PyPI](https://pypi.org/project/wunderscout/0.1.24/); same pattern in Roboflow's own write-up — [Roboflow blog](https://blog.roboflow.com/identify-basketball-players/)
- roboflow/sports repository licence is MIT; it has no PyPI package (install from source); README lists soccer player/ball/pitch-keypoint datasets on Roboflow Universe without stating their licences; 34 commits, 32 open issues, 21 open PRs at time of reading — [roboflow/sports](https://github.com/roboflow/sports)
- A cheaper alternative in the wild: K-means on HSV jersey-colour histograms — [sportvision on PyPI](https://pypi.org/project/sportvision/)

### Inferences
- Most robust cheap design for us, combining the cited ideas: (a) compute a per-detection kit descriptor from the torso region only (mask out grass with a green-hue mask or a cheap silhouette; 8th-place idea), (b) fit clusters once per match/clip on a clean subset — outfield "player"-class detections in the central part of the pitch, not occluded (low IoU with other boxes) — with k=2 for players plus separate handling of referee and goalkeeper classes from our detector, (c) assign per TRACKLET by majority/mean-embedding vote, never per frame, (d) goalkeeper team = nearest goal side via pitch coordinates (GK in the left third belongs to the team defending left), (e) enforce sanity constraints (<= 11 per team incl. exactly one GK; balanced cluster sizes).
- Our "weak team grouping" is probably caused more by per-frame assignment on fragmented tracks and background contamination in the crop than by the choice of colour vs. SigLIP features. Fixing fragmentation first and voting per tracklet should lift team accuracy regardless of descriptor. This is a reasoned hypothesis, not measured.
- SigLIP embeddings are heavier than colour histograms (a ViT forward pass per crop) but can be subsampled (e.g. every 5th-10th frame per track) since assignment is per tracklet; cost then becomes negligible relative to detection.
- PRTReID is the only open model that natively outputs ReID + team + role, but the Hippocratic licence, missing public weights on its own page (weights are obtained through sn-gamestate's GPL-3.0 baseline download), and SoccerNet training data make it a research-only choice.

### Gaps
- No quantitative team-classification accuracy comparison (SigLIP+UMAP+KMeans vs colour histogram vs PRTReID vs kit-trained OSNet) was found; all evidence is which method top teams chose.
- I could not confirm from the roboflow/sports README itself the exact TeamClassifier implementation details (model variant, UMAP dims, batch size); the SigLIP+UMAP+KMeans description comes from derivative projects and Roboflow's blog. Background knowledge (unverified this session): it uses google/siglip-base-patch16-224, UMAP to 3 dimensions and KMeans with 2 clusters, and resolves goalkeepers by distance to each team's player centroid.
- Licences of Roboflow Universe football datasets/models and of the SigLIP weights (background knowledge: Apache-2.0) were not verified this session.

---

## 5. Frameworks: TrackLab, sn-gamestate, Roboflow sports/supervision, BoxMOT — what they provide, licences, adoption effort

### Takeaway
TrackLab (MIT) is the research framework most GSR teams build on, with sn-gamestate (GPL-3.0) as the football-specific layer adding calibration, jersey OCR and team clustering; BoxMOT (AGPL-3.0) is the easiest drop-in for swapping online trackers inside an existing Ultralytics loop; roboflow/sports (MIT) is example code rather than a maintained library. For an existing production pipeline, borrowing components (BoxMOT tracker call, GTA-Link script, PARSeq) is cheaper than migrating into TrackLab.

### Cited Findings
- TrackLab: MIT licence; modular Hydra-configured pipeline; built-in detectors (YOLO, YOLOX, RTMDet, RT-DETR), pose (RTMPose, RTMO, ViTPose, YOLO-pose), ReID (KPReID, BPBReID), trackers (DeepSORT, StrongSORT, OC-SORT); CAMELTrack released 2025-05-02; extra detectors/pose added 2025-05-22; install examples use Python 3.12 and PyTorch 2.6; extension points are ImageLevelModule (per frame), DetectionLevelModule (per detection), VideoLevelModule (whole video, for offline methods), plus Callback / Evaluator / TrackingDataset; has an OfflineTrackingEngine and a MajorityVoteTracklet module; no built-in tracklet-merging module listed; no tagged releases — [TrackingLaboratory/tracklab](https://github.com/TrackingLaboratory/tracklab)
- sn-gamestate: GPL-3.0; baseline = YOLO detection + PRTReID/BPBreID + StrongSORT + MMOCR jersey OCR + team clustering + TVCalib (or PnLCalib / NBJW) calibration; Python 3.9, conda option with PyTorch 1.13.1 / CUDA 11.7, mmcv 2.0.1; advises lowering batch sizes on memory issues; evaluation fix on 2026-05-01 — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- The sn-gamestate baseline is weak as a system: 29.01 GS-HOTA (GS-DetA 13.71, GS-AssA 61.41) on the 2025 challenge set vs 63.90 for the winner — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- BoxMOT: AGPL-3.0; nine trackers behind one `tracker.update(dets, frame)` API taking plain numpy detections; requires `ultralytics>=8.4.58` in its install line; publishes its own MOT17 / SportsMOT / MMOT benchmark tables; has an "Evaluation and Postprocessing" guide (contents not read) — [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
- roboflow/sports: MIT; no Python package ("We don't have a Python package yet"); lists reading jersey numbers and ball tracking as open challenges rather than solved features; links football detection and pitch-keypoint datasets — [roboflow/sports](https://github.com/roboflow/sports)
- GTA-Link: MIT; two scripts (generate_tracklets.py -> .pkl; refine_tracklets.py) operating on MOT-format txt — [sjc042/gta-link](https://github.com/sjc042/gta-link)

### Inferences
- Adoption effort ranking for our YOLO + BoT-SORT pipeline (lowest first): (1) enable ReID in Ultralytics' own BoT-SORT config (background knowledge: `with_reid: True` in botsort.yaml; recent Ultralytics versions can use native YOLO features or a separate classification model as the encoder — verify against 8.4 docs); (2) call a BoxMOT tracker in place of the Ultralytics tracker with our existing detections; (3) add GTA-Link-style offline refinement on saved per-detection embeddings; (4) bolt on jersey-number and team modules operating on merged tracklets; (5) full migration to TrackLab + sn-gamestate (highest effort: Hydra configs, own dataset class, old-PyTorch/mmcv pins in sn-gamestate, GPL-3.0).
- sn-gamestate's dependency pins (Python 3.9, PyTorch 1.13, mmcv 2.0.1) conflict with a modern Ultralytics 8.4 environment; running it side by side in a separate Modal image is more realistic than merging environments.
- TrackLab's VideoLevelModule is the natural place for an offline split/merge stage if we did adopt it, which is what the 2025 4th-place team did (TrackLab + BoT-SORT + GTA).

### Gaps
- `supervision` (Roboflow) library specifics (ByteTrack implementation, licence — background knowledge: MIT) were not verified this session.
- Whether BoxMOT's postprocessing guide includes GSI / interpolation / tracklet merging: page not read.
- No evidence on memory footprint of any framework on a 24 GB L4.

---

## 6. Offline / global post-processing that fixes fragmentation

### Takeaway
The proven pattern is split-then-merge on stored per-detection embeddings: split impure tracklets (DBSCAN over embeddings, or on conflicting jersey number/team), then greedily merge tracklets by jersey number and mean-embedding cosine distance subject to hard gates (no temporal overlap, physically reachable, same team), cap identities (<= 22 players + officials), then interpolate and smooth. This is exactly the stage our pipeline lacks.

### Cited Findings
- GTA-Link splitter: DBSCAN on a tracklet's embeddings (eps 0.6, min_samples 10, only tracklets >= 100 frames, at most 3 sub-tracklets); connector: iterative pairwise merge by averaged cosine distance (threshold 0.4) with a spatial-constraint factor — [sjc042/gta-link](https://github.com/sjc042/gta-link)
- GTA-Link effect: Deep-EIoU 77.2 HOTA / 67.7 AssA / 79.8 IDF1 -> 81.0 / 74.5 / 86.5 on SportsMOT test — [arXiv 2606.13033 table](https://arxiv.org/html/2606.13033v3); SoccerNet HOTA 79.41 -> 83.11 across trackers (self-reported) — [GTA paper](https://openaccess.thecvf.com/content/ACCV2024W/MLCSA2024/html/Sun_GTA_Global_Tracklet_Association_for_Multi-Object_Tracking_in_Sports_ACCVW_2024_paper.html)
- Constructor Tech's 4-step constrained merge (split to single number + single team; merge on same number with non-overlap / reachability / same team; merge on ReID mean-vector + pairwise similarity; linear interpolation), claimed to cut tracklet count by 90% — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1)
- KIST 2025: split on identity-prediction inconsistency; merge on ReID similarity under spatio-temporal and identity constraints; majority voting; filter unreliable trajectories — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Hard identity cap used in the 2025 challenge: at most 24 identities (22 players + 2 referees) — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Gap filling: interpolation + Savitzky-Golay smoothing (2025 4th place); linear interpolation (2024 winner); Deep-EIoU ships tools/sport_interpolation.py — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1); [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1); [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU)
- The 2026 selective-mask paper's global association uses jersey numbers, team identity and appearance together for off-screen re-identification — [arXiv 2606.13033](https://arxiv.org/html/2606.13033v3)
- Outlier rejection before aggregating a tracklet's appearance/number: fit a Gaussian to the tracklet's ReID features and drop outliers (occluded / wrong-player crops) — [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)

### Inferences
- Concrete design for our pipeline (synthesis):
  1. Online pass: keep YOLO detections; tracker with ReID on; persist per detection {frame, box, class, conf, track id, embedding (e.g. 256-512-d float16), pitch xy if homography available}.
  2. Tracklet hygiene: drop tracklets < N frames with low confidence; split with DBSCAN on embeddings and on conflicting confident jersey numbers / team labels.
  3. Per-tracklet attributes: team (vote), role (detector class vote: player/goalkeeper/referee), jersey number (confidence-weighted, time-binned vote), robust mean embedding (after Gaussian outlier rejection).
  4. Merge as an assignment problem: cost = cosine distance of mean embeddings; forbidden if temporal overlap, different team/role, conflicting numbers, or implied speed above a physical bound in pitch coordinates (or pixel bound after camera compensation); same confident number + same team = forced merge. Greedy/hierarchical agglomeration until cost > threshold or until identity count per team reaches 11 (+ referees).
  5. Interpolate short gaps only (long gaps = off-screen in broadcast; do not invent positions), then smooth.
- Storage for 45 minutes: 67,500 frames x ~25 detections x 512-d float16 is roughly 1.7 GB of embeddings (my arithmetic) — fine on disk/RAM; merging itself is CPU-only and operates on at most a few thousand tracklets, so the offline stage adds negligible GPU cost.
- For 45-minute halves on broadcast video, players leave the frame for long periods; appearance embeddings alone cannot separate same-team outfield players reliably over that horizon (identical kits). Long-horizon identity should be anchored on jersey number, with ReID used only to bridge short gaps (seconds to tens of seconds) and a tactical/wide camera — where all players remain in view — making the problem far easier than broadcast. This is a reasoned expectation; benchmarks cited above are all on 30 s (GSR) or short SportsMOT/SoccerNet-tracking clips.
- Process long halves in windows (e.g. 1-2 minute chunks with overlap) for the online tracker and run global merging hierarchically (within chunk, then across chunks keyed on team + number) to keep the merge problem small.

### Gaps
- No verified football numbers for GSI (Gaussian-smoothed interpolation, StrongSORT++) or AFLink in this session.
- No published evaluation of identity continuity over full 45-minute halves from open methods was found; all benchmarks are short clips.
- The "90% tracklet reduction" claim (Constructor Tech) has no supporting table.

---

## 7. Mapping tracks to named roster players; commercial practice; human-in-the-loop tools

### Takeaway
No open-source end-to-end system for roster-named player identification, and no dedicated open human-in-the-loop correction tool for football tracking, was found. The identity key used across SoccerNet-GSR and the top systems is (team, role, jersey number), which maps to a name by a simple roster lookup once the team side and kit are known; everything else must be built.

### Cited Findings
- SoccerNet-GSR identifies athletes by role + team + jersey number, with a plain track ID as fallback when those do not yield a unique identity; if the number is visible in at least one frame the whole tracklet carries it — [SoccerNet GSR paper, arXiv 2404.11335](https://arxiv.org/pdf/2404.11335)
- A 2022 semi-automatic annotation paper describes incremental learning of a ReID model from a few user interactions to strengthen re-identification throughout the annotation process (closest published human-in-the-loop precedent; tool release not confirmed) — [arXiv 2204.04049](https://arxiv.org/pdf/2204.04049.pdf)
- Top GSR systems treat the jersey number as the primary cross-tracklet key and team side (left/right) as the team identifier, i.e. the output is "left team #10", not a name — [From Broadcast to Minimap](https://arxiv.org/html/2504.06357v1); [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- SoccerTrack v2 is a full-pitch panoramic video dataset aimed at game state reconstruction and ball action spotting (relevant to tactical-camera rather than broadcast footage) — [SoccerTrack v2, arXiv 2508.01802](https://arxiv.org/html/2508.01802)

### Inferences
- Practical roster-mapping design (my synthesis; no public source documents commercial vendors' internals in what I read): user supplies per match a roster {team, kit colour, shirt number, name, position}; after merging, each track has (team cluster, role, number posterior). Solve a per-team assignment (Hungarian) between merged tracks and roster entries using number posterior as the main cost, restricted to numbers that exist on the team sheet (this alone removes many OCR errors, e.g. "18" vs "13" when only one is on the pitch), with weak priors from position (GK role fixed; average pitch position vs nominal position/formation slot) for tracks with no legible number. Map team cluster -> roster team by kit colour or by which side the team defends.
- Restricting OCR outputs to the 11 on-pitch numbers per team (plus substitutes after substitution events) is the cheapest accuracy gain available and is not exploited in the SoccerNet benchmarks (which have no roster).
- Human correction is needed for the residual: a review UI needs only three operations — merge two tracks, split a track at a frame, assign a roster player to a track — each of which should propagate through the constraint solver (a confirmed label becomes a hard constraint). General-purpose annotation tools (background knowledge, unverified this session: CVAT is MIT-licensed and supports track-mode video annotation; Label Studio is Apache-2.0) could host this, but no football-specific open tool was found.
- Commercial optical tracking providers (background knowledge, unverified, no source located this session) are generally understood to combine multi-camera tracking with human operators for identity QA; I found no citable public technical description and none should be asserted in the final report without a source.

### Gaps
- No reliable sources found on how commercial systems (e.g. broadcast-tracking vendors) map tracks to named players; any statement about them would be speculation.
- No open-source human-in-the-loop football tracking correction tool found.
- No open dataset with roster-name ground truth for broadcast tracking was identified in this session.

---

## Cross-cutting licence table (verified this session unless marked)

### Takeaway
Clean-licence building blocks: GTA-Link (MIT), TrackLab (MIT), roboflow/sports (MIT). Copyleft: BoxMOT (AGPL-3.0), sn-gamestate (GPL-3.0). Restricted or unclear: Koshkina jersey pipeline (CC BY-NC 3.0, non-commercial), PRTReID (Hippocratic License 3.0), Deep-EIoU (no licence stated), McByte / McByte++ (no code licence stated; code "will be made available").

### Cited Findings
- GTA-Link: MIT — [sjc042/gta-link](https://github.com/sjc042/gta-link)
- TrackLab: MIT — [TrackingLaboratory/tracklab](https://github.com/TrackingLaboratory/tracklab)
- roboflow/sports: MIT (datasets/models' licences not stated in README) — [roboflow/sports](https://github.com/roboflow/sports)
- BoxMOT: AGPL-3.0 — [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
- sn-gamestate: GPL-3.0 — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- mkoshkina/jersey-number-pipeline: CC BY-NC 3.0 (non-commercial) — [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)
- PRTReID: Hippocratic License 3.0 (HL3-LAW-MEDIA-MIL-SOC-SV) — [VlSomers/prtreid](https://github.com/VlSomers/prtreid)
- Deep-EIoU: no licence stated on the repo page — [hsiangwei0903/Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU)
- McByte / McByte++: code announced, no licence stated — [arXiv 2506.01373](https://arxiv.org/html/2506.01373v1); [arXiv 2608.15688](https://arxiv.org/abs/2608.15688)

### Inferences
- Any weights trained on SoccerNet data (PRTReID, SoccerNet-fine-tuned PARSeq, sn-gamestate baseline checkpoints) or SportsMOT (sports OSNet checkpoint) inherit dataset-terms risk for commercial use in addition to the code licence; for a commercial product the safest path is to re-train the small models (OSNet ReID, legibility classifier, PARSeq/digit head) on self-collected and self-labelled footage, using merged-and-corrected tracks from the pipeline itself as training data.

### Gaps
- Not verified this session: licences of Ultralytics YOLO 8.4 (background: AGPL-3.0 / enterprise), RF-DETR, YOLOX, torchreid/OSNet weights, PARSeq, ViTPose, CLIP-ReID, SigLIP, SAM 2 / SAM 3 / EdgeTAM / Cutie, uncertainty-jnr, and the SoccerNet / SportsMOT dataset terms. Each must be checked before the report recommends them for commercial use.
