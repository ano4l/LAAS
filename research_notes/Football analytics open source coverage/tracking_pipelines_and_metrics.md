# Open-source football video-to-tracking pipelines and tracking-data analytics libraries (state as of 18 November 2026)

Notes on method: repository licence, star count and last-push date were read from the GitHub API on 2026-11-18 and are cited to the repository URL. "NO-LICENSE" means the repository has no licence file (default copyright applies: no grant to use commercially or otherwise). Numbers are quoted from the named source only; where I saw a figure only in a search excerpt and could not open the primary page, that is stated.

## 1. (A) SoccerNet Game State Reconstruction: baseline, challenge winners, components, what is released

### Takeaway
The only complete, runnable, weights-included open pipeline for broadcast video to per-player pitch coordinates with role, team and jersey number is SoccerNet `sn-gamestate` on TrackLab, but it is GPL-3.0, slow (about 11 minutes per 30 s clip on an A100) and scores only 22 to 29 GS-HOTA; the 2024 and 2025 challenge leaders reach about 64 GS-HOTA but none of them released code or weights that I could find. Game State Reconstruction was not run as a SoccerNet challenge task in 2026, so the 2025 leaderboard is the latest.

### Cited Findings

**Baseline (sn-gamestate + TrackLab)**
- `SoccerNet/sn-gamestate`: GPL-3.0, 472 stars, last push 2026-05-02, not archived. Latest README entry: "Update to Tracklab 1.3.24 to fix broken GS-HOTA evaluation" (2026-05-01); a uv install and new visualisation tool were added 2025-05-02; dataset version 1.3 (2024-05-13) is still current. — [sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- `TrackingLaboratory/tracklab` (the generic framework underneath): MIT, 258 stars, last push 2026-05-01, latest tag v1.3.24. — [tracklab](https://github.com/TrackingLaboratory/tracklab)
- Outputs: per athlete 2D pitch position, role (player, goalkeeper, referee, other), team (left/right) and jersey number; an .mp4 visualisation; a saved tracker state (.pklz) with boxes, ReID embeddings, jersey numbers and track IDs; a GS-HOTA score. — [sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- Current components named in the README: YOLOv11 (Ultralytics) detection; PRTReID and BPBreID re-identification (PRTReID also gives team and role); TVCalib calibration by default with PnLCalib and "No Bells, Just Whistles" (NBJW) selectable in `soccernet.yaml`; MMOCR jersey numbers. Dataset and all model weights download automatically on first run. — [sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- Original 2024 baseline: YOLOv8 (not fine-tuned on SoccerNet), StrongSORT, PRTReID, TVCalib, MMOCR (DBNet + SAR). GS-HOTA 22.26 on test, 18.05 on valid, 23.36 on the 2024 challenge set. Plain image-space HOTA for the same tracker is 57.64 (DetA 67.42, AssA 49.42). — [Somers et al. 2024, arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- Where the baseline loses score (GS-HOTA when requirements are added): pitch position only 42.65; pitch + role 40.76; pitch + team 37.03; pitch + jersey number 25.65; everything 22.26. The paper names jersey number recognition as the main weakness, then pitch localisation. — [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- Runtime: about 11 minutes per 30 s sequence on an NVIDIA A100; full pipeline 1.1 FPS; module speeds ReID 14.5 FPS, YOLOv8 16.5 FPS, calibration 7.6 FPS, pitch (TVCalib) 2.9 FPS, jersey (MMOCR) 3.8 FPS. — [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- Dataset: 200 sequences of 30 s, 9.37 million pitch line points, 2.36 million athlete positions, built on SoccerNet-Tracking. — [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- Hugging Face dataset `SoccerNet/SN-GSR-2025`: not gated, card licence field "gpl-3.0", last modified 2024-10-22. — [SN-GSR-2025 on Hugging Face](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025)

**Licences of the baseline's components (this matters more than the top-level licence)**
- PRTReID and BPBreID: Hippocratic License 3.0 (HL3-LAW-MEDIA-MIL-SOC-SV), not an OSI licence. — [prtreid](https://github.com/VlSomers/prtreid), [bpbreid](https://github.com/VlSomers/bpbreid)
- Ultralytics YOLO: AGPL-3.0. Ultralytics states "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default", that the licence "covers the training code and the models produced by that training code", and that an Enterprise License is required for "Any commercial product or service" and "SaaS platforms, APIs, or cloud systems that use YOLO behind the scenes" unless the whole project is open-sourced under AGPL-3.0. — [Ultralytics licence page](https://www.ultralytics.com/license)
- PnLCalib: code GPL-2.0-or-later, model weights CC BY 4.0; pretrained keypoint and line models released (single-view base plus WC14, TS-WorldCup and WorldPose fine-tunes; multi-view base). 114 stars, last push 2026-10-26; accepted to Computer Vision and Image Understanding (README note dated 2026-03-10). — [PnLCalib](https://github.com/mguti97/PnLCalib)
- NBJW ("No Bells, Just Whistles"): licence file is GPL v2; pretrained weights in release v1.0.0; 61 stars; last push 2024-10-25. — [No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles)
- TVCalib: MIT, 56 stars, last push 2024-04-27. — [tvcalib](https://github.com/MM4SPA/tvcalib)
- MMOCR: Apache-2.0, last push 2024-11-27. — [mmocr](https://github.com/open-mmlab/mmocr)
- `SoccerNet/sn-calibration` (dev kit): no licence file, last push 2024-06-18. — [sn-calibration](https://github.com/SoccerNet/sn-calibration)

**2024 challenge**
- Constructor Tech won with GS-HOTA 63.81 (GS-DetA 49.52, GS-AssA 82.23); second UPCxMobius 43.15; third JAM 34.40; baseline 23.36. — [Golovkin et al., arXiv 2504.06357](https://arxiv.org/html/2504.06357v1)
- Constructor Tech components: fine-tuned YOLOv5m detector; DeepSORT-based tracker run in pitch coordinates with orientation and team constraints; ResNet-50 ReID; OSNet team embeddings clustered into five groups; modified ResNet jersey classifier on 32x32 crops with separate heads per digit; SegFormer-based camera parameter regressor refined by a field keypoint model; orientation classifier; anomaly filter. Post-processing claimed a "90% reduction in the number of tracklets". Tracking reported at "up to 80 FPS" on an RTX 3080Ti laptop GPU with DeepStream and FP16 TensorRT. No code or weights repository is linked. — [arXiv 2504.06357](https://arxiv.org/html/2504.06357v1)

**2025 challenge (latest GSR leaderboard)**
- 14 teams, 66 submissions, 12 beat the baseline. Top five: KIST-GSR 63.90 (DetA 51.36, AssA 79.55); Constructor.Tech 63.81 (49.52 / 82.23); lianyou 62.76 (46.17 / 85.33); Playbox & MIXI 61.64 (45.04 / 84.36); tyler_durden 57.08. Baseline 29.01 (13.71 / 61.41). — [SoccerNet 2025 Challenges Results, arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)
- KIST-GSR (1st): YOLO-X detector; Deep-EIoU tracker with OSNet ReID; multi-frame keypoint model for pitch mapping; LLaMA-3.2-Vision for role, jersey number and colour; jersey-colour clustering for left/right. — [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)
- lianyou (3rd): YOLOv12 for people, RT-DETR for field lines; radial distortion correction plus PnLCalib with temporal smoothing; IOF-Tracker with Farneback optical flow and Kalman filtering; CLIP-ReIdent; K-means teams; fine-tuned ViT-L/14 CLIP for jersey numbers. — [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)
- Playbox & MIXI (4th): RF-DETR detector; CLIP-ReID features with BoT-SORT and GTA (global tracklet association); BroadTrack calibration propagated with optical flow; ViTPose torso crops and PARSeq OCR for jersey numbers; heuristic goalkeeper assignment. — [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)
- Lower-ranked but instructive: UTokyo (8th, 40.13) used fine-tuned YOLOv8x, DeblurGAN-v2 and GTA-Link; hjkim (9th, 36.90) enforced a cap of 24 identities (22 players, 2 referees); eidos.ai (7th, 46.24) used Qwen2-VL with a zoom tool for jersey numbers. — [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)
- The results paper gives no code URL for any GSR entry. — [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)

**2026**
- The SoccerNet 2026 challenges (arXiv 2607.07320, 8 July 2026) had five tasks: Ball Action Anticipation, Player-Centric Ball Action Spotting, Novel View Synthesis, Spiideo SoccerNet SynLoc, Visual Question Answering. Game State Reconstruction, tracking and calibration were not among them. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320)
- Player-Centric Ball Action Spotting 2026 requires each action to be attributed to a player by team side and jersey number (correct only if class, team and jersey number all match within ±12 frames). — [arXiv 2607.07320](https://arxiv.org/html/2607.07320)
- "Broadcast2Pitch: Game State Reconstruction from Unconstrained Soccer Videos" is listed as a WACV 2026 oral claiming state of the art on SoccerNet-GSR. — [WACV 2026 listing](https://wacv.thecvf.com/virtual/2026/oral/1201)

**Component building blocks referenced by these systems**
- PRTReID (Mansourian et al.): ReID mAP 72.59 / rank-1 89.57 on its SoccerNet-derived ReID test set (BPBreID 71.43 / 89.31); team affiliation mAP 92.89 / rank-1 97.60; role accuracy 94.27; two-cluster K-means team accuracy 91.87% from ReID features alone, 95.6% with joint ReID + team + role training. PRT-Track on SoccerNet-Tracking test with YOLOv8 detections: HOTA 59.77, AssA 58.55 (StrongSORT++ 56.21 / 50.27; ByteTrack 47.22 / 50.25). — [arXiv 2401.09942](https://arxiv.org/html/2401.09942v1)
- Koshkina & Elder jersey number pipeline (legibility classifier, ViTPose torso crop, PARSeq, Centroid-ReID outlier removal): 87.4% accuracy on SoccerNet tracklets, 91.4% on hockey images. — [arXiv 2405.13896](https://arxiv.org/abs/2405.13896)
- Its repository is licensed CC BY-NC 3.0 (non-commercial); weights (PARSeq original, hockey and SoccerNet fine-tunes, legibility classifier) are on Google Drive/OneDrive; 75 stars; last push 2024-10-07; requirements pin PyTorch 1.9. — [jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)
- PARSeq itself: Apache-2.0, last push 2024-05-29. — [parseq](https://github.com/baudm/parseq)
- BroadTrack (Magera et al., WACV 2025): camera tracking for broadcast; paper claims it halves mean reprojection error and gains more than 15% Jaccard index for calibration on SoccerNet versus prior methods; evaluated on sn-gamestate sequences. — [arXiv 2412.01721](https://arxiv.org/abs/2412.01721)
- BroadTrack code licence: EVS licence granting use "for noncommercial research purposes" only, with derivatives owned by the licensor; 26 stars; last push 2025-02-17. — [evs-broadcast/BroadTrack](https://github.com/evs-broadcast/BroadTrack)
- 2023 calibration challenge winner (Sportlight) code is public but has no licence file; 66 stars; last push 2024-12-30. — [soccernet-calibration-sportlight](https://github.com/NikolasEnt/soccernet-calibration-sportlight)

### Inferences
- A commercial product cannot ship the sn-gamestate baseline as-is: the top level is GPL-3.0, the detector is AGPL-3.0, the ReID models carry the Hippocratic licence, and two of the three calibration options are GPL-2.0. It is usable as a reference implementation and as an evaluation harness (TrackLab itself is MIT).
- At 11 minutes per 30 s clip on an A100, the baseline would need roughly 16 to 17 hours for a 45-minute half; it is a benchmark tool, not a production path. Constructor Tech's 80 FPS claim shows a real-time GSR system is feasible, but it is unreleased.
- The top of the leaderboard has barely moved between 2024 (63.81) and 2025 (63.90), and GS-DetA is around 50 even for the leaders while GS-AssA is 80 to 85. Detection plus localisation plus exact attribute match, not association, is what caps the score. The prototype's own profile (97% team accuracy, 27/29 two-digit numbers, but 47% recall and 5 to 21 m position error) is weak on exactly the part the leaders also find hardest.
- Constructor.Tech's 2025 row is numerically identical to its 2024 winning score, which suggests the 2024 entry was carried over rather than resubmitted; I could not confirm this.
- Since 2026 the community's identity problem has moved into action spotting (player-centric BAS), which is directly relevant to "feed player identity to event statistics".

### Gaps
- No public code or weights found for KIST-GSR, Constructor Tech, lianyou, Playbox & MIXI, UPCxMobius or Broadcast2Pitch (a GitHub search and web searches returned nothing). Broadcast2Pitch's GS-HOTA figure was not retrievable.
- The GSR dataset licence is ambiguous: the Hugging Face card says gpl-3.0 (unusual for data), and the underlying broadcast video comes from SoccerNet, whose videos are normally distributed under an NDA for research. I could not open the NDA text; commercial display of SoccerNet frames should be treated as not permitted until confirmed with SoccerNet.
- GS-HOTA's distance tolerance (I recall a Gaussian similarity with a 5 m scale) was not re-verified in this session.
- A third-party search excerpt of "Can Geometry Save Central Views for Sports Field Registration?" (arXiv 2504.20052) reports PnLCalib at only JaC@5 22.4 on a full-HD central-view SoccerNet subset, far below PnLCalib's own 80.6 on SN22-test-center. I could not open that paper to reconcile the protocols.

## 2. (A) Popular GitHub pipelines: roboflow/sports, tutorials, Eagle, Tryolabs and similar

### Takeaway
The popular repositories are tutorials or demos, not validated pipelines: none publishes detection, tracking or position accuracy on a benchmark, none reads shirt numbers or keeps identity across a match, and almost all depend on Ultralytics YOLO (AGPL-3.0) or have no licence at all. `roboflow/sports` is the most useful as permissively licensed glue code (MIT) and as a source of labelled pitch-keypoint data.

### Cited Findings
- `roboflow/sports`: MIT code, 5,506 stars, last push 2026-08-28, 34 commits, no PyPI package ("We don't have a Python package yet"). — [roboflow/sports](https://github.com/roboflow/sports)
- Its soccer example requires `ultralytics`; models are YOLOv8 player detection (players, goalkeepers, referees, ball), YOLOv8 pitch keypoints, and SigLIP + UMAP + KMeans for team classification; modes include PITCH_DETECTION, PLAYER_DETECTION, BALL_DETECTION and PLAYER_TRACKING. — [examples/soccer README](https://github.com/roboflow/sports/blob/main/examples/soccer/README.md)
- Its datasets are on Roboflow Universe (`football-players-detection-3zvbc`, `football-ball-detection-rejhg`, `football-field-detection-f07vi`) and the README says the original data comes from the DFL Bundesliga Data Shootout Kaggle competition. — [examples/soccer README](https://github.com/roboflow/sports/blob/main/examples/soccer/README.md)
- Forks of the players dataset on Roboflow Universe list CC BY 4.0. — [Roboflow Universe fork](https://universe.roboflow.com/chanjinkim/football-players-detection-3zvbc-x4y9s)
- `SkalskiP/sports` (the earlier personal tutorial repo): no licence file, 559 stars, last push 2023-12-12. — [SkalskiP/sports](https://github.com/SkalskiP/sports)
- `abdullahtarek/football_analysis`: no licence file, 1,032 stars, last push 2024-04-23. YOLO detection (a trained YOLOv5 on Google Drive), KMeans shirt-colour teams, optical-flow camera-motion compensation, perspective transform to metres, per-player speed and distance, team ball-possession share; requires `ultralytics` and `supervision`. No accuracy figures. — [football_analysis](https://github.com/abdullahtarek/football_analysis)
- `nreHieW/Eagle`: no licence file, 58 stars, last push 2026-05-13. Broadcast clip to per-frame pitch coordinates on a 105x68 pitch: YOLOv8 detectors (medium, large, large at 960 px), BoT-SORT, an HRNet model predicting 57 pitch keypoints, homography recomputed at a configurable rate with optical flow between; weights via `get_weights.sh`; example scripts for minimap, Voronoi, pass plot, trajectory. The README states it "is not 100% accurate" and recommends checking `annotated.mp4` before using the data; it works best when the camera position is stable. No benchmark numbers. — [Eagle](https://github.com/nreHieW/Eagle)
- Tryolabs: `tryolabs/norfair` tracker is BSD-3-Clause, 2,698 stars, last push 2025-04-30; the soccer demo `tryolabs/soccer-video-analytics` (automatic ball possession) is MIT, 307 stars, last push 2023-12-20. — [norfair](https://github.com/tryolabs/norfair), [soccer-video-analytics](https://github.com/tryolabs/soccer-video-analytics)
- `AtomScott/SportsLabKit` (formerly the SoccerTrack repo): GPL-3.0, 330 stars, last push 2023-12-19; in-house SORT, DeepSORT, ByteTrack and TeamTrack trackers, pluggable YOLOv8 / torch-ReID, 2D pitch calibration, bounding-box and coordinate DataFrame wrappers. — [SportsLabKit](https://github.com/AtomScott/SportsLabKit)
- `DonsetPG/narya`: MIT, 177 stars, last push 2023-03-25. — [narya](https://github.com/DonsetPG/narya)
- `nathankyryk/pitchview`: AGPL-3.0, 0 stars, last push 2026-11-02; described as soccer game-state reconstruction from broadcast video using RF-DETR, SAM 3.1, pitch mapping and jersey OCR, scored with GS-HOTA. — [pitchview](https://github.com/nathankyryk/pitchview)
- Permissive building blocks: `roboflow/supervision` MIT; `roboflow/rf-detr` Apache-2.0 (10,105 stars, pushed 2026-11-18); YOLOX Apache-2.0; RT-DETR Apache-2.0; SAHI MIT; BoT-SORT (original) MIT; ByteTrack MIT; deep-person-reid / OSNet MIT; CLIP-ReID MIT; ViTPose Apache-2.0. — [supervision](https://github.com/roboflow/supervision), [rf-detr](https://github.com/roboflow/rf-detr), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [sahi](https://github.com/obss/sahi), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [ByteTrack](https://github.com/FoundationVision/ByteTrack), [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid), [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- Copyleft building blocks to note: `mikel-brostrom/boxmot` AGPL-3.0; YOLOv9 GPL-3.0. — [boxmot](https://github.com/mikel-brostrom/boxmot), [yolov9](https://github.com/WongKinYiu/yolov9)

### Inferences
- The prototype's own stack (YOLO + BoT-SORT + kit colours + keypoint homography + PARSeq) is architecturally the same as these projects and already goes further than any of them on identity (shirt numbers, offline linking) and on measurement (it has been scored against SN-GSR labels; none of these has). Adopting one of them wholesale would not fix the measured problems.
- If the prototype's YOLO is Ultralytics, the AGPL question applies to its weights as well as its code. A cloud-hosted inference service is squarely in the cases Ultralytics lists as needing an Enterprise licence. RF-DETR, YOLOX and RT-DETR are Apache-2.0 alternatives; Playbox & MIXI (4th in GSR 2025) used RF-DETR.
- The Roboflow datasets being derived from the DFL Bundesliga Data Shootout Kaggle competition means the CC BY 4.0 label on Universe may not be the whole story; the competition's own data rules should be read before training commercial weights on them.

### Gaps
- Could not open the original Roboflow Universe pages (HTTP 403), so the licence of the three `roboflow-jvuqo` datasets and of the hosted model versions is not directly confirmed; only forks showing CC BY 4.0 were seen.
- "FootyVision" and "Soccer-Analytics" as named could not be matched to a notable repository: GitHub search returned only 0-to-1-star repositories called FootyVision (one MIT, one GPL-3.0, unrelated to each other).
- No published accuracy numbers exist for any repository in this section.

## 3. (A) Fixed wide-angle, panoramic and tactical camera resources

### Takeaway
For the eventual fixed full-pitch camera the most relevant open assets are SoccerTrack v2 (10 full matches, 4K panoramic, CC BY 4.0 data and MIT code, commercial use explicitly permitted) and Spiideo SoccerNet SynLoc (single calibrated static 4K camera, localisation scored in metres); the best published tracking result on full-pitch fisheye footage is only HOTA 0.60, so small-player detection, not calibration, is the hard part in wide views. Calibration for a static camera is done once from manual keypoint correspondences with lens-distortion coefficients, not per frame.

### Cited Findings
- SoccerTrack v2: 10 university-level full matches (about 900 minutes), 4K panoramic video from BePro systems (2 matches Cerberus, 8 matches 3-camera stitched), per-frame 2D pitch coordinates in metres, track IDs, jersey-based player IDs, roles, team sides, plus 12 ball-action classes and MOT-format boxes for a subset. — [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/)
- Licence: code MIT, dataset (videos and annotations) CC BY 4.0, "Both permit commercial use". Repo 43 stars, last push 2026-10-05; Hugging Face dataset `atomscott/soccertrack-v2` is cc-by-4.0, gated "auto", last modified 2026-09-24. — [SoccerTrack-v2 repo](https://github.com/AtomScott/SoccerTrack-v2), [Hugging Face dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2)
- Baselines are "scaffolding" only for GSR, BAS and MOT, with no pretrained weights and no published baseline scores; the journal paper with benchmarks is still in preparation. — [baselines README](https://github.com/AtomScott/SoccerTrack-v2/blob/main/baselines/README.md), [arXiv 2508.01802](https://arxiv.org/html/2508.01802v1)
- Its calibration procedure for the static panorama: load image-to-pitch keypoint correspondences, estimate camera parameters and distortion coefficients, compute a homography with RANSAC (threshold 3.0, min 10 matches), save undistortion maps (`_mapx.npy`, `_mapy.npy`), intrinsics and homography once per match, then apply to video and coordinates. — [docs/calibration.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/calibration.md)
- Ground-truth pitch coordinates are derived and refined from the synchronised BePro system's positional data. — [arXiv 2508.01802](https://arxiv.org/html/2508.01802v1)
- SoccerTrack Challenge 2025 (static fisheye footage): winner GTATrack scored HOTA 0.60 with 982 false positives, using YOLOv11x with multi-scale test-time augmentation and pseudo-labelling for small, distorted targets, then Deep-EIoU online tracking, then GTA-Link offline tracklet split-and-reconnect. — [arXiv 2602.00484](https://arxiv.org/abs/2602.00484), [GTATrack-STC2025](https://github.com/ron941/GTATrack-STC2025)
- GTATrack repo has no licence file (7 stars, last push 2026-01-29); it bundles Ultralytics (AGPL), Deep-EIoU (no licence file) and gta-link (MIT). — [GTATrack-STC2025](https://github.com/ron941/GTATrack-STC2025), [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU), [gta-link](https://github.com/sjc042/gta-link)
- SoccerTrack v1 (CVPRW 2022): 8K fisheye wide view and 4K drone top view of 11-v-11 college games with bounding boxes and GNSS tracking; v2's paper describes v1 as 30 s clips with 10 Hz GNSS, a single match, no jersey numbers or roles. — [CVPRW 2022 paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.html), [arXiv 2508.01802](https://arxiv.org/html/2508.01802v1)
- TeamTrack soccer subset: 30 minutes side view (Z CAM E2-F8 fisheye, 8K) and 30 minutes top view (DJI Mavic 3 drone, 4K), 1.24 million boxes each; fisheye distortion corrected with Zhang's calibration method. — [arXiv 2404.13868](https://arxiv.org/html/2404.13868v1)
- TeamTrack benchmarks: YOLOv8 mAP50:95 on soccer side view 11.2 pretrained versus 52.7 fine-tuned; top view 1.4 versus 23.5. Tracking HOTA side view BoT-SORT 58.4, ByteTrack 59.3; top view 51.9 and 53.7. — [arXiv 2404.13868](https://arxiv.org/html/2404.13868v1)
- TeamTrack code repo is MIT (80 stars, last push 2024-04-22); data on Kaggle and Google Drive, including a `teamtrack-trajectory` set already projected to pitch coordinates for the fixed side-view camera. — [TeamTrack](https://github.com/AtomScott/TeamTrack)
- Spiideo SoccerNet SynLoc: task is to localise athletes in real-world pitch coordinates from a single calibrated static-camera image (4K, with a full-HD variant); dev kit `sskit` on PyPI; data from research.spiideo.com after registration; metric mAP-LocSim with ground positions in metres. — [sskit](https://github.com/Spiideo/sskit)
- SynLoc 2026 results: SELab Soccer 97.67 mAP-LocSim (frame accuracy 81.91), PitchSeer 94.95, FC AllClip Research 94.70, Sarthi-GameChanger 94.05, baseline 77.30; 88 teams. The winner used YOLO26-Large detections to drive adaptive tiling, then RTMPose-X adapted to predict the pelvis and its ground projection, then ray casting onto the pitch plane. — [arXiv 2607.07320](https://arxiv.org/html/2607.07320)
- `Spiideo/sskit` has no licence file (29 stars, last push 2026-10-21); the baseline is in `Spiideo/mmpose` (Apache-2.0). — [sskit](https://github.com/Spiideo/sskit), [Spiideo/mmpose](https://github.com/Spiideo/mmpose)
- SoccerNet-Tracking (broadcast main camera, 30 s clips at 25 fps): 2023 challenge (detection + association) winner Kalisteo HOTA 75.61 (DetA 75.38, AssA 75.94); baseline 42.38. 2022 challenge with ground-truth detections: Kalisteo 93.64. — [sn-tracking](https://github.com/SoccerNet/sn-tracking)
- `sn-tracking`, `sn-jersey` and `MCG-NJU/SportsMOT` repositories have no licence file. — [sn-tracking](https://github.com/SoccerNet/sn-tracking), [sn-jersey](https://github.com/SoccerNet/sn-jersey), [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- SAHI (sliced inference for small objects) is MIT, 5,607 stars, pushed 2026-11-01; Norfair ships a SAHI demo. — [sahi](https://github.com/obss/sahi), [norfair](https://github.com/tryolabs/norfair)

### Inferences
- With a fixed camera the 5 to 21 m per-clip calibration error largely disappears as a class of problem: one manually checked calibration per installation (intrinsics, distortion, homography) replaces per-frame keypoint estimation. The remaining position error comes from the foot-point estimate; the SynLoc winner's use of a pose model to find the ground-projected pelvis instead of the box bottom-centre is the published best practice.
- TeamTrack's numbers (pretrained YOLOv8 at 1 to 11 mAP, 24 to 53 after fine-tuning) say an off-the-shelf detector will fail on wide views and that fine-tuning on the target camera is mandatory. SoccerTrack v2's CC BY 4.0 data is the only large, commercially usable training source found for this.
- Both the fisheye challenge winner (multi-scale TTA) and the SynLoc winner (adaptive tiling) rely on processing at higher effective resolution; SAHI-style slicing is the generic open equivalent. This multiplies detector cost, which matters for a 45-minute half on an L4/A10G.
- HOTA 0.60 for the best entry on full-pitch fisheye footage, versus 75.6 on broadcast SoccerNet-Tracking, means identity over a full half from a wide camera will fragment without an offline linking stage and a hard identity cap.

### Gaps
- ISSIA-CNR and Veo-specific open research were not checked in this session; nothing citable to add.
- No published benchmark numbers exist yet for SoccerTrack v2 (GS-HOTA, HOTA); the leaderboard is empty scaffolding as far as I could see.
- Dataset licence for TeamTrack, SoccerTrack v1 and SynLoc data was not stated in any page I could open (TeamTrack's repo is MIT, but that covers code; SynLoc requires registration).
- SportsMOT's soccer subset numbers were not separated out from the overall figures.

## 4. Which swaps would most plausibly fix the measured problems

### Takeaway
Published evidence points to three changes: fine-tune (or replace) the detector on football data and run it at higher effective resolution; replace the per-frame keypoint homography with a points-and-lines calibrator plus temporal smoothing (PnLCalib reports mean pitch projection error of 0.60 m on WorldCup14 and 0.23 m on TS-WorldCup); and add an offline global tracklet association stage with an identity cap (GTA-Link lifts SoccerNet HOTA from 79.41 to 83.11). The licence-clean options are narrower than the accurate ones.

### Cited Findings

**Detection (measured: recall 47%, precision 37%, split boxes)**
- The GSR baseline used YOLOv8 without football fine-tuning; with everything else replaced by ground truth, detection alone held GS-HOTA to 35.28, the lowest of any single-module oracle test. — [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- Fine-tuning YOLOv8 on target footage moved mAP50:95 from 11.2 to 52.7 (side view) and 1.4 to 23.5 (top view) on TeamTrack soccer. — [arXiv 2404.13868](https://arxiv.org/html/2404.13868v1)
- Detectors used by 2025 GSR leaders: YOLO-X (1st), YOLOv12 (3rd), RF-DETR (4th), fine-tuned YOLOv8x (8th). Winning GS-DetA was 51.36. — [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)
- SoccerNet-Tracking 2023 (detection + association) top DetA 75.38. — [sn-tracking](https://github.com/SoccerNet/sn-tracking)

**Calibration (measured: 5 to 21 m median error per clip)**
- PnLCalib camera calibration, SoccerNet-Calibration SN22-test-center: JaC@5 80.6, completeness 98.7, final score 79.5 (single-view + PnL refinement); TVCalib 54.8 / 100.0 / 54.8. SN23-test: JaC@5 78.7, completeness 78.4, final score 61.8. WorldCup14: JaC@5 85.9, completeness 100, final score 85.9 (TVCalib 39.9). — [PnLCalib paper, arXiv 2404.08401](https://arxiv.org/html/2404.08401v5)
- PnLCalib homography accuracy: WC14-test IoU-whole mean 93.4, projection error mean 0.60 m / median 0.42 m; TS-WorldCup IoU-whole 96.3, projection error mean 0.23 m / median 0.20 m. Prior methods on WC14 range 0.65 to 0.84 m mean. — [arXiv 2404.08401](https://arxiv.org/html/2404.08401v5)
- PnLCalib cost: 164 to 439 ms per frame depending on RANSAC configurations and keypoint subsets (networks on an RTX 2080 Ti, optimisation on a Xeon Silver 4214 CPU). — [arXiv 2404.08401](https://arxiv.org/html/2404.08401v5)
- In the GSR baseline, calibration was a major limiter: oracle tests gave 51.39 (calibration) and 49.99 (pitch) GS-HOTA with TVCalib. — [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- 2025 leaders did not use raw per-frame calibration: lianyou applied radial distortion correction, PnLCalib and temporal smoothing; Playbox & MIXI propagated BroadTrack calibration with optical flow; KIST used a multi-frame keypoint model; a calibration-only entry used NBJW with Lucas-Kanade tracking and RANSAC updates. — [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)
- BroadTrack claims half the mean reprojection error of prior methods on SoccerNet, but its licence is non-commercial research only. — [arXiv 2412.01721](https://arxiv.org/abs/2412.01721), [BroadTrack licence](https://github.com/evs-broadcast/BroadTrack)

**Identity fragmentation (measured: about 1.9 identities per real player)**
- GTA-Link (offline, tracker-agnostic: DBSCAN split of impure tracklets on ReID embeddings, then iterative merge by mean cosine distance, OSNet features trained on SportsMOT): SoccerNet HOTA 79.41 to 83.11; SportsMOT 81.04. MIT, 94 stars, last push 2025-12-12. — [arXiv 2411.08216](https://arxiv.org/abs/2411.08216), [gta-link](https://github.com/sjc042/gta-link)
- CAMELTrack (learned association, TrackLab-based): SportsMOT HOTA 80.3 with appearance + keypoint cues; weights on Hugging Face; Apache-2.0; 138 stars; last push 2026-11-10. — [CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack)
- Deep HM-SORT reports 80.1 HOTA on SportsMOT and 85.4 on SoccerNet-Tracking (seen in search excerpt of arXiv 2406.12081 only). — [arXiv 2406.12081](https://arxiv.org/html/2406.12081v1)
- PRT-Track with YOLOv8 detections: HOTA 59.77 on SoccerNet-Tracking test versus StrongSORT++ 56.21, ByteTrack 47.22; with ground-truth boxes 90.77. — [arXiv 2401.09942](https://arxiv.org/html/2401.09942v1)
- Structural constraints used by challenge teams: a hard cap of 24 identities (22 players + 2 referees); tracking in pitch coordinates with team constraints and a post-process that cut tracklet count by 90%. — [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1), [arXiv 2504.06357](https://arxiv.org/html/2504.06357v1)
- ReID options: PRTReID mAP 72.59 / rank-1 89.57 (Hippocratic licence); CLIP-ReID was used by two 2025 top-four teams (MIT). — [arXiv 2401.09942](https://arxiv.org/html/2401.09942v1), [arXiv 2508.19182](https://arxiv.org/html/2508.19182v1), [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID)

### Inferences
- Split boxes with precision below recall is characteristic of a detector not trained on this domain plus a permissive NMS; the highest-yield fix is fine-tuning on football boxes (SoccerNet-Tracking/GSR for broadcast, SoccerTrack v2 for wide view), then class-agnostic NMS, before changing tracker. Every later stage inherits this: 1.9 identities per player is partly two boxes per player.
- The prototype's 5 to 21 m error is one to two orders of magnitude worse than PnLCalib's published 0.2 to 0.6 m on World Cup broadcast benchmarks, so the calibrator is the single largest accuracy gap. Even allowing that benchmark frames are easier than arbitrary clips, a points-and-lines method with temporal smoothing across frames (reject frames with too few keypoints, interpolate camera parameters) is the published norm. PnLCalib's code is GPL-2.0-or-later (weights CC BY 4.0), so it can be run as a separate service or used to validate a clean-room keypoint + line approach; TVCalib (MIT) is the permissive but weaker option.
- A 1.6x distance overstatement is what jittering positions produce: a 5 m error that changes frame to frame adds path length on every frame. Fixing calibration stability and filtering positions (section 6) addresses distance more than any tracker change will.
- GTA-Link is the most direct, licence-clean (MIT) addition for fragmentation and overlaps with the prototype's existing "offline track linking"; the published split step (detecting two identities inside one tracklet) is the piece worth checking the prototype has. Shirt number plus kit group plus an 11-per-team cap gives a stronger merge key than appearance embeddings alone, and is what the challenge leaders exploit.
- At 164 to 439 ms per frame, PnLCalib on every frame of a 120 s 25 fps clip costs 8 to 22 minutes; keyframe calibration with optical-flow propagation (as Eagle and Playbox & MIXI do) is needed on an L4/A10G.

### Gaps
- No published head-to-head of these swaps on the same SN-GSR clips the prototype used; expected gains are inferred, not measured.
- No source quantifies how much of GS-HOTA loss is due to duplicate boxes specifically.
- CLIP-ReID and OSNet weights are trained on person-ReID datasets (e.g. Market-1501, MSMT17) whose terms are commonly research-only; I did not verify weight-level licences.

## 5. (B) Tracking-data analytics libraries: what each computes out of the box

### Takeaway
No single library covers the whole product list. The practical open stack is kloppy (loading and coordinate standardisation) + floodlight or databallpy (distance, speed, acceleration with filtering, metabolic power, centroid/stretch index/convex hull, space control, pressure, possession) + unravelsports (formation detection via EFPI, pressing intensity) + mplsoccer (pitches, heatmaps), all under permissive licences (BSD-3, MIT, MPL-2.0). None ships fixed speed-zone defaults at 19.8/25.2 km/h; zone thresholds are user inputs.

### Cited Findings

| Library | Licence, version, activity | What it computes |
|---|---|---|
| [kloppy](https://github.com/PySport/kloppy) | BSD-3-Clause; v3.19.1 (2026-10-03); 572 stars; pushed 2026-11-06 | Loading and standardising only: tracking from HawkEye, Metrica, PFF, SecondSpectrum, Signality, SkillCorner, Sportec, Stats Perform, Tracab; events from 8+ providers; coordinate-system and orientation transforms; export to pandas/Polars. No physical or tactical metrics. |
| [floodlight](https://github.com/floodlight-sports/floodlight) | MIT; 1.2.0 (2026-05-04); 135 stars | Distance and cumulative distance, velocity, acceleration; metabolic power and equivalent distance; team centroid, distance to centroid, stretch index; nearest-mate distance and team spread; nearest-opponent distance; convex hull area; discrete Voronoi space control; approximate entropy. Filters: Butterworth, Savitzky-Golay, FIR, Kalman, Wiener. |
| [databallpy](https://github.com/Alek050/databallpy) | MIT; v0.8.1 (2026-08-11); 106 stars; pushed 2026-11-03; JOSS paper | Velocity and acceleration with filtering; covered distance in user-defined velocity and acceleration zones; pressure (Herold & Kempe 2022); team possession and individual player possession; event-to-tracking synchronisation (Needleman-Wunsch); Voronoi and Gaussian space occupation; pitch-control/influence module citing Fernandez & Bornn "Wide Open Spaces"; optional Dangerous Accessible Space; simple xG, xT. |
| [unravelsports](https://github.com/UnravelSports/unravelsports) | MPL-2.0; v1.2.1 (2026-01-13); 250 stars | Kloppy-to-Polars dataset with velocity/acceleration; pressing intensity (Bekkers 2024, time-to-intercept based); EFPI formation and position identification (Bekkers 2025), 65 formation templates by default; graph conversion for GNN training. |
| [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) | MIT; 367 stars; last push 2022-11-03 (unmaintained tutorial code) | Metrica loaders and plotting; Savitzky-Golay velocities; distance and distance by speed band; sustained-sprint counts; Spearman (2018) pitch control; EPV-weighted passing options. |
| [accessible-space](https://github.com/jonas-bischofberger/accessible-space) | MIT; 5 stars; pushed 2026-10-30; paper in J Big Data 13, 76 (2026) | Pass-simulation model: expected pass completion (xC), accessible space (AS) and dangerous accessible space (DAS) per frame for team and individual, AS/DAS gained per pass; provider-agnostic pandas interface needing x, y, vx, vy and team in possession. |
| [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) | MIT; 555 stars; pushed 2026-10-27 | Plotting only: pitches in Matplotlib and StatsBomb open-data loading. |
| [socceraction](https://github.com/ML-KULeuven/socceraction) | MIT; 829 stars; pushed 2026-10-18 | Event-data valuation (SPADL, VAEP, xT); not a tracking-data library. |

**Details that matter for physical metrics**
- floodlight kinematics: central difference (recommended) or backward difference; `butterworth_lowpass` defaults order 3, Wn 1; `savgol_lowpass`, `fir_lowpass`, `kalman`, `wiener` also provided. — [floodlight transforms/filter.py](https://github.com/floodlight-sports/floodlight/blob/main/floodlight/transforms/filter.py), [floodlight models/kinematics.py](https://github.com/floodlight-sports/floodlight/blob/main/floodlight/models/kinematics.py)
- floodlight metabolic power follows di Prampero & Osgnach (2018) with the Minetti & Pavei (2018) energy-cost update. — [floodlight models/kinetics.py](https://github.com/floodlight-sports/floodlight/blob/main/floodlight/models/kinetics.py)
- floodlight geometry and space models (CentroidModel with `stretch_index`, NearestMateModel `team_spread`, ConvexHullModel `convex_hull_area`, DiscreteVoronoiModel `player_controls` / `team_controls`). — [floodlight models docs](https://floodlight.readthedocs.io/en/latest/modules/models/models.html)
- floodlight IO: Tracab, DFL/STS, Kinexon, Opta, Second Spectrum, SkillCorner, Sportradar, StatsPerform, StatsBomb; bundled public datasets EIGD-H (handball), IDSSE (football tracking + events), StatsBomb open data. — [floodlight](https://github.com/floodlight-sports/floodlight)
- databallpy filter: Savitzky-Golay (default window 7, polyorder 2) or moving average; covered distance sums per-frame distance inside each user-supplied (min, max) velocity or acceleration interval. — [databallpy features/filters.py](https://github.com/Alek050/databallpy/blob/develop/databallpy/features/filters.py), [databallpy features/covered_distance.py](https://github.com/Alek050/databallpy/blob/develop/databallpy/features/covered_distance.py)
- databallpy loads Tracab, Metrica, Inmotio, Sportec (DFL) tracking plus anything kloppy supports; `get_open_game()` loads the seven DFL games under CC-BY 4.0. — [databallpy](https://github.com/Alek050/databallpy)
- unravelsports defaults: Savitzky-Golay smoothing on by default (players window 7, polyorder 1; ball window 3, polyorder 1); caps of 12 m/s player speed, 28 m/s ball speed, 6 m/s² player acceleration. — [unravelsports kloppy_polars.py](https://github.com/UnravelSports/unravelsports/blob/main/unravel/soccer/dataset/kloppy_polars.py)
- EFPI: assigns players to predefined static formation templates by linear sum assignment after scaling positions to template dimensions, picks the lowest-cost formation; runs per frame, possession, period or time window, with a stability threshold to suppress flicker. — [EFPI summary of arXiv 2506.23843](https://www.emergentmind.com/papers/2506.23843), [unravelsports](https://github.com/UnravelSports/unravelsports)
- LaurieOnTracking velocities: `calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed=12)`; raw speeds above 12 m/s are set to NaN before smoothing. — [Metrica_Velocities.py](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Velocities.py)
- LaurieOnTracking speed bands: walking < 2 m/s, jogging 2 to 4, running 4 to 7, sprinting ≥ 7 m/s; a sustained sprint is ≥ 7 m/s held for at least 1 s (25 frames). — [Tutorial2_DelvingDeeper.py](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Tutorial2_DelvingDeeper.py)
- LaurieOnTracking pitch control defaults: max player speed 5 m/s, reaction time 0.7 s, tti_sigma 0.45, average ball speed 15 m/s, kappa_def set to 1 (1.72 in Spearman's paper). — [Metrica_PitchControl.py](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_PitchControl.py)
- SportsLabKit offers trackers and DataFrame wrappers but is GPL-3.0 and last pushed 2023-12. — [SportsLabKit](https://github.com/AtomScott/SportsLabKit)

**Coverage against the product's list**

| Product need | Open implementation found |
|---|---|
| Trajectories, distance, speed, acceleration | floodlight, databallpy, unravelsports, LaurieOnTracking |
| Speed zones | databallpy (user-defined intervals); LaurieOnTracking example bands; none ships 19.8/25.2 km/h defaults |
| Metabolic power | floodlight only |
| Heatmaps | mplsoccer for drawing; no library computes them as a named metric |
| Centroid, stretch index, convex hull, spread | floodlight |
| Formation / roles | unravelsports EFPI |
| Pitch control / space | LaurieOnTracking (Spearman), databallpy (Fernandez & Bornn influence, Voronoi), floodlight (discrete Voronoi), accessible-space (DAS) |
| Pressing / pressure | unravelsports (pressing intensity), databallpy (Herold & Kempe pressure) |
| Possession from tracking | databallpy (team and individual player possession) |

### Inferences
- The libraries all assume clean, provider-grade tracking (stable IDs, 10 to 25 Hz, centimetre-to-decimetre noise). Fed positions with metre-level jitter, their default 7-frame Savitzky-Golay windows will not prevent inflated distance; the prototype needs heavier position filtering first, then these tools.
- Every listed metric the product needs has a permissively licensed implementation, so the analytics layer carries no licence risk; the risk is entirely in the vision layer (section 1 and 2).
- Writing kloppy-compatible output (e.g. Metrica or SkillCorner-style format) from the prototype would make all of these libraries usable without adapters, and lets the same analytics run on open provider data for demonstration.
- Team length/width and line-breaking metrics are not named functions in any library I inspected; they are simple derivations (min/max extents; pass segments crossing defensive lines) that would need to be written.

### Gaps
- Open implementations of Shaw & Glickman (2019) formation clustering and Bialkowski role assignment were not found (search returned nothing reliable); EFPI in unravelsports is the only maintained formation detector identified.
- databallpy's exact public function names and whether its pitch-control module is documented as stable were not checked beyond the source file.
- pysport.org listings were not reviewed directly; SkillCorner's own tooling beyond the open-data repo was not checked.
- mplsoccer's latest version number could not be read from the API (no GitHub releases).

## 6. Accuracy of video-derived distance and speed, and standard processing before reporting

### Takeaway
Even validated multi-camera optical systems are accurate for total distance (well under 3% error) but poor for high-speed and sprint distance (errors of roughly 9% to over 40%), and independently tested commercial broadcast-tracking products showed total-distance bias from −21.8% to +24.3% against TRACAB; a 1.6x overstatement is far outside any published tolerance. Sports-science reviews find thresholds are not standardised (19.8 and 25.2 km/h are merely the most common) and that filtering and minimum effort duration are rarely reported yet change sprint counts materially.

### Cited Findings
- TRACAB validation against VICON (Linke, Link & Lames 2020, PLOS ONE): position RMSE 0.09 m (Gen4) and 0.08 m (Gen5); instantaneous speed RMSE 0.09 and 0.08 m/s; total distance deviation 0.42 ± 0.60% (Gen4) and 0.27 ± 0.35% (Gen5); Gen4 low-speed distance −19.41 ± 13.24% and high-speed distance 8.94 ± 9.49%; peak speed and acceleration differences < 0.5%; frames where players were clearly not tracked were excluded. (Figures from search excerpt of the paper.) — [Linke et al. 2020, PMC7064167](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064167)
- EPTS comparison against VICON (Linke et al. 2018, PLOS ONE): total distance error in small-sided games 2.2% (GPS), 2.7% (video), 4.0% (LPS); instantaneous speed error 0.28 ± 0.07 m/s (GPS), 0.25 ± 0.06 m/s (LPS), 0.41 ± 0.08 m/s (video); error grew with speed for all systems; high-speed distance showed > 40% deviations for each technology. (Figures from search excerpt.) — [Linke et al. 2018, PMC6056042](https://pmc.ncbi.nlm.nih.gov/articles/PMC6056042)
- Independent test of commercial computer-vision broadcast-tracking software against TRACAB Gen5 on one 2022 World Cup match (SAGE, 2026): "fair precision" for speed and, to a lesser extent, position; total distance bias ranged from −1745 m (−21.8%) to 1945 m (24.3%) across providers; accuracy was best when players were actually detected rather than inferred. (Publisher page returned 403; figures from search excerpt.) — [SAGE 2026 study](https://journals.sagepub.com/doi/10.1177/22150218261445834)
- SkillCorner states of its own open broadcast-tracking data: "Around 97% of the player identity we provide are accurate" and that some speed and acceleration data needs smoothing; each player has an `is_detected` flag distinguishing observed from extrapolated positions. — [SkillCorner opendata](https://github.com/SkillCorner/opendata)
- Scoping review of high-intensity locomotion in professional men's soccer (Biology of Sport): of 20 studies, 10 used about 19.8 km/h (5.5 m/s) for high-speed running and 10 used 25.2 km/h (7 m/s) for sprinting; other choices were 20, 21 and 21 to 24 km/h for HSR and 24 or 25 km/h for sprint; only 2 of 20 reported filtering and 3 of 20 a minimum effort duration; 19 of 20 used GPS. — [Biology of Sport scoping review](https://www.termedia.pl/Current-approaches-to-measuring-high-intensity-locomotor-actions-in-adult-male-professional-soccer-A-scoping-review,78,57387,1,1.html)
- Same review: "no standardised agreement currently exists between GPS, LPS, and OTS technologies"; optical tracking is generally reliable for distance but HSR and sprint distances can be 12 to 18% higher than GPS despite r² > 0.99; optical tracking remains unvalidated by FIFA for accelerations and decelerations; it recommends standardising thresholds first and requiring transparency on smoothing and interpolation. — [Biology of Sport scoping review](https://www.termedia.pl/Current-approaches-to-measuring-high-intensity-locomotor-actions-in-adult-male-professional-soccer-A-scoping-review,78,57387,1,1.html)
- A 10 Hz GPS study found the velocity filter (median or exponential) and the minimum effort duration both changed the number of high-speed, sprint and acceleration efforts detected, with differences once the minimum duration was under 0.5 s (0.7 s for accelerations); authors advise consistent processing. — [Varley et al., methodological considerations](https://read.qxmd.com/read/28051343/methodological-considerations-when-quantifying-high-intensity-efforts-in-team-sport-using-global-positioning-system-technology)
- FIFA's EPTS programme tests position and velocity accuracy by velocity band against VICON (30 x 30 m area) and Victoria University's Vision Kit (full pitch); the test institute has extended testing to broadcast-derived tracking under the "FIFA Basic" mark. — [FIFA EPTS testing process](https://inside.fifa.com/innovation/standards/epts/epts-testing-process)
- Open-source processing conventions: cap player speed at 12 m/s and treat above as outliers, then Savitzky-Golay smooth velocities over 7 frames at 25 Hz (LaurieOnTracking, unravelsports, databallpy defaults); floodlight recommends central differences and offers Butterworth low-pass; sprint counted only when sustained ≥ 1 s. — [Metrica_Velocities.py](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Velocities.py), [unravelsports kloppy_polars.py](https://github.com/UnravelSports/unravelsports/blob/main/unravel/soccer/dataset/kloppy_polars.py), [floodlight kinematics](https://github.com/floodlight-sports/floodlight/blob/main/floodlight/models/kinematics.py), [Tutorial2_DelvingDeeper.py](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Tutorial2_DelvingDeeper.py)

### Inferences
- A defensible reporting pipeline, assembled from the above: (1) drop or flag extrapolated/undetected samples rather than integrating through them; (2) low-pass filter positions before differentiating; (3) differentiate with central differences; (4) cap speed (12 m/s) and acceleration (about 6 m/s²) and treat exceedances as tracking errors; (5) smooth velocity; (6) compute distance from filtered speed; (7) require a minimum dwell (0.5 to 1 s) before counting a high-speed run or sprint; (8) state thresholds explicitly (19.8 and 25.2 km/h as the most-cited convention) and the filter used.
- The prototype should validate distance the way the SAGE study did: per-player total distance and distance per zone against a reference on the same clip. SN-GSR labels give reference pitch positions for 30 s clips, and SoccerTrack v2 gives full-match reference positions for wide-angle footage, so a per-clip distance ratio can be published alongside each metric.
- Because even reference-grade optical systems are 9 to 40%+ off for high-speed distance, sprint distance and sprint counts from a single broadcast camera should be labelled as estimates, and acceleration-based metrics (and metabolic power, which depends on acceleration) should not be presented as measurements.
- Total distance from broadcast clips also suffers from players being out of frame; the `is_detected` convention SkillCorner uses is the honest way to expose this.

### Gaps
- Could not open the full text of Linke 2018/2020 or the SAGE 2026 paper (bot checks / 403); figures above come from search-result excerpts of those pages and should be re-verified before quotation.
- No primary source found recommending a specific Butterworth cut-off frequency for optical football tracking, nor a FIFA document giving numeric pass/fail accuracy limits by velocity band.
- A claim seen in one search excerpt that UEFA/FIFA official thresholds are 19 and 23 km/h could not be traced to a source and is not relied on.
- Taberner et al. and other named validation studies were not retrieved.
- No independent, peer-reviewed validation of SkillCorner specifically against GPS was found; SkillCorner's own validation at Sunderland is self-reported without numbers.

## 7. Open tracking datasets usable to demonstrate the analytics honestly

### Takeaway
Three sources are clearly licensed for commercial demonstration with attribution: the DFL/IDSSE Bundesliga set (7 full matches, CC-BY 4.0), SoccerTrack v2 (10 full matches with video, CC BY 4.0) and SkillCorner open data (10 A-League matches, repository under MIT). Metrica and Last Row are permitted "with credit" but under informal terms; PFF FC World Cup 2022 and anything derived from SoccerNet broadcast video have unclear or research-oriented terms.

### Cited Findings
- DFL / IDSSE (Bassek, Rein, Weber & Memmert, Scientific Data 12:195, 2025): official match information, event and position data for seven matches from the German Bundesliga first and second divisions (databallpy describes them as two 1. Bundesliga and five 2. Bundesliga games); licensed CC-BY 4.0; hosted on Figshare with a Hugging Face mirror for kloppy; no broadcast video. — [Bassek et al. 2025, PMC11787359](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/), [idsse-data](https://github.com/spoho-datascience/idsse-data), [databallpy](https://github.com/Alek050/databallpy)
- SkillCorner open data: 10 matches from the 2024/2025 Australian A-League; broadcast-derived tracking at 10 fps in metres with `is_detected` flags; dynamic events and phases of play per match; season physical, off-ball-run and passing aggregates; 3D pose (29 joints, 25 fps) for two matches; repository licence MIT; README asks users to credit SkillCorner; last push 2026-09-14. — [SkillCorner opendata](https://github.com/SkillCorner/opendata)
- Metrica sample data: two anonymised games in CSV (tracking + synchronised events, 105 x 68 m pitch, coordinates 0 to 1) plus a third in EPTS/FIFA format; no licence file; terms are "Please be responsible with the use of this data" and "If you use it for anything public, please aknowledge the source"; last push 2021-04-15. — [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)
- PFF FC (now Gradient Sports) 2022 World Cup: broadcast tracking plus event data for all 64 matches, with rosters and metadata; obtained by submitting a form; the release page states no licence and only asks users to tag @PFF_FC when publishing; a Terms & Conditions link exists but was not readable. — [Gradient/PFF release post](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data)
- SoccerTrack v2: see section 3; CC BY 4.0 for video and annotations with explicit statement that commercial use is permitted. — [SoccerTrack-v2 repo](https://github.com/AtomScott/SoccerTrack-v2)
- SoccerNet-GSR: 200 x 30 s broadcast clips with pitch positions, roles, teams and jersey numbers; Hugging Face card says gpl-3.0, not gated. — [SN-GSR-2025](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025), [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- Last Row: 19 Liverpool goals from 2019 at 20 fps; shared "for educational (and recreational) purposes"; the README says it "lacks the accuracy needed for research", not all players are included, and IDs are not consistent between plays; "Feel free to use the dataset for whatever purpose, but please credit the source"; no licence file; last push 2020. — [Last-Row](https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row)
- kloppy links public data for Metrica, SkillCorner, Sportec/IDSSE and PFF and has direct loaders (e.g. `sportec.load_open_event_data`). — [kloppy](https://github.com/PySport/kloppy)

### Inferences
- For showing physical metrics honestly, the IDSSE set is the best fit: official optical tracking, full 90 minutes, all 22 players, a clear CC-BY licence, and loaders in kloppy, floodlight and databallpy. It lets the analytics layer be demonstrated independently of the vision layer's current accuracy.
- SoccerTrack v2 is the only open set that pairs commercially usable video with reference positions, so it is the right basis for an end-to-end accuracy claim for the fixed-camera target.
- SkillCorner's set is the closest analogue to the prototype's own output (single broadcast camera, partial visibility, extrapolated players) and is useful to show what broadcast-derived tracking realistically supports. Note the MIT licence is on the repository; it is unusual for data, and team/player names and league marks are separate rights.
- For Metrica, Last Row and PFF, public demonstration with attribution appears to be within the stated wishes, but none grants explicit commercial rights; using them inside a paid product should be cleared with the owners.

### Gaps
- PFF's actual terms and conditions were not readable; tracking frame rate not stated on the release page.
- SoccerNet NDA text and whether SN-GSR frames may be shown commercially were not confirmed.
- The season of the IDSSE matches (a Hugging Face card says 2022/23, TRACAB) was not verified against the paper.
- Whether the 2020 SkillCorner release (9 matches, mentioned in Metrica's README) is still distributed alongside the 2024/25 A-League matches was not checked.
