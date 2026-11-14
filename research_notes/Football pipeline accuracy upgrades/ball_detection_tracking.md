# Football ball detection and ball trajectory tracking (broadcast and wide tactical camera), state as of Nov 2026

Reading notes for the report writer:
- "Verified" below means I read the number on the primary source (paper or repo page) in this session. Anything marked "from memory, not verified this session" was NOT checked and must be treated as a lead, not a fact.
- No source I found reports a head-to-head benchmark of heatmap trackers vs. sliced YOLO vs. DETR-family on the same football ball test set. All cross-family comparisons below are inference.
- The most important single caveat: the only heatmap ball-tracker with published soccer weights (WASB and its re-implemented baselines) was trained and tested on ONE game from fixed cameras (ISSIA-CNR-derived), so its headline numbers do not describe generalisation to arbitrary broadcast or Veo/BePro-style footage.

## 1. Heatmap / temporal ball detectors (TrackNet v2/v3/v4, WASB, MonoTrack, BlurBall, TOTNet, DeepBall, FootAndBall): soccer weights, accuracy, speed

### Takeaway
WASB-SBDT (MIT) is the only repo found that ships soccer-trained weights for a whole family of heatmap trackers (WASB, TrackNetV2, ResTrackNetV2, MonoTrack, DeepBall, DeepBall-Large, BallSeg); on its soccer test split WASB reaches F1 88.3 / AP 83.6 at a 4 px threshold versus 86.6 / 77.2 for TrackNetV2, at about 56 FPS on a V100. The newer TrackNetV3, TrackNetV4, TOTNet and BlurBall have no soccer results or soccer weights; they are racket-sport models that would need retraining.

### Cited Findings
- WASB-SBDT repo is MIT-licensed; it provides pretrained WASB models plus evaluation code for TrackNetV2, ResTrackNetV2, MonoTrack, DeepBall, DeepBall-Large and BallSeg; paper is arXiv 2311.05237, BMVC 2023 — [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)
- The model zoo lists downloadable soccer weights (Google Drive) for all seven models: DeepBall, DeepBall-Large, BallSeg, TrackNetV2, ResTrackNetV2, MonoTrack, WASB; a setup script downloads them all — [WASB-SBDT MODEL_ZOO.md](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md)
- WASB "Soccer" dataset: derived from D'Orazio et al. 2009 (six synchronised fixed cameras, the ISSIA-CNR recording), 1920x1080, 25 FPS, a single game. Clips 1-4 train (11,994 frames), clips 5-6 test (5,999 frames). Original ball annotations were unusable so the authors re-annotated the ball in all frames — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- Metric: per-frame TP/TN/FP/FN by a distance threshold tau; headline table uses tau = 4 px (measured at the network's 288x512 working resolution per the paper's protocol); F1, Accuracy, AP reported — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- Verified soccer results at tau = 4 px (params, F1, Acc, AP, FPS):
  - DeepBall: 0.1M, 44.5, 92.7, 26.3, 44.6 FPS
  - DeepBall-Large: 1.0M, 44.9, 89.5, 34.0, 42.0 FPS
  - BallSeg: 12.7M, 36.1, 92.6, 20.0, 64.8 FPS
  - TrackNetV2: 11.3M, 86.6, 97.7, 77.2, 66.0 FPS
  - ResTrackNetV2: 1.2M, 84.6, 97.4, 75.5, 56.2 FPS
  - MonoTrack: 2.9M, 85.2, 97.4, 78.6, 58.0 FPS
  - WASB (step=3): 1.5M, 88.3, 97.9, 83.6, 55.7 FPS
  - WASB (step=1, every frame predicted in all three MIMO positions): 1.5M, 88.2, 97.9, 86.2, 23.6 FPS
  — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- Speeds were measured on an Ubuntu server with 4 V100 GPUs, not an L4 — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- WASB design: frames resized to 288x512; 3 consecutive frames concatenated on the channel axis, 3 heatmaps out (MIMO); small 4-stage HRNet-style backbone with strides removed from the stem; "hard-to-localize sample mining" from epoch 20 — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- Removing stem strides (keeping full resolution features) is the single largest ablation gain on soccer: F1 81.7 -> 86.4 -> 88.3 and AP 71.7 -> 79.0 -> 83.6 for original / partial / full stride removal — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- WASB's own online tracker (constant-acceleration prediction + distance gate + highest-confidence pick) "does not help on Soccer or Badminton" according to the paper; and WASB predicts at most one ball per frame — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- A third-party leaderboard mirror shows the same soccer numbers (WASB step=1 F1 88.2, AP 86.2; DeepBall F1 44.5) — [hyper.ai SOTA mirror](https://hyper.ai/en/sota/tasks/sports-ball-detection-and-tracking/benchmark/sports-ball-detection-and-tracking-on-1) (aggregator; primary source is the paper)
- TrackNetV3 (qaz812345): MIT licence covering code and checkpoints; badminton shuttlecock only; tracking module trained with sequence length 8, InpaintNet rectification module with sequence length 16 (builds repair masks from the predicted trajectory and inpaints occluded stretches); reported 97.51% accuracy / 98.56% F1 / 97.79% precision / 99.33% recall on its badminton test split at 25.11 FPS (TrackNetV2 27.70 FPS in the same table). No soccer support mentioned — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- TrackNetV4 (ICASSP 2025) adds motion-attention maps as a plug-in to heatmap trackers; evaluated on tennis and shuttlecock data only; no soccer results found — [TrackNetV4 project page](https://time.griffith.edu.au/paper-sites/tracknet-v4)
- TOTNet (arXiv 2508.09650, Aug 2025): evaluated on tennis, table tennis and badminton; soccer is not evaluated. 5 input frames at 288x512; 28.08 FPS with 8.65M params (optical-flow variant 12.19 FPS); in the same table WASB 33.44 FPS and TrackNetV2 40.74 FPS. On fully occluded frames it is far better than WASB (tennis full-occlusion RMSE 27.98 vs 264.45; TTA full-occlusion accuracy 0.74-0.80 vs 0.63), but on badminton "not visible" frames TrackNetV2 beats it (acc 0.88 vs 0.84). Code at AugustRushG/TOTNet; weights availability and code licence not stated in the paper — [TOTNet paper](https://arxiv.org/html/2508.09650v1)
- BlurBall jointly estimates ball position and motion blur; it targets table tennis, not soccer — [BlurBall paper](https://arxiv.org/pdf/2509.18387v2)
- DeepBall and FootAndBall were both developed and evaluated on ISSIA-CNR (six Full-HD long-shot cameras, ~20,000 annotated frames with ball positions and player boxes); FootAndBall's second dataset (Soccer Player Detection, 2,019 images at 1280x720) has no ball annotation — [FootAndBall paper](https://arxiv.org/pdf/1912.05445); [DeepBall paper](https://arxiv.org/pdf/1902.07304)

### Inferences
- In WASB's controlled re-implementation DeepBall-style single-frame heatmap nets collapse on soccer (F1 ~45) while every 3-frame model scores 85-88. That is direct evidence that temporal input, not architecture detail, is what recovers a small/blurred football. The current pipeline is purely single-frame, which is consistent with its 56-75% candidate rate.
- The WASB soccer weights are a narrow-domain checkpoint (one game, one stadium, fixed cameras, ~6k test frames from the same game). Expect a clear drop on unseen broadcast or tactical footage; treat the weights as an initialisation for fine-tuning, not as a drop-in. The architecture and training code (MIT) are the reusable asset.
- At 288x512 a full 1080p frame is downscaled 3.75x, so a 10 px ball becomes under 3 px. For wide tactical cameras (4K panoramas, ball often 5-12 px) WASB should be run on native-resolution crops (e.g. a 512x288 or 1024x576 window around the last known ball / predicted position, plus periodic global tiles), which fits the existing "recent-ball crop search + periodic global reacquisition" structure.
- Throughput: a 45-minute half at 25 FPS is 67,500 frames. At WASB's published ~56 FPS (V100, step=3) one pass is about 20 minutes; an L4 is typically somewhat slower than a V100 in FP32, so batching and FP16/TensorRT would be needed to stay near real time. This is an estimate; no L4 measurement was found.
- TOTNet's occlusion results suggest a 5-frame, occlusion-aware model is the right direction if retraining anyway, but the numbers are from racket sports where the ball is hidden for a few frames; football occlusions (ball under a player for seconds) are a different regime that no heatmap paper evaluates.

### Gaps
- No public evaluation of WASB/TrackNet-family soccer weights on SoccerNet-Tracking, SoccerTrack or any broadcast set was found.
- No L4 speed measurement for any of these models was found.
- The pixel threshold in WASB is in resized-input pixels as I read the protocol; the summary I obtained did not state this explicitly, so confirm in the paper before quoting "4 px" against 1080p coordinates.
- MonoTrack's original paper and any football-specific heatmap ball networks from 2024-2026 beyond WASB were not located in this session.
- Licence of the ISSIA-CNR source data (and hence of weights trained on it) was not found; the DeepBall paper only calls it "publicly available".

## 2. Slicing/tiling and high-resolution detectors (SAHI, Roboflow InferenceSlicer, P2 heads, RF-DETR / RT-DETR / YOLO11/12/26 / D-FINE)

### Takeaway
The best-documented open football recipe is Roboflow's: a dedicated single-class ball detector (YOLOv8x) trained on 2x2 tiles at 640 px and run with sliced inference, reaching mAP@50 0.925 on its own validation set while its author still reports frequent false negatives. No source gives a football-ball recall comparison between slicing, P2 heads, and DETR-family models, so the choice among them cannot be settled from literature.

### Cited Findings
- Roboflow's ball model: YOLOv8x, 50 epochs, trained and inferred at 640x640 on tiles (each frame split 2x2, each tile resized to 640); data extracted at 3 FPS from the DFL Bundesliga Data Shootout Kaggle videos; reported mAP@0.50 0.925 and mAP@0.5:0.95 0.57; "confusion matrix shows the model still makes errors, especially false negatives"; deployed id `football-ball-detection-rejhg/4`, run at confidence 0.3 — [Roboflow blog, ball tracking](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- Slicing settings used: `sv.InferenceSlicer` with slice size (w/2 + 100, h/2 + 100), 100 px overlap, NMS IoU 0.1 across tiles; the post states slicing reduces speed (one forward pass per tile) but gives no FPS — [Roboflow blog, ball tracking](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- Stated failure modes: small ball at distance, motion blur, cluttered background (crowd), look-alikes such as player heads, lighting/shadow — [Roboflow blog, ball tracking](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- roboflow/sports is MIT-licensed, ships no weights in the repo, links three Roboflow Universe datasets (players, ball, pitch keypoints) and lists ball tracking as an open challenge: "Tracking the ball is extremely difficult due to its small size and rapid movements, especially in high-resolution videos" — [roboflow/sports](https://github.com/roboflow/sports)
- SAHI is evaluated with YOLOv5/YOLOX in a dedicated study of sliced inference for small objects — [Evaluation of YOLO Models with Sliced Inference](https://arxiv.org/pdf/2203.04799); a 2025/26 adaptive-slicing paper reports +1.7% mAP over baseline SAHI on VisDrone with 20-25% faster inference — [arXiv 2512.07379](https://arxiv.org/pdf/2512.07379) (aerial imagery, not football)
- The 2023 SoccerNet Tracking 3rd-place team (MOT4MOT) treated the ball as single-object detection with a fine-tuned YOLOv8l and "proprietary filtering" to raise precision; overall HOTA 66.27; no ball-specific metric published — [MOT4MOT report](https://arxiv.org/abs/2308.16651)
- A 2025 study of Kalman-based tracking for fast tiny balls lists YOLOv8 recall 0.80 and YOLOv5 recall 0.845 on its ball data (not football-specific) — [arXiv 2509.18451](https://arxiv.org/pdf/2509.18451)
- A non-football small-object benchmark (2026, road users) found RF-DETR best overall (mAP 0.726 vs YOLO11 0.687), with YOLO11 nearly closing the gap on small objects — [MicroVision, arXiv 2603.18192](https://arxiv.org/pdf/2603.18192)
- A sports ball-detection review attributes single-shot detector weakness on balls to low recall on small objects — [PeerJ CS review of ball detection](https://peerj.com/articles/cs-3079/)

### Inferences
- The existing pipeline (multi-class YOLO at 1280 + low-threshold raw ball path + COCO sports-ball rescue) differs from the Roboflow recipe in one important way: the ball shares a model with three large person classes. A dedicated single-class ball detector trained on tiles (ball occupies more pixels, loss not dominated by players) is the cheapest change with documented evidence, and it is compatible with the current tiling code.
- The COCO "sports ball" rescue model is trained on close-up consumer photos; it is unlikely to add recall on 6-12 px footballs and is a plausible source of false positives. Replacing it with a football-tile-trained ball model or a temporal heatmap model is better supported by the evidence above.
- Cost model for tiling: 2x2 tiles at 640 = 4 forward passes per frame (270,000 for a 25 FPS half), 3x2 at 640 for 1080p = 6. A ROI-first strategy (one tile centred on the predicted ball position, full tile sweep only every N frames or when the track is lost) reduces this by roughly 3-5x; this is already the shape of the current rescue path.
- A stride-4 (P2) detection head attacks the same problem as slicing (feature-map resolution at the ball) in one forward pass. WASB's stride-removal ablation (+6.6 F1, +11.9 AP on soccer) is indirect evidence that resolution of the feature map at the ball matters a great deal, but there is no football measurement of P2 vs. slicing.
- DETR-family detectors (RF-DETR, RT-DETR, D-FINE) have no published football-ball evidence either way; the one small-object comparison found does not show an advantage over YOLO11 on small objects. They should be treated as an experiment, not a recommendation.

### Gaps
- No football-ball recall/precision comparison of SAHI vs. InferenceSlicer vs. P2-head vs. RF-DETR/RT-DETR/D-FINE/YOLO11/12/26 was found. I found no reliable source on this.
- Licences not verified this session (from memory, confirm before use): Ultralytics YOLO code and weights AGPL-3.0 (commercial use needs an enterprise licence); obss/sahi MIT; roboflow/supervision MIT; RF-DETR Apache-2.0 for base/small sizes; D-FINE and RT-DETR Apache-2.0.
- Size and licence of the Roboflow `football-ball-detection-rejhg` dataset were not stated on the pages read (from memory it is CC BY 4.0 with a few thousand tiled images; unverified). The underlying frames come from the DFL Bundesliga Kaggle competition, whose competition data rules may restrict commercial reuse; this needs checking.
- mAP 0.925 is on Roboflow's own validation split of the same Bundesliga broadcast domain; no out-of-domain number exists.
- No FPS figures for sliced YOLOv8x on an L4 were found.

## 3. Trajectory post-processing (filtering, outlier rejection, gap interpolation, graph/Viterbi selection, player-proximity priors, occluded/airborne state)

### Takeaway
Published football pipelines converge on the same pattern: over-generate candidates at low confidence, select one physically consistent path, then smooth and interpolate short gaps. The strongest classical formulation is a candidate graph solved with Viterbi/DP; modern open-source code (Roboflow BallTracker, WASB tracker) uses much weaker greedy heuristics, so this is where an in-house implementation can add the most without new models.

### Cited Findings
- Roboflow `BallTracker`: keeps a deque of the last 10 ball centres, computes their mean, and keeps the detection nearest that centroid; returns empty when nothing is detected; assumes one ball. The post explicitly lists as not implemented: a maximum distance gate, position prediction when the ball is missed, a recency-weighted centroid, and interpolation of missed frames — [Roboflow blog, ball tracking](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- WASB tracker: predicts next position with p_hat(t+1) = p_t + v_t + a_t/2 from the last three positions, discards candidates beyond a distance threshold, picks highest confidence; no Kalman/particle filter; reported not to help on the soccer set — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- SoccerNet 2023 tracking 3rd place: ball from fine-tuned YOLOv8l plus proprietary filtering for precision — [MOT4MOT report](https://arxiv.org/abs/2308.16651); a search summary of the same report describes low confidence threshold, trajectory smoothing and linear interpolation of gaps, but the abstract page I read attributes interpolation to the player tracker, so the ball-interpolation detail is unconfirmed — [MOT4MOT PDF](https://arxiv.org/pdf/2308.16651)
- Candidate-graph selection: classic broadcast-soccer methods build a weighted graph with a node per ball candidate and edges between candidates in adjacent frames, then extract the optimal path with Viterbi — [Trajectory-Based Ball Detection and Tracking in Broadcast Soccer Video](https://www.researchgate.net/publication/3424486_Trajectory-Based_Ball_Detection_and_Tracking_in_Broadcast_Soccer_Video); [A Scheme for Ball Detection and Tracking in Broadcast Soccer Video](https://www.researchgate.net/publication/220763925_A_Scheme_for_Ball_Detection_and_Tracking_in_Broadcast_Soccer_Video); MAP-Viterbi data association for occlusion and fast motion in team sports — [MAP Viterbi data association](https://www.researchgate.net/publication/220932540_A_Maximum_A_Posteriori_Probability_Viterbi_Data_Association_Algorithm_for_Ball_Tracking_in_Sports_Video)
- Interpolation is the standard way to bridge short occlusions or brief out-of-frame intervals in trajectory-based methods; multi-camera work estimates the ball during player occlusion by linear interpolation between positions just before and after — [Tracking the Soccer Ball using Multiple Fixed Cameras](https://strathprints.strath.ac.uk/29269/1/cviu_text.pdf)
- TrackNetV3's learned alternative: a trajectory "inpainting" network (sequence length 16) that takes the predicted trajectory plus a repair mask and fills occluded stretches — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- SoccerEDMF (Visual Computer, Aug 2025) uses a dual-model player/ball framework with interpolation to predict ball positions during occlusions — [Springer, SoccerEDMF](https://link.springer.com/article/10.1007/s00371-025-04118-x)
- A 2025 analysis is dedicated to the weaknesses of Kalman-filter trackers on fast-moving tiny objects — [arXiv 2509.18451](https://arxiv.org/pdf/2509.18451)

### Inferences
(Design recommendations; these are engineering judgement built on the findings above, not benchmarked claims.)
- Replace greedy per-frame selection with offline global selection. Since processing is offline (whole halves on Modal), keep top-K candidates per frame from every source (main detector raw path, tile detector, heatmap peaks), with scores, and solve a shortest-path/Viterbi problem over a sliding window (e.g. 2-5 s). Node cost = -log(confidence) (source-calibrated); edge cost = motion inconsistency (speed gate in pitch metres after homography where available, otherwise pixels scaled by local player-box height); include a "missing" state per frame with a fixed skip penalty so the path may bridge gaps rather than being forced through a false positive. This directly targets the two observed symptoms: low coverage (low thresholds become safe because the path rejects clutter) and wrong candidates (heads, penalty spots, second balls).
- Static-candidate suppression before the graph: candidates that stay at the same pitch coordinate for many seconds while play continues elsewhere (penalty spot, centre spot, spare balls behind the goal, logos) should be down-weighted; with a moving broadcast camera this must be done in pitch coordinates, not pixels.
- Player-proximity prior: when the path terminates within about one player-box height of a player and reappears near the same or another player, classify the gap as "possessed/occluded" and represent the ball as attached to that player (position = feet of the nearest player track) rather than linearly interpolated through space.
- Interpolation policy and flagging: interpolate only when both endpoints are confident detections. Suggested tiers: gaps up to ~0.3-0.5 s (8-12 frames at 25 FPS) with consistent endpoint velocities -> linear (ground) interpolation, flag `interpolated`; gaps up to ~1-1.5 s in open space -> constant-velocity or quadratic (flight) fit on the surrounding detections, flag `interpolated_long` with lower confidence; gap bounded by the same player -> `occluded_by_player` carried with the player; anything longer or crossing a shot change / replay cut -> leave as missing. Store per-frame `source` (detected / heatmap / tile / interpolated / carried), `confidence` and `gap_length` so possession and pass logic can ignore or down-weight synthetic points; never compute ball speed or event timing from interpolated samples alone.
- Smoothing: run a forward-backward (RTS) smoother rather than a causal Kalman filter because the job is offline; use a constant-velocity model in image space with high process noise and reset at detected kicks (acceleration spikes), since a single low-noise Kalman model lags badly at passes and deflections, which is the failure mode the tiny-object Kalman analysis is about.
- Airborne state cannot be recovered reliably in 2D from one camera; a practical proxy flag is "image trajectory is well fit by a parabola-like curve and departs from the straight ground line between kick and reception, and ball is not near any player's feet". Mark those frames `airborne_likely` and exclude them from foot-proximity possession tests.
- Shot-boundary handling is a prerequisite on broadcast footage: reset the trajectory graph at cuts and drop replays/close-ups, otherwise interpolation will bridge unrelated shots.

### Gaps
- No 2024-2026 paper was found that benchmarks graph/Viterbi selection, Kalman/RTS smoothing or player-proximity priors for the football ball with quantitative ablations; the citable evidence is 2005-2010 classical work plus undocumented "proprietary filtering" in a challenge report.
- No published guidance on a maximum legitimate interpolation gap for football was found; the thresholds above are my suggestions and need tuning against labelled clips.
- I could not confirm from the primary text whether MOT4MOT interpolates the ball.

## 4. Datasets to fine-tune on (SoccerNet, SoccerTrack v1/v2, ISSIA-CNR, Roboflow Universe, SoccerSynth, others): sizes and licences

### Takeaway
Ball-annotated football data exists for both target domains: SoccerNet-Tracking for broadcast (ball tracklets in 200 x 30 s clips), ISSIA-CNR and SoccerTrack for fixed wide cameras, the Roboflow ball set for tiled Bundesliga broadcast frames, and SoccerSynth-Detection for synthetic pre-training. Licences are the weak point: most are unclear or research-oriented, and several could not be confirmed.

### Cited Findings
- SoccerNet-Tracking: 200 sequences of 30 s each plus one complete 45-minute half, annotated with tracklets for players, referees and balls; part of the annotations are withheld for challenges — [SoccerNet-Tracking paper](https://arxiv.org/pdf/2204.06918). A secondary benchmark page reports 201 sequences from 12 professional games with 297 ball tracklets and 215,156 ball boxes (not cross-checked against the paper) — [alphaXiv benchmark page](https://alphaxiv.org/benchmarks/kaust/soccernet-tracking)
- Copyright restrictions severely limit real football dataset availability, with SoccerNet-Tracking and SportsMOT named as the limited real options — [SoccerSynth-Detection paper](https://arxiv.org/html/2501.09281)
- ISSIA-CNR: six synchronised long-shot Full-HD cameras, about 20,000 annotated frames with ball positions and player boxes; described as publicly available — [FootAndBall paper](https://arxiv.org/pdf/1912.05445); [DeepBall paper](https://arxiv.org/pdf/1902.07304). WASB's re-annotated version: 17,993 frames from one game (11,994 train / 5,999 test) — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- SoccerTrack v2 (arXiv 2508.01802, Aug 2025): 10 full-length panoramic 4K matches of university-level football (~900 minutes); 8 filmed with BePro's 3-camera stitched system, 2 with BePro Cerberus; per-frame GSR labels for players, goalkeepers and referees (pitch coordinates, jersey IDs, roles, teams) and Ball Action Spotting labels in 12 classes (Pass, Drive, Shot, ...). Per-frame ball position annotation is not described in the sources read; a curated bounding-box subset was announced for the SoccerTrack Challenge at MMSports 2025 — [SoccerTrack v2 paper](https://arxiv.org/html/2508.01802v1); [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/)
- SoccerSynth-Detection: synthetic dataset with two classes (0 player, 1 ball), YOLO-format labels, clear 1920x1080 images and motion-blurred 800x600 images, about 84 GB on Zenodo. Licence is inconsistent across sources: Zenodo lists CC BY 4.0, the arXiv text says CC BY-SA 4.0, and the GitHub repo shows Apache-2.0 (probably code only) — [Zenodo record](https://zenodo.org/records/20839314); [arXiv 2501.09281](https://arxiv.org/html/2501.09281); [open-starlab/SoccerSynth-Detection](https://github.com/open-starlab/SoccerSynth-Detection)
- Roboflow Universe football datasets linked by roboflow/sports: `football-ball-detection-rejhg` (ball, tiles from DFL Bundesliga Data Shootout frames sampled at 3 FPS), `football-players-detection-3zvbc`, `football-field-detection-f07vi`; licences not stated on the repo page — [roboflow/sports](https://github.com/roboflow/sports); [Roboflow blog](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- Soccer Player Detection dataset (FootAndBall): 2,019 images, 1280x720, no ball annotations, so not useful for the ball — [FootAndBall paper](https://arxiv.org/pdf/1912.05445)
- SoccerNet ball action spotting data is event-timestamp supervision (T-DEED trains on 100-frame clips at 796x448 from raw frames), not ball position labels — [SoccerNet 2024 challenge results](https://arxiv.org/pdf/2409.10587)

### Inferences
- Domain mapping for the two target footage types: broadcast -> SoccerNet-Tracking ball tracklets + Roboflow tiled ball set; wide tactical camera -> SoccerTrack v1/v2 and ISSIA-CNR. Because temporal models need consecutive frames, SoccerNet-Tracking (25 FPS, 30 s contiguous clips) and ISSIA are usable for WASB/TrackNet-style training, whereas the Roboflow set (3 FPS stills, tiled) is only usable for single-frame detectors.
- The most valuable data will be self-labelled: pseudo-label the user's own clips with the current ensemble plus the offline trajectory selection, keep only high-confidence path segments, hand-correct a few thousand frames, and fine-tune. No public set matches a specific amateur/tactical camera rig as well.
- Synthetic motion-blurred images in SoccerSynth-Detection are relevant to the blur failure mode, but the paper's focus is player detection; benefit for ball recall is unproven.

### Gaps
- SoccerNet data licence/NDA terms were not found in this session (from memory: SoccerNet tracking/GSR data require accepting an NDA and are for research use; unverified). Commercial use of models trained on it is a legal question to check.
- SoccerTrack v1 details (from memory: fisheye wide-view + drone top-view, ball boxes included, permissive CC-type licence) and SoccerTrack v2 licence were not verified.
- Whether SoccerNet-GSR includes ball pitch positions usable for training was not verified (from memory: GSR annotations include a ball class but the GS-HOTA metric evaluates only people).
- Footovision and other commercial-provider datasets: no public, licensed ball dataset found.
- ISSIA-CNR licence and current download availability not verified.

## 5. Point trackers and video foundation models for the ball (SAM 2 / SAM 3, CoTracker3, TAPIR)

### Takeaway
There is no quantitative evidence that SAM 2-class or point-tracking models solve the football ball; the available evidence is anecdotal and mixed, and reported speeds (about 7-8 FPS for SAM-based football player tracking) are too slow for 45-minute halves on one L4. At most they are useful as an offline labelling aid.

### Cited Findings
- A practitioner guide to sports ball tracking with SAM 2 notes SAM 2 needs an externally supplied initial ball box and "may still fail, even with accurate initial bounding boxes, especially when dealing with blurry or indistinct objects"; it shows irregular ball tracks in failure examples, without quantitative results — [Sieve blog: ball tracking with SAM 2](https://alpha.sievedata.com/blog/ball-tracking-with-sam2)
- A SAM-based football player tracking paper (Dec 2025) reports 7.6-7.7 FPS and ~1.9 GB memory, 100% success in light occlusion and 90% in crowded penalty-box scenes; it covers players, not the ball — [arXiv 2512.08467](https://arxiv.org/html/2512.08467v1)
- The same line of work notes SAM 2 struggles with crowded scenes with fast-moving or self-occluding objects and with large viewpoint changes due to reliance on appearance features — [arXiv 2512.08467](https://arxiv.org/html/2512.08467v1)

### Inferences
- A football at 6-12 px with motion blur is close to the worst case for appearance-memory trackers (few pixels, texture-less, many look-alike white blobs) and for point trackers (which track surface texture). Combined with the absence of any benchmark, they should not be on the critical path.
- A defensible use: semi-automatic annotation. Seed SAM 2 with a confident detection, propagate forward/backward for a few seconds at native-resolution crops, and accept propagated positions only when they agree with detector candidates; use the result to grow a fine-tuning set.

### Gaps
- I found no reliable sources evaluating SAM 2, SAM 3, CoTracker3 or TAPIR on football ball tracking with metrics. No search result covered CoTracker3/TAPIR for balls at all.
- SAM 3 (concept-prompted video segmentation) was not found in any football-ball evaluation.

## 6. What the SoccerNet GSR / ball action spotting community and open full pipelines do for the ball

### Takeaway
The benchmark community largely sidesteps the ball: Game State Reconstruction entries optimise people-centric GS-HOTA with no ball component described, and ball action spotting winners are end-to-end video models on raw frames that never localise the ball. Open pipelines that do handle the ball use a fine-tuned YOLO with low thresholds and light heuristics, so there is no hidden state-of-the-art ball tracker to borrow from this community.

### Cited Findings
- SoccerNet GSR 2025 winner (KIST-GSR, GS-HOTA 36.90%; 14 teams, 66 submissions): modular pipeline of camera calibration, player tracking (Deep-EIoU with OSNet ReID), role/jersey recognition and tracklet refinement; no ball module described — [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182); [yinmayoo185/SoccernetGSR](https://github.com/yinmayoo185/SoccernetGSR)
- Ball action spotting 2024 winner T-DEED: 2D backbone with Gate-Shift-Fuse modules on raw 100-frame clips at 796x448, multitask-trained on SoccerNet Ball and SoccerNet action spotting; no ball detection/tracking input — [SoccerNet 2024 Challenges Results](https://arxiv.org/pdf/2409.10587)
- Team ball action spotting 2025 winner re-implemented T-DEED with a joint team-action head, pre-trained on 500+ broadcast matches then fine-tuned; most top entries started from the T-DEED baseline — [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182)
- SoccerNet-Tracking's own framing: tracking players, referees and ball in broadcast video "is far from being solved", especially under severe occlusion — [SoccerNet-Tracking paper](https://arxiv.org/pdf/2204.06918)
- MOT4MOT (SoccerNet tracking 2023, 3rd): ball = single-object detection with fine-tuned YOLOv8l + proprietary precision filter — [MOT4MOT report](https://arxiv.org/abs/2308.16651)
- roboflow/sports: dedicated tiled ball detector + InferenceSlicer + centroid-buffer BallTracker; ball tracking listed as an open challenge — [roboflow/sports](https://github.com/roboflow/sports); [Roboflow blog](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- A 2026 "AI Driven Soccer Analysis Using Computer Vision" paper appeared in search results but was not read — [alphaXiv 2604.08722](https://www.alphaxiv.org/abs/2604.08722.md)

### Inferences
- For event analytics there are two viable architectures: (a) trajectory-first (robust ball track -> rule/ML events), which the current pipeline assumes; (b) event-first (T-DEED-style spotting of pass/drive/shot directly from frames, with the ball track used only for attribution and location). Because spotting models do not depend on per-frame ball visibility, (b) is a hedge against residual ball gaps and its outputs (pass and reception timestamps) can also serve as anchors/reset points for the trajectory smoother.
- Since nobody in the open community publishes a strong football ball tracker, the realistic best-practice stack has to be assembled: dedicated tile-trained ball detector + temporal heatmap model on ROI crops + offline graph selection + flagged interpolation.

### Gaps
- TrackLab / sn-gamestate baseline handling of the ball was not verified this session (from memory: the baseline's YOLO detector has a ball class but tracking, calibration and evaluation concern only persons). T-DEED licence and L4 throughput not checked.
- No ball-specific metric (ball recall, ball HOTA) is published in the challenge reports read.

## 7. 3D ball localisation from monocular video (brief)

### Takeaway
Recent monocular 3D ball work exists but is not practically reusable here: the main 2025 paper relies on a proprietary 6K dataset with no released code, and all such methods presuppose an accurate 2D ball track and camera calibration, which is the current bottleneck.

### Cited Findings
- "Real-time Localization of a Soccer Ball from a Single Camera" (Vorobev, Prosvetov, Daou; arXiv 2506.07981, June 2025): reconstructs the 3D ball trajectory from one broadcast camera using a multi-mode state model with W discrete modes; claims centimetre-level accuracy comparable to multi-camera systems and CPU real-time operation; evaluated on a proprietary 6K Russian Premier League dataset; no code release stated; numbers in the abstract are qualitative — [arXiv 2506.07981](https://arxiv.org/abs/2506.07981). A search summary states its 2D detections come from RTMDet-t (not confirmed on the abstract page) — [arXiv PDF](https://arxiv.org/pdf/2506.07981)
- SoccerNet-v3D (arXiv 2504.10106, 2025) uses broadcast replays for 3D scene understanding (title-level evidence only; not read) — [SoccerNet-v3D](https://arxiv.org/pdf/2504.10106)

### Inferences
- The useful idea to borrow is the multi-mode state (e.g. rolling / flying / possessed) for the 2D smoother; fitting ballistic arcs in pitch coordinates between detected kick and landing points can give approximate height for long balls once calibration and 2D tracking are reliable. Full 3D is not worth pursuing before 2D coverage is fixed.

### Gaps
- Accuracy claims in 2506.07981 are author claims on private data and could not be verified; no open implementation was found.

## Cross-cutting integration advice for the existing pipeline (one L4 on Modal, 45-minute halves)

### Takeaway
The evidence supports a staged upgrade: first fix selection and gap handling offline (no new training), then add a dedicated tile-trained ball detector, then add a temporal heatmap model (WASB architecture) on ROI crops fine-tuned on in-domain data; measure each step on a small hand-labelled ball set using recall/precision at a pixel threshold plus coverage.

### Cited Findings
- Temporal 3-frame heatmap models roughly double F1 over single-frame heatmap models on soccer under identical training (85-88 vs ~45) — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- A dedicated ball detector trained and run on tiles reaches mAP@50 0.925 in-domain yet still shows frequent false negatives, and its reference tracker lacks gating, prediction and interpolation — [Roboflow blog](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- WASB runs 55.7 FPS (step=3) / 23.6 FPS (step=1) at 288x512 on a V100 with 1.5M parameters — [WASB paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237)
- Occlusion-aware 5-frame models (TOTNet) greatly outperform WASB/TrackNetV2 on fully occluded frames in racket sports at about 28 FPS — [TOTNet paper](https://arxiv.org/html/2508.09650v1)

### Inferences
Suggested order of work (all estimates; none benchmarked on the user's footage):
1. Build an evaluation set first: 10-20 clips x 30-60 s from each camera type, ball centre labelled every frame with a visibility flag (visible / occluded / out of frame). Report recall and precision at a distance threshold scaled to ball size (e.g. 1x ball diameter, and WASB-style fixed px), coverage of visible frames, and number of identity jumps (track on wrong object). Current "56-75% of frames have a candidate" is not recall on visible frames; the true ceiling is unknown until out-of-frame/occluded frames are excluded.
2. Offline candidate-graph selection + RTS smoothing + flagged interpolation (section 3). Zero training cost; lets the raw-ball threshold drop below 0.1 safely; expected to convert existing candidates into usable trajectories and to fill sub-second gaps.
3. Dedicated single-class ball detector fine-tuned on tiles (Roboflow ball set + SoccerNet-Tracking ball boxes + own pseudo-labels), replacing the COCO sports-ball rescue model. Run ROI-first (one or two tiles around the predicted position) with a global tile sweep on track loss or every N frames to bound cost.
4. Temporal heatmap model: WASB architecture (MIT), initialised from the soccer checkpoint, fine-tuned on SoccerNet-Tracking / ISSIA / own labels, run at native resolution on a crop around the predicted ball and fused into the candidate graph as another candidate source. Use step=3 batching for speed; consider 5-frame input and occlusion-focused training (TOTNet-style) only if step 1's metrics show occlusion, not detection, is the dominant loss.
5. Optional hedge: T-DEED-style ball action spotting for pass/shot timestamps so event analytics degrade gracefully where the ball track is missing.
- Budget sanity check for one half (67,500 frames at 25 FPS): main 1280 detector pass (already paid) + ~1-2 ROI tiles per frame for the ball detector + one 288x512x3-frame heatmap pass per 3 frames is a modest increment; full 4-6 tile sweeps on every frame would multiply detector cost 4-6x and should be reserved for reacquisition.
- Licensing summary for the recommended stack: WASB-SBDT code MIT (weights trained on ISSIA-derived data of unclear licence); TrackNetV3 MIT; roboflow/sports MIT; Ultralytics YOLO is AGPL-3.0 (from memory, unverified this session), which matters if the product is closed-source; training-data licences (SoccerNet NDA, DFL Kaggle-derived Roboflow set) are the main commercial risk and need legal confirmation.

### Gaps
- No measured numbers exist for any of these steps on the user's footage or on an L4; the ordering reflects cost and strength of evidence, not a verified ranking.
- No public, football-specific benchmark compares single-frame sliced detection against temporal heatmap tracking on broadcast footage; this comparison would have to be run in-house on the evaluation set from step 1.
