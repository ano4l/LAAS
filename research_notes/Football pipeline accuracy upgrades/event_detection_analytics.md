# Automatic football event detection and analytics from video and tracking data (state as of Nov 2026)

Reading guide: "Verified" = read on a primary page (paper HTML or repo README) in this session. "Search-summary" = taken from a search-engine summary of a primary source, not opened directly; re-check before quoting. Anything I could not source this session is in Gaps, marked "prior knowledge, unverified".

---

## 1. SoccerNet Action Spotting / Ball Action Spotting 2023-2026: top methods, code, weights, mAP, compute, generalisation

### Takeaway
The practical open-source state of the art for on-ball event spotting from video is the T-DEED family (RegNetY + temporal encoder-decoder, GPL-3.0, checkpoints published), which won 2024 and was the baseline or backbone for nearly every top 2025 entry; on the 12-class task the best published scores are only ~60-64 mAP at a 1-second tolerance on professional broadcast video, so it is a proposal generator, not a hands-off event feed. The 2026 edition moved to player-attributed spotting (FOOTPASS), where the winner reaches 58.9 macro-F1, and the best methods fuse video with tracking/game-state, which is the direction this pipeline should take. Transfer to amateur single-camera footage is documented as poor without fine-tuning.

### Cited Findings

**2023 (2 classes: pass, drive)**
- Verified: lRomul/ball-action-spotting (1st place 2023) is MIT licensed, shares trained weights and predictions on Google Drive, and reports mAP@1 of 86.47% on the challenge set and 87.03% on test (transfer-learning stage: 86.35% challenge / 86.51% test; basic training 84.26% test). — [lRomul/ball-action-spotting](https://github.com/lRomul/ball-action-spotting)
- Verified: architecture is slow-fusion: EfficientNetV2-B0 2D encoder, four inverted-residual 3D blocks, GeM pooling, linear classifier for the middle frame; input is grayscale 1280x736; 15 frames sampled at ~12.5 fps stacked in threes (33 frames in fine-tuning, ~2.6 s window); horizontal-flip TTA. — [lRomul/ball-action-spotting](https://github.com/lRomul/ball-action-spotting)
- Verified: compute: tuned for an RTX 3090; one basic-training epoch (train+val) takes ~7 min 10 s on one RTX 3090 Ti; no absolute inference fps stated. — [lRomul/ball-action-spotting](https://github.com/lRomul/ball-action-spotting)

**2024 (12 classes)**
- Search-summary: T-DEED won the 2024 Ball Action Spotting challenge with challenge-set mAP@1 of 73.39 (78.51 at mAP@5), trained jointly on SoccerNet Ball and the original SoccerNet action spotting data with two prediction heads. — [Deep learning for action spotting in association football videos (survey)](https://arxiv.org/pdf/2410.01304); [SoccerNet 2024 Challenges Results](https://arxiv.org/pdf/2409.10587)
- Verified: arturxe2/T-DEED is GPL-3.0 (with an additional "LICENSE E2E-Spot" file, as it derives from E2E-Spot); checkpoints are on a linked Google Drive; supported datasets include SoccerNet (action spotting) and SoccerNetBall; an `inference.py` runs on an arbitrary video, e.g. `--model SoccerNetBall_challenge1 --frame_width 796 --frame_height 448 --inference_threshold 0.3`; frame size must match training resolution. — [arturxe2/T-DEED](https://github.com/arturxe2/T-DEED)
- Verified: T-DEED uses RegNetY backbones (200MF and 800MF variants, hidden dims 368/768) and 100-frame clips (ablation-best). The CVsports paper itself evaluates only FigureSkating and FineDiving; SoccerNet numbers are in the extended paper / challenge report. — [T-DEED paper](https://arxiv.org/html/2404.05392v2)

**2025 (Team Ball Action Spotting: 12 classes + which team)**
- Verified: dataset is 77 annotated English Football League games for training plus 22 held-out challenge games; classes are Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Verified: metric Team-mAP@1: AP per action-team pair, class AP = count-weighted mean over teams, tolerance delta = 1 second. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Verified leaderboard (Team-mAP@1 / mAP@1): 1) dudek 60.03 / 61.56; 2) Intellindust-AI-Lab 56.78 / 63.39; 3) KIST-BAS 56.08 / 58.09; 4) kistimrcteam 55.98 / 58.22; 5) FSI-Tahakom 55.51 / 63.89. Baseline (T-DEED based): 51.72 / 58.38. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Verified: winner (Kamil Czarnogorski) re-implemented T-DEED in the open-source "dude.k" framework with a single head over joint team-action classes; augmentations = mirroring with left/right label swap, random crops, brightness jitter, synthetic camera pan; pre-trained on 500+ broadcast matches then fine-tuned; Bayesian hyper-parameter search. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Verified: 2nd place merged SoccerNet action labels into the BAS label set (single head), trained on an 85% SoccerNet / 15% BAS mix, used flip TTA, class-specific ensembles for Player Successful Tackle, and a "team alignment" post-process using the previous 4 s of video to fix team labels on Shot/Goal. 5th place used two frozen T-DEED backbones on the same 100-frame clip with new heads. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Verified: the 2025 report gives no per-class AP, no input resolution and no GPU/compute figures for this task; the only code pointer is the SoccerNet GitHub org (baseline dev kit sn-teamspotting). — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1); [SoccerNet/sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting)

**2026 (Player-Centric Ball Action Spotting, and Ball Action Anticipation)**
- Verified: SoccerNet 2026 had five tasks: Ball Action Anticipation, Player-Centric Ball Action Spotting, Novel View Synthesis, Spiideo SoccerNet Synloc (athlete localisation in pitch coordinates), Visual Question Answering; 427 teams, 1,129 entries. There was no plain (team) ball action spotting task. — [SoccerNet 2026 Challenges Results](https://arxiv.org/abs/2607.07320)
- Verified: PCBAS uses the FOOTPASS dataset; 8 classes (Drive, Pass, Cross, Shot, Header, Throw-in, Tackle, Block); a true positive needs correct class, team AND jersey number within +/-12 frames; metric Macro F1@0.15 (confidence threshold 0.15). — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320)
- Verified PCBAS leaderboard: 1) FSITAHAKOM "PAVE" 58.94; 2) AISATSANZ 56.40; 3) TeamKIST 55.69; baseline (TAAD + DST) 46.41. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320)
- Verified: PAVE extends TAAD-DST: a temporal transformer over ROI-aligned X3D per-player features (TAAD) plus a game-state model (DST) with per-player attention over 26 role slots; four variants fused with agreement filtering to suppress single-model false positives. It consumes the provided game-state (tracking/role) information; other entries also used ball tracking. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320)
- Search-summary: FOOTPASS = 54 full men's matches (2023/24), 1920x1080 @ 25 fps, ~81 h of broadcast video, 102,992 manually validated play-by-play on-ball events; best paper baseline TAAD+GNN F1 52.1%. Annotations and baseline code are said to be on GitHub/HuggingFace, with video frames behind the SoccerNet NDA. — [FOOTPASS paper](https://arxiv.org/pdf/2511.16183)
- Search-summary: a rank-7 PCBAS entry that only retrained and post-processed the FOOTPASS baselines reached 0.446 macro-F1 on the challenge set. — [arXiv 2606.09679](https://arxiv.org/abs/2606.09679)
- Verified: Ball Action Anticipation (10 classes, SN-BAA, 7 public + 2 private matches) is far from usable: winner 24.08 mAP_avg vs 16.76 baseline (FAANTRA, RegNetY-GSF backbone, 448p frames). — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320)

**Generalisation beyond broadcast**
- Search-summary: Cabado et al. (CVPRW CVsports 2024, "Beyond the Premier") benchmark SoccerNet-trained spotters on a professional, a semi-professional and an amateur league and conclude that transfer "from professional broadcasts to amateur AI-piloted camera is challenging" because of recording and editing differences; it is an analysis paper calling for domain adaptation, not a fix. — [Cabado et al., CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.pdf)
- Search-summary: OSL-ActionSpotting (`pip install oslactionspotting`) is the SoccerNet group's plug-and-play library unifying feature-based and end-to-end spotting algorithms. — [oslactionspotting on PyPI](https://pypi.org/project/oslactionspotting/0.1.2)

### Inferences
- Expect materially lower accuracy than the leaderboard on the client's footage. The 60-64 mAP@1 and ~59 F1 numbers are on professional EFL/top-league broadcast footage at 1080p/25 fps; amateur wide-angle footage has much smaller players and ball in pixels, no close-ups, and a documented domain gap. Budget for fine-tuning on a few of the client's own matches before trusting any class beyond pass/drive.
- The gap between the 2-class result (86-87 mAP@1) and the 12-class result (~60-64) says: pass and drive are detectable from video; rare/ambiguous classes (tackle, block, header, shot) pull the mean down. Per-class AP is not published in the challenge reports, so which classes are weak is an inference, consistent with the 2nd-place team needing a class-specific ensemble for tackles.
- Best fit for one L4 (24 GB): T-DEED RegNetY-200MF/800MF inference at 796x448 on 100-frame clips is a small 2D-CNN-plus-temporal model and should run faster than real time on an L4; lRomul's 1280x736 EfficientNetV2-B0 model is also light. Neither repo publishes inference fps, so this must be measured. Training/fine-tuning T-DEED on an L4 is plausible at batch sizes smaller than the authors' (their GPU is not stated).
- Licence consequence: T-DEED is GPL-3.0. Running it as a separate service on Modal and consuming its JSON output does not by itself force the rest of the product open, but distributing a combined binary would; get this confirmed if the product is ever shipped on-prem. lRomul's repo is MIT and is the safer code base, but its weights only cover pass/drive.
- The 2026 results indicate the winning recipe is video + tracking/game-state fusion (per-player ROI features + role/formation context). This pipeline already has per-frame detections and tracks, so a staged design is natural: (1) rule-based candidates from tracks, (2) a video spotter for timing/class, (3) attribute the event to the nearest tracked player at the spotted frame.
- All SoccerNet-trained weights inherit the SoccerNet data terms (see Gaps): treat pretrained SoccerNet checkpoints as a commercial-licensing risk until checked.

### Gaps
- Per-class AP (pass, drive, shot, tackle...) for T-DEED and the 2025/2026 winners: not in the pages I could read. The 2024 results PDF could not be parsed (binary PDF); the 73.39 figure is from a search summary of a survey table.
- UGLF, ASTRA, COMEDIAN, E2E-Spot: not researched this session; no verified numbers or licences. Prior knowledge, unverified: these target the 17-class SoccerNet-v2 action spotting task (goals, cards, substitutions, corners, etc. at coarse tolerances), which is less relevant to pass/shot/tackle analytics than Ball Action Spotting; E2E-Spot (jhong93/spot) is the code base T-DEED builds on and I believe is BSD-3-Clause.
- SoccerNet dataset/video licence: prior knowledge, unverified this session — SoccerNet videos are obtained under an NDA for research use; whether models trained on them may be used commercially is not something I could confirm. Must be checked on soccer-net.org before shipping SoccerNet-trained weights in a paid product.
- No source found for inference speed (fps) or VRAM of any of these models on an L4/T4-class GPU.
- "dude.k" framework repo link and licence were not given in the 2025 report. Whether the 2025/2026 winners released weights is unknown.
- The Cabado paper PDF returned HTTP 403; its exact zero-shot vs fine-tuned mAP numbers and how much target data was used are not captured here.
- No public benchmark was found for ball action spotting on amateur wide-angle single-camera (Veo/Pixellot/Spiideo-style) footage.

---

## 2. Deriving events from tracking data: rule-based and learned, accuracy, data-quality needs

### Takeaway
The reference rule-based method (Vidal-Codina et al., FIFA/MIT) reaches >90% precision and recall for passes and most set pieces from 25 Hz 2D tracking, but only 78% precision / 53% recall for shots and cannot detect tackles at all; it presumes professional-grade ball tracking (sub-metre ball position at pass time, all 22 players visible). With this pipeline's sparse ball detections the method will not hit those numbers until ball coverage and pitch-coordinate calibration are fixed.

### Cited Findings
- Verified: two-stage design: (1) per-frame possession from player-ball proximity, (2) possession changes + laws of the game give events. Evaluated on 25 Hz data from Track160, Tracab and Hawk-Eye against manually annotated events (STS). — [Vidal-Codina et al., arXiv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified: possession zone = circle around each player, radius 0.5 m (provider A) or 1 m (providers B, C). Duel zone = 1 m circle around the ball; two opponents inside = duel, which takes precedence over possession. — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified: possession loss = ball leaves the player's zone, ball displacement exceeds a threshold, and the player is not in the next control frame. Possession gain = over a control sequence, the ball's incoming vs outgoing direction differs beyond an angle threshold OR speed changes beyond a threshold on at least one frame; sequences with neither are discarded (i.e. the ball merely passed near the player). Threshold values are in the paper's online supplement. — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified: noisy ball data is smoothed with a Savitzky-Golay filter (2nd order, 7-frame window, ~0.25 s at 25 Hz). — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified: rules — pass = loss followed by reception (teammate) or interception (opponent); shot = loss by attacker followed by goal, corner, goal kick or save (on/off target from trajectory with 0.25 m post tolerance); blocked shots are recorded as passes; cross = loss from a cross zone with next controller in the box; set pieces = dead-ball interval then spatial pattern (kick-off, penalty, goal kick, corner, throw-in; free kick is the fallback). — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified accuracy: passes precision and recall both >90%; corners, throw-ins, goal kicks both >90%; shots precision 78%, recall 53% (~70% of the 254 shot errors were blocked shots labelled as passes, ~15% missed or mis-attributed due to occlusion); 569 crosses were labelled as passes; free kicks are the weakest set piece. — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified: tackles are NOT detected; duels exist only as a state. The authors say tackles would need limb tracking or ball height. Ball z is not required for pass/shot logic but would help separate throw-ins from other restarts and shots over the bar. — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified: the method needs ball tracking and notes ball accuracy at passing time is under 1 m in their data. — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Search-summary: a later touch-detection paper notes that Vidal-Codina (2022) and BallRadar (2023) need full visibility of all players, limiting robustness under occlusion / partial views. — [IRI-UPC touch events paper](https://www.iri.upc.edu/download/scidoc/3004)
- Search-summary: BallRadar (KDD 2023; pientist/ballradar) infers ball trajectory and ball possessor from player trajectories alone (Set Transformer + hierarchical Bi-LSTM), with Metrica preprocessing and training scripts. A 2026 follow-up (PathCRF) by the same author says such player-only approaches can give implausible ball paths and too-frequent possession changes. — [pientist/ballradar](https://github.com/pientist/ballradar); [arXiv 2306.08206](https://arxiv.org/abs/2306.08206); [PathCRF arXiv 2602.12080](https://www.arxiv.org/pdf/2602.12080)
- Search-summary: synchronising externally-tagged events to tracking: ELASTIC (KAIST, hyunsungkim-ds/elastic) reports 88.4% exact-frame and 94.8% within-1-second alignment vs ETSY's 21.1% / 74.7%, using only tracking-derived features (no annotated event locations) and a tutorial on the Sportec open dataset; a 2026 version (CIKM 2026) reformulates it as extended Needleman-Wunsch sequence alignment and re-annotates three Sportec matches frame by frame. — [ELASTIC arXiv 2508.09238](https://arxiv.org/html/2508.09238v1); [ELASTIC 2026 arXiv 2608.30227](https://arxiv.org/pdf/2608.30227)
- Search-summary: databallpy synchronises event and tracking data with Needleman-Wunsch (keeps event order), "smart" sync is the default since 0.4.2, supports Opta/Metrica/Instat events and Tracab/Metrica/Inmotio tracking, added kloppy integration in 0.7.x (Nov 2025). — [databallpy on PyPI](https://pypi.org/project/databallpy); [Alek050/databallpy](https://github.com/Alek050/databallpy)
- Open tracking datasets to develop/validate rules against ground-truth events:
  - Search-summary: Sportec/IDSSE open dataset — 7 full Bundesliga 1/2 matches 2022/23, TRACAB optical tracking + events, CC BY 4.0 (commercial use allowed with attribution), loadable via kloppy. — [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
  - Search-summary: SkillCorner open data — 9 or 10 matches of broadcast-derived tracking (sources differ); README asks for credit; an aggregator lists MIT. — [SkillCorner/opendata](https://www.github.com/SkillCorner/opendata)
  - Search-summary: PFF FC (now Gradient Sports) released free 2022 World Cup tracking + event data; licence terms not found. — [PFF FC blog](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)

### Inferences
- The Vidal-Codina rules are the right template for this pipeline's "possible pass" logic, but the published accuracy is conditional on inputs this pipeline does not yet have: (a) positions in metres on the pitch (homography), (b) a ball position on nearly every in-play frame, (c) stable identities at least at team level. The 1-12% possession coverage reported for the current pipeline almost certainly stems from missing ball detections and/or pixel-space thresholds, not from the rule design. Fix ball recall and interpolate short gaps before tuning event rules.
- SkillCorner open data is the most relevant public proxy for "tracking derived from a single broadcast camera" (players off-screen are missing/extrapolated), so it is the best set for testing whether rules survive partial visibility; Sportec is the best "clean" set with CC BY 4.0 and frame-accurate re-annotations (via ELASTIC 2026).
- A defensible staged approach: rule-based detector tuned on Sportec (clean) -> stress-tested on SkillCorner (broadcast) -> run on own tracks -> human review. Because blocked shots and tackles are structural blind spots of 2D tracking rules, those classes should come from the video spotter (section 1) or the human reviewer.
- ELASTIC/ETSY/databallpy solve a different problem (aligning already-existing event feeds to tracking). They become useful here only if the client ever buys or hand-tags events with coarse timestamps; then ELASTIC-style snapping gives frame-accurate events for xG/pressure features.

### Gaps
- No verified 2024-2026 open-source *learned* pass/shot detector operating on tracking data alone with published weights was found. FOOTPASS baselines (TAAD+GNN/DST) are the closest, but they are video+tracking.
- Vidal-Codina threshold values (speed, angle, displacement) and the temporal matching tolerance are in an online supplement not read here. No official code release for that paper was found.
- Licences for ELASTIC, BallRadar, ETSY repos and for the PFF FC and Metrica sample datasets were not confirmed; SkillCorner MIT is from an aggregator only.
- kloppy does not itself detect events (prior knowledge, unverified this session): it is a loader/standardiser. floodlight and Friends-of-Tracking materials were not checked this session for event detection features.
- No published accuracy for rule-based event detection on amateur single-camera tracking was found.

---

## 3. Possession models: definitions, touch detection, learned alternatives

### Takeaway
Best practice is a three-state per-frame label (controlled by player X / contested-duel / no control-ball in flight or dead) built from a ~1 m player-ball radius plus a ball-kinematics test (direction or speed change) to confirm a real touch, then aggregated into team possession sequences that persist while the ball is in flight between team-mates. Team possession should therefore cover nearly all in-play time; 1-12% coverage indicates the pipeline is only counting "ball within radius" frames.

### Cited Findings
- Verified: player possession radius 0.5-1 m depending on tracking provider; duel (contested) when at least two opponents are within 1 m of the ball, taking precedence over possession. — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified: a "gain" is only confirmed when the ball's direction changes by more than an angle threshold or its speed changes by more than a speed threshold while inside the player's zone; otherwise the proximity sequence is discarded. A "loss" requires the ball to leave the zone with displacement above a threshold. — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Verified: ball coordinates are Savitzky-Golay smoothed (order 2, 7 frames at 25 Hz) before these kinematic tests when noisy. — [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Search-summary: classical touch detection from ball acceleration spikes in the x-y plane, assigning the nearest foot within a radius, was demonstrated on 2000 Hz radio tracking (2016). — [Detection of Single Ball Contacts using a Radio-based Tracking](https://www.scitepress.org/PublishedPapers/2016/60811/pdf/index.html)
- Search-summary: a 2025 last-touch framework (Hawk-Eye skeletal + ball data, 330 duels, ball IMU at 500 Hz as ground truth) uses a touch-probability model to find the decision frame and then rules or a supervised model to pick the responsible player. — [Springer, Sports Engineering 2025](https://link.springer.com/article/10.1007/s12283-025-00518-3)
- Search-summary: learned alternative: BallRadar's Ball Possessor Classifier predicts the possessor from player trajectories alone, usable when the ball is not tracked; known failure is unrealistically frequent possession changes. — [pientist/ballradar](https://github.com/pientist/ballradar); [PathCRF](https://www.arxiv.org/pdf/2602.12080)
- Verified: in 2026 PCBAS, the strongest systems attribute each action to a player using per-player video features plus game state rather than proximity alone. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320)

### Inferences
- Recommended definition for this pipeline (synthesis, not a published standard): work in pitch metres; control candidate = nearest player within ~1-1.5 m of the ball (wider than Vidal-Codina's 0.5-1 m because single-camera homography error at the far touchline is often >1 m); confirm a touch by a kink in the smoothed ball trajectory (direction or speed change); mark "contested" when opponents are both within radius; carry *team* possession forward through ball-in-flight frames until an opponent confirms a touch or the ball goes dead; mark "unknown" explicitly when the ball is undetected for more than ~1-2 s instead of dropping the time.
- Possession percentage should be reported as team-controlled time / (team A + team B controlled time), excluding dead-ball and unknown time, and the unknown share should be shown to the user as a data-quality indicator.
- At 25-30 fps with detector jitter, acceleration thresholds tuned on 2000 Hz radio data do not transfer; direction-change on a smoothed track (Vidal-Codina style) is the more robust kink test.
- When the ball is missing, a BallRadar-style possessor classifier is a plausible gap-filler, but it needs all 22 players and its licence and accuracy on partial-view data are unverified, so it is a second-phase option.

### Gaps
- No source found giving a minimum control duration standard (e.g. N frames) or an industry definition of possession percentage from tracking; the numbers in Inferences are engineering judgement.
- BallRadar's reported possessor accuracy and position error were not retrieved.
- No public evaluation of kink-based touch detection on single-camera video-derived ball tracks was found.

---

## 4. Video-language / foundation models for football (2024-2026)

### Takeaway
Soccer-specific VLMs (MatchVision/UniSoccer, SoccerChat, SoccerAgent) classify *pre-clipped* broadcast events into coarse classes at roughly 80% top-1 at best, and general VLMs score 0.34-0.45 on SoccerNet VQA; none has demonstrated frame-accurate spotting of passes/tackles. They are usable as a second-opinion labeller or description generator on candidate clips for human review, not as the primary event detector.

### Cited Findings
- Search-summary: MatchVision (UniSoccer, arXiv 2412.01820) introduces SoccerReplay-1988 (1,988 full match videos) and reports event-classification top-1 of 82.5% (vs 55.9% for SigLIP under the same training), 84.0% with supervised pre-training on SoccerNet + SoccerReplay-1988; an ablation row on SoccerReplay-test shows 62.67 / 83.00 / 89.81 top-1/3/5. — [Towards Universal Soccer Video Understanding](https://arxiv.org/html/2412.01820v3)
- Search-summary: SoccerChat (arXiv 2505.16630) is a Qwen2-VL-7B fine-tune on SoccerNet clips with jersey-colour annotations and ASR transcripts; its action classification is reported as judge "mean scores" (6.80 on a 6-class task, 6.42 on a 16-class task), not accuracy. — [SoccerChat](https://arxiv.org/html/2505.16630v1)
- Search-summary: SoccerAgent (arXiv 2505.03735, ACM MM 2025) is a multi-agent system invoking 18 tools (17 open-source) over SoccerWiki, evaluated on SoccerBench (~10,000 multiple-choice QA pairs, 13 tasks); project page https://jyrao.github.io/SoccerAgent/. — [Multi-Agent System for Comprehensive Soccer Understanding](https://arxiv.org/pdf/2505.03735)
- Search-summary: on the official SoccerNet VQA test set, GPT-4o scores 0.39 accuracy and open VLMs 0.34-0.45 (Qwen3-VL-32B best at 0.45), per the 2026 MSUE paper. — [MSUE](https://www.alphaxiv.org/abs/2606.12106.md)
- Search-summary: on ActionAtlas (domain-specialised action recognition, not soccer-only) the best model, GPT-4o, reaches 45.52%; the authors note high frame-sampling rate matters and some Gemini video settings do not sample densely by default. — [ActionAtlas](https://arxiv.org/html/2410.05774v1)
- Search-summary: "Spot the Ball" (Nov 2025): Gemini/GPT/LLaMA/Qwen reach 17% or less at locating a hidden ball in sports images, with humans 2-3x better. — [Spot the Ball](https://arxiv.org/html/2511.00261v2)
- Search-summary: a text-only route (Llama 3.1 8B judges over SoccerNet-Echoes ASR commentary) is claimed to approach video spotters on the 17 coarse SoccerNet-v2 classes (goals, cards, substitutions) with zero video compute. — [Do We Need Large VLMs for Spotting Soccer Actions?](https://arxiv.org/html/2506.17144v2)
- Verified: SoccerNet 2026 added a dedicated Visual Question Answering task (multiple-choice questions on broadcasts). — [SoccerNet 2026 Challenges Results](https://arxiv.org/abs/2607.07320)
- Search-summary: OpenSTARLab (arXiv 2502.02785) is an open framework with a Pre-processing package (UIED unified event format, SAR format), an Event modelling package (deep event prediction), RLearn (deep RL pipeline) and an "STE Label Tool" for labelling video with event data. — [OpenSTARLab](https://arxiv.org/html/2502.02785v2)

### Inferences
- Realistic role for a VLM here: given a 4-8 s clip around a rule-based or T-DEED candidate, ask a constrained question ("pass / shot / clearance / tackle / none; which team colour?") and use agreement between VLM, spotter and rules as the confidence shown in the review UI. With ~0.4-0.45 VQA accuracy for general models and ~80% for a fine-tuned clip classifier on broadcast footage, it cannot replace review.
- Commentary-based spotting is irrelevant to amateur footage (no commentary) and to fine-grained events.
- Wide-angle amateur footage is the worst case for VLMs: the ball occupies a handful of pixels and API models downsample frames; the "Spot the Ball" and ActionAtlas results suggest fine spatial/temporal detail is exactly where they fail. Cropping around the ball using the existing tracker before sending frames is the obvious mitigation.
- Self-hosting a 7B VLM (SoccerChat-class) fits an L4 in 4-bit/8-bit or fp16 with short clips, but throughput will be low; using it only on candidate clips (hundreds per match) keeps cost bounded.

### Gaps
- No published zero-shot benchmark of Gemini/GPT/Claude on SoccerNet Ball Action Spotting (timestamped pass/shot/tackle) was found; realistic accuracy for "VLM as event tagger" is therefore unmeasured in the literature I could reach, and cost per match is not documented anywhere I found.
- Licences and weight availability for MatchVision, SoccerChat, SoccerAgent, and the OpenSTARLab repos were not confirmed (an aggregator lists CC BY-NC-SA 4.0 for the OpenSTARLab paper, which is not necessarily the code licence). SoccerNet-derived models carry the SoccerNet data terms.
- SoccerNet Dense Video Captioning and MatchTime results were not researched this session.
- SoccerAgent's accuracy numbers on SoccerBench were not retrieved.

---

## 5. Analytics libraries: xG, xT, VAEP, SPADL, loaders, plotting

### Takeaway
The PySport stack covers everything downstream once events are in a standard schema: emit events in SPADL (or kloppy's event model), then use socceraction (MIT) for xT and VAEP, mplsoccer (MIT) for pitch/heatmap/pass-network plots, kloppy (BSD-3) as the tracking/event container, and unravelsports (MPL-2.0) for pressing intensity, formations and GNN features. The main licence trap is StatsBomb open data, which is released for non-commercial use, so an xG model trained on it should not be shipped commercially without permission.

### Cited Findings
- Search-summary: socceraction is MIT licensed; provides SPADL and Atomic-SPADL converters, xT (possession-based Markov model for ball-progressing actions) and VAEP/Atomic-VAEP. The maintainers state it "is not actively developed", exists primarily for research reproducibility, and feature requests are unlikely to be accepted; 1.2.0 officially supported Python 3.7.1-3.9, latest seen on PyPI is 1.2.3. — [socceraction on PyPI](https://pypi.org/project/socceraction/1.2.3/)
- Search-summary: StatsBomb open data is under StatsBomb's own Public Data User Agreement (LICENSE.pdf, "other" licence), described in the socceraction docs as "freely available for public non-commercial use", with required attribution and logo. — [statsbomb/open-data](https://github.com/statsbomb/open-data); [socceraction docs mirror](https://huggingface.co/spaces/scfive/socr/blob/main/docs/documentation/data/statsbomb.rst)
- Search-summary: kloppy is BSD 3-Clause. — [PySport/kloppy](https://github.com/PySport/kloppy)
- Search-summary: mplsoccer is MIT (football pitch plotting for matplotlib). — [mplsoccer on PyPI](https://pypi.python.org/project/mplsoccer/0.0.7/)
- Verified: unravelsports is MPL-2.0 (file-level copyleft: modifications to its files must stay MPL; using it as a library in a closed product is allowed); features: Pressing Intensity (Bekkers 2024), EFPI formation/position identification (Bekkers 2025), GNN conversion/training/prediction (PyTorch Geometric or Spektral), kloppy -> Polars conversion; input is a kloppy tracking dataset via `KloppyPolarsDataset`; providers supported through kloppy include Sportec, SkillCorner, PFF/Gradient, Metrica, StatsPerform, Tracab, SecondSpectrum, HawkEye, Signality. Pitch control is not listed. — [UnravelSports/unravelsports](https://github.com/UnravelSports/unravelsports)
- Search-summary: databallpy loads, synchronises and analyses tracking+event data, 0.7.x (Nov 2025) with kloppy integration, JOSS publication. — [databallpy on PyPI](https://pypi.org/project/databallpy)
- Search-summary: floodlight is a high-level sports analytics framework (JOSS paper; docs include a football match-analysis compendium using open data). — [floodlight-sports/floodlight](https://github.com/floodlight-sports/floodlight)
- Search-summary: OpenSTARLab pre-processing defines UIED (Unified and Integrated Event Data) and SAR formats and an event-prediction package. — [OpenSTARLab](https://arxiv.org/html/2502.02785v2)
- Search-summary: a Spearman-style pitch control notebook exists in eddwebster/Metrica-pitch-control (FastSpearmanPitchControl.ipynb, has a LICENSE file, licence type not shown). — [eddwebster/Metrica-pitch-control](https://nbviewer.org/github/eddwebster/Metrica-pitch-control)

### Inferences
- Integration path for own events: write an exporter from the approved-events table to a SPADL DataFrame (game_id, period_id, time_seconds, team_id, player_id, start_x/y, end_x/y on a 105x68 pitch, type_name, result_name, bodypart_name). That one schema feeds socceraction xT/VAEP directly. Because socceraction is in maintenance mode and pinned to old Python versions, vendor/fork it or reimplement xT (it is a small Markov-grid computation) rather than depending on upstream.
- xT is the lowest-risk value model for this product: it needs only move actions (passes, carries/drives) with start/end locations and shot/goal counts per grid cell, and a pre-computed grid can be used at inference. VAEP needs a complete, consistent action stream (3-action context, results, body parts) and will be noisy on auto-detected events; defer it until event recall is high.
- xG: with detected shots that carry only location (plus, from tracking, goalkeeper and defender positions), train a location/angle/distance logistic or gradient-boosted model. For a commercial product, train on a commercially usable source (own approved shots over time, or licensed data) or obtain StatsBomb's permission; a model trained on StatsBomb open data is acceptable for internal prototyping only. Label xG on amateur footage as "indicative": finishing rates differ from the professional data the model was fitted on.
- Pass networks, heatmaps, shot maps: mplsoccer server-side, or export the same aggregates as JSON to the existing web frontend.
- Pressing: unravelsports Pressing Intensity works on tracking only (time-to-intercept between defenders and ball carrier/attackers), so it needs positions and velocities in metres but not events — a good early win once pitch calibration is solid. PPDA-style event-based pressing metrics need reliable pass and defensive-action events and should wait.
- Pitch control: implement Spearman's model directly from the Friends of Tracking material; run on GPU/vectorised NumPy only for key frames (events), not every frame.

### Gaps
- soccer_xg (ML-KULeuven): licence and status not found this session.
- floodlight and LaurieOnTracking licences not confirmed (prior knowledge, unverified: both MIT). accessible-space (a pass-reception / dangerous-accessible-space package) was not found by search.
- Whether training a model on StatsBomb open data and deploying it commercially is permitted was not answered by any source; treat as not permitted until StatsBomb confirms.
- statsbombpy licence not confirmed.
- No benchmark numbers (e.g. xG Brier/AUC, VAEP calibration) were collected this session.

---

## 6. Formation and tactical-pattern detection from tracking

### Takeaway
EFPI (in unravelsports, MPL-2.0) is the most practical maintained option: template matching plus linear assignment gives a formation label and a role per player per frame/possession/period from tracking alone. SoccerCPD adds change-point detection for tactical switches but its published accuracy is modest (about 72% formation / 87% position accuracy) and its licence is unconfirmed.

### Cited Findings
- Verified/search-summary: EFPI (Bekkers 2025, arXiv 2506.23843) matches scaled player positions to predefined formation templates via linear assignment and picks the lowest-cost template; outputs formation (4-4-2, 4-3-3, ...) and role (CB, CM, LW, ...) for frames, possessions, periods or time intervals; shipped in unravelsports. — [EFPI tutorial](https://unravelsports.readthedocs.io/en/stable/tutorials/formation_detection.html); [EFPI paper](https://arxiv.org/html/2506.23843v1); [unravelsports](https://github.com/UnravelSports/unravelsports)
- Search-summary: SoccerCPD (Kim et al., KDD 2022) detects formation and role change-points; best configuration reports ~72.4% formation accuracy and ~86.5% position accuracy against expert annotation vs 59.1% / 80.3% for a no-change-point baseline (table extraction ambiguous; verify). Code at github.com/pientist/soccercpd with sample GPS data from one anonymised match. — [SoccerCPD arXiv 2206.10926](https://arxiv.org/pdf/2206.10926)
- Verified: 2026 PCBAS winners encode game state as per-player attention over 26 role slots, i.e. role assignment is now a standard input to event models as well as an output for coaches. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320)

### Inferences
- Formation detection needs all ten outfield players of a team in pitch coordinates for the frames analysed. With a single panning broadcast camera this holds only part of the time; with a static wide-angle amateur camera it holds most of the time. Run EFPI only on frames where >= 9-10 outfield players of the team are tracked and aggregate per possession phase (in possession / out of possession), reporting the share of frames used.
- Separate in-possession and out-of-possession shapes (as Shaw & Glickman do) — this depends on the possession model in section 3, another reason to fix possession first.
- Template matching degrades gracefully with noisy positions; change-point detection (SoccerCPD) is more sensitive to identity switches, which this pipeline's "temporary tracks" will cause. Prefer EFPI windowed over 1-5 minute intervals first.

### Gaps
- Shaw & Glickman ("Dynamic analysis of team strategy", 2019) and Bialkowski et al. role assignment: not researched this session; no verified open implementation or licence to cite.
- EFPI accuracy figures against labelled formations were not retrieved.
- SoccerCPD licence not found. The repo owner is "pientist", not "pi-lab-korea" as in the brief.
- No evaluation of any formation method on video-derived (non-vendor) tracking was found.

---

## 7. Workload / physical metrics from optical tracking: what is defensible, error vs GPS

### Takeaway
Total distance from professional optical tracking agrees well with GPS (ICC ~0.97, optical ~4-5% higher), but high-speed running and especially sprint distance differ by 12-18% versus GPS and by up to ~60% between optical systems, so only total distance and coarse speed-zone trends are defensible, and nothing here supports injury-risk claims from single-camera video tracking.

### Cited Findings
- Search-summary: 16 official matches, 10 Hz GNSS vs 25 Hz Tracab: absolute-agreement ICC 0.974 for total distance, 0.766 for high-speed running distance, 0.822 for sprint distance. — [PMC10358465](https://pmc.ncbi.nlm.nih.gov/articles/PMC10358465)
- Search-summary: 26 Premier League players, 10 Hz GPS vs Second Spectrum optical: optical higher on all variables — total distance +4%, HSR +12%, sprint distance +18%; authors recommend regression equations to align sources. — [UCLan repository, 2025 study](https://knowledge.lancashire.ac.uk/id/eprint/56538/)
- Search-summary: two optical systems in the Premier League (Prozone vs Tracab): mean bias 5% for total distance, -3% for HSR, 61% for sprint distance. — [Biology of Sport / Cardiff Met](https://pure.cardiffmet.ac.uk/en/publications/interchangeability-of-external-player-load-variables-from-differe/)
- Search-summary: 30 Premier League players: "small to huge" differences between tracking systems for most movement variables; optical data processed through GPS software showed lower values above 19.8 km/h. — [PubMed 35236222](https://pubmed.ncbi.nlm.nih.gov/35236222/)

### Inferences
- These validations are for multi-camera, stadium-installed systems with continuous identity. This pipeline has single-camera, pixel-space, temporary tracks, so its error will be larger on every metric: players leave the frame (broadcast), identities fragment, and homography error grows with distance from camera. Per-player match totals are only meaningful when track fragments are stitched to a persistent identity and coverage (minutes observed / minutes played) is reported.
- Defensible tiers: (1) total distance per observed minute and average speed — reasonable after smoothing positions (e.g. Savitzky-Golay or Kalman on metres at 25 fps) and capping implausible speeds (>~36 km/h); (2) distance in speed zones (HSR > ~19.8 km/h, sprint > ~25 km/h) — report as estimates with a caveat, compare within the same camera setup only; (3) accelerations/decelerations counts — not defensible from this data, since differentiation twice amplifies position noise; (4) "injury indicators" — only as load trends (e.g. week-on-week change in a player's own distance/HSR), never as injury prediction.
- For broadcast footage, extrapolating per-player physical output is not credible (SkillCorner does it with proprietary off-screen modelling); restrict physical metrics to static wide-angle footage covering the whole pitch.

### Gaps
- No validation study of single-camera or broadcast-derived tracking (e.g. SkillCorner, Veo, Pixellot) against GPS was found this session; searching specifically for one returned nothing.
- No sourced error figures for acceleration/deceleration counts from optical tracking were retrieved.
- Speed-zone thresholds quoted in Inferences (19.8, 25 km/h) are common conventions; only the 19.8 km/h figure appears in a retrieved source.

---

## 8. Human-in-the-loop: annotation tools and active-learning workflow

### Takeaway
I found no football-specific open-source tagging tool that beats the pipeline's existing review UI; the evidence supports keeping that UI as the labelling front end and wiring model proposals (rules + T-DEED) into it, exporting approved events in a SoccerNet-style timestamped JSON so the same labels can fine-tune the spotter. OpenSTARLab's STE Label Tool is the only football event-labelling tool surfaced, with licence unverified.

### Cited Findings
- Search-summary: OpenSTARLab publishes an "STE Label Tool" (github.com/open-starlab/Event_Data_LabelTool) for labelling video with event data. — [OpenSTARLab](https://arxiv.org/html/2502.02785v2)
- Search-summary: the original SoccerNet labels were bootstrapped from match reports and then manually refined to one-second resolution — precedent for "weak proposal -> human refinement" annotation. — [SoccerNet (2018)](https://arxiv.org/pdf/1804.04527)
- Verified: T-DEED inference exposes a tunable `--inference_threshold` (example 0.3), i.e. the precision/recall operating point of proposals can be set for review. — [arturxe2/T-DEED](https://github.com/arturxe2/T-DEED)
- Verified: 2025's winning recipe was pre-training on 500+ broadcast matches then fine-tuning on 77 task games, and a 5th-place variant fine-tuned heads on as few as 4 team-labelled videos plus 500 action-only videos — evidence that few-match fine-tuning on top of a large pretrain is the standard approach. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Verified: the 2026 PCBAS metric discards predictions below 0.15 confidence and matches within +/-12 frames; winners suppress false positives by requiring agreement across models. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320)
- Search-summary: FOOTPASS's 102,992 events were "manually validated" play-by-play annotations on 54 matches, i.e. about 1,900 on-ball events per match — the volume a reviewer faces if every event is checked. — [FOOTPASS paper](https://arxiv.org/pdf/2511.16183)

### Inferences
- Workflow recommendation: (1) run rules + spotter at a low threshold for high recall; (2) present candidates sorted by uncertainty/disagreement (rules vs spotter vs optional VLM) with one-key accept/relabel/reject and +/- frame nudging; (3) store approved AND rejected candidates (rejections are hard negatives); (4) after 3-5 fully reviewed matches per camera type, fine-tune the spotter head on the client's footage and re-measure mAP@1 on a held-out reviewed match; (5) repeat, moving high-confidence classes (pass, drive, throw-in, out) to auto-accept and keeping shots, tackles, blocks, headers in review.
- At ~1,900 on-ball events per match, full manual review is hours per match; prioritise review of analytics-critical, low-volume events (shots, goals, set pieces, tackles) and sample-audit passes.
- Keep a fixed, reviewed "gold" set (2-3 matches per footage type: broadcast, amateur wide-angle) that is never trained on, so every pipeline change is scored with the SoccerNet metric (mAP@1 s) and with possession-coverage and pass precision/recall.
- Label Studio and CVAT support video timeline/segment labelling and model pre-annotation in general (prior knowledge, unverified this session), but adopting them would duplicate the existing UI; only worth it if multi-annotator management is needed.

### Gaps
- Searches returned no primary sources on Label Studio / CVAT configurations for football event spotting, no SoccerNet-published annotation tool, and no published active-learning study for football action spotting; their licences (prior knowledge, unverified: Label Studio Apache-2.0, CVAT MIT) were not confirmed this session.
- No data found on how many reviewed matches are needed to close the broadcast-to-amateur domain gap; the "3-5 matches" figure is a planning assumption, not a sourced number.
- Licence of the OpenSTARLab STE Label Tool not confirmed.
