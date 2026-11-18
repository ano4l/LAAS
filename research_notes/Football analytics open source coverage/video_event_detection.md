# Open-source models and code for detecting football on-ball events from match video (state as of 18 Nov 2026)

Researcher notes. Repository "last push" dates and SPDX licence IDs marked "(GitHub API)" were read from `https://api.github.com/repos/<owner>/<repo>` on 2026-11-18. Anything I could not confirm from a primary page is listed under Gaps, not under Cited Findings.

---

## Q1. Which SoccerNet action-spotting / ball-action-spotting (BAS) solutions have public code AND public weights? What are the 12 BAS classes, how do they map to the product's events, and what accuracy do the best public models reach?

### Takeaway
The only BAS models with both public code and a downloadable checkpoint for the 12-class task are T-DEED (2024 winner) and the SoccerNet 2025 team-BAS baseline built on it (both GPL-3.0, Google Drive weights); they reach 73.39 mAP@1 (2024 challenge set, ensemble) and 51.72 Team-mAP@1 / 58.38 mAP@1 (2025 challenge set). The 12 classes cover pass variants, drive, shot, goal, block and *successful* tackle only; there is no interception, duel, unsuccessful tackle, or pass-outcome class, and no per-class AP table was found in any source read.

### Cited Findings

**The 12 classes (identical in 2024 and 2025)**
- The 12 ball-action classes are: Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal. — [SoccerNet 2024 Challenges Results](https://arxiv.org/html/2409.10587v1); [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Dataset: 77 annotated English Football League games (2024 paper says broadcast at 720p), plus 22 held-out challenge games with private annotations. — [SoccerNet 2024 Challenges Results](https://arxiv.org/html/2409.10587v1); [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- In 2024 the task was expanded from 2 classes (2023) to 12. — [SoccerNet 2024 Challenges Results](https://arxiv.org/pdf/2409.10587)
- 2025 added team attribution: each action must be assigned to the left or right team (relative to camera); metric is Team-mAP@1 (AP per action-team pair, weighted by ground-truth counts, 1-second tolerance). — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)

**2024 BAS leaderboard (challenge set, mAP@1 / tight avg-mAP)**
- T-DEED 73.39 / 77.25; UniBw Munich-VIS 71.35 / 75.24; FS-TAHAKOM 67.09 / 70.64; MobiusLabs 62.53 / 66.13; AI4Sports 62.44 / 65.91; "sota" 62.44 / 66.19; SAIVA 56.74 / 60.58; Baseline 56.15 / 60.60. — [SoccerNet 2024 Challenges Results](https://arxiv.org/html/2409.10587v1)
- The winning T-DEED entry trained jointly on SoccerNet Ball and the original SoccerNet action-spotting data with two prediction heads, and ensembled two clip-sampling strategies (i.e. 73.39 is an ensemble figure). — [SoccerNet 2024 Challenges Results](https://arxiv.org/html/2409.10587v1)

**2025 Team BAS leaderboard (challenge set, Team-mAP@1 / mAP@1)**
- dudek 60.03 / 61.56; Intellindust-AI-Lab 56.78 / 63.39; KIST-BAS 56.08 / 58.09; kistimrcteam 55.98 / 58.22; FSI-Tahakom 55.51 / 63.89; liujian 55.41 / 63.21; UniBW Munich VIS 55.36 / 68.92; testken 53.90 / 60.26; NYCU AINT Lab 52.56 / 59.04; Baseline (T-DEED) 51.72 / 58.38; bupt miclab 46.21 / 52.59. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 16 teams, 67 submissions; best beat the baseline by 8.31 points. — [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182)
- Winner (Kamil Czarnogórski, "dudek") re-implemented T-DEED in the open `dude.k` framework with a single head predicting joint team-action classes, pre-trained on 500+ broadcast matches then fine-tuned; augmentations included horizontal mirroring with team-label swap and synthetic camera pans. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- 2nd place used a "team alignment" post-processing step that corrects team labels for SHOT and GOAL from preceding correlated actions, and mixed 85% SoccerNet Action Spotting / 15% BAS data. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)

**2026: the BAS task was replaced**
- SoccerNet 2026 had five tasks: Ball Action Anticipation, Player-Centric Ball Action Spotting, Novel View Synthesis, Spiideo SoccerNet Synloc, Visual Question Answering. There was no plain 12-class BAS or team-BAS task in 2026. — [SoccerNet 2026 Challenges Results](https://arxiv.org/abs/2607.07320)
- (Player-centric task details are under Q2.)

**Project inventory: code + weights status**

| Project | Repo | Output classes | Weights public? | Code licence | Last push (GitHub API) | Published accuracy |
|---|---|---|---|---|---|---|
| T-DEED (2024 BAS winner) | https://github.com/arturxe2/T-DEED | Configs for SoccerNet (action spotting) and SoccerNetBall (12-class BAS) | Yes, Google Drive folder linked from README | GPL-3.0 (plus a separate "LICENSE E2E-Spot" file) | 2026-01-07 | 73.39 mAP@1 / 77.25 tight avg-mAP, 2024 BAS challenge set (ensemble) |
| SoccerNet sn-teamspotting (2025 baseline, T-DEED + team head) | https://github.com/SoccerNet/sn-teamspotting | 12 BAS classes + left/right team | Yes, baseline checkpoint on Google Drive | GPL-3.0 | 2025-08-26 | Test split: Team-mAP@1 47.18, mAP@1 53.59. Challenge split: 51.72 / 58.38 |
| dude.k (2025 team-BAS winner) | https://github.com/int8/dude.k (also `pip install dude.k`) | Joint team-action classes (12 x 2) | Not stated anywhere I could read | No licence detected by GitHub | 2025-03-14 | 60.03 Team-mAP@1 challenge set |
| lRomul/ball-action-spotting (2023 BAS winner) | https://github.com/lRomul/ball-action-spotting | 2023 task: passes and drives only | Yes, Google Drive (weights + predictions) | MIT | 2023-07-02 | 86.47 mAP@1 challenge set, 87.03 test (2-class 2023 task) |
| E2E-Spot | https://github.com/jhong93/spot ; models at https://github.com/jhong93/e2e-spot-models | SoccerNet-v2 action spotting and SoccerNet-ball listed as supported datasets | A separate models repo exists; whether it contains SoccerNet weights not confirmed | BSD-3-Clause (models repo: none detected) | 2023-02-15 (models repo 2022-11-10) | Not on the README |
| ASTRA (3rd, 2023 Action Spotting) | https://github.com/arturxe2/ASTRA | SoccerNet-v2 action spotting (broadcast highlight classes) | Not stated on README | GPL-3.0 | 2024-05-20 | Not on README |
| OSL-ActionSpotting | https://github.com/OpenSportsLab/OSL-ActionSpotting | SoccerNet-v2 style action spotting | Model zoo lists methods; no download links/scores visible on the page read | GPL-3.0 | 2025-02-27 | Not on README |
| Yahoo spivak | https://github.com/yahoo/spivak | SoccerNet-v2 action spotting | Not checked | Apache-2.0 | 2024-09-20 | Not checked |
| COMEDIAN | https://github.com/juliendenize/eztorch | SoccerNet-v2 action spotting | Not checked | Licence not auto-detected (NOASSERTION) | 2026-01-21, repo archived | Not checked |
| SoccerNet sn-spotting (devkit, baselines) | https://github.com/SoccerNet/sn-spotting | SoccerNet-v2 action spotting; evaluation code | n/a | None detected | 2024-02-07 | n/a |

Sources for the table rows: [T-DEED repo](https://github.com/arturxe2/T-DEED); [sn-teamspotting repo](https://github.com/SoccerNet/sn-teamspotting); [dude.k author write-up](https://int8.io/team-ball-action-spotting-challenge-2025); [dude.k on PyPI](https://pypi.org/project/dude.k/0.1.0/); [lRomul repo](https://github.com/lRomul/ball-action-spotting); [E2E-Spot repo](https://github.com/jhong93/spot); [ASTRA repo](https://github.com/arturxe2/ASTRA); [OSL-ActionSpotting repo](https://github.com/OpenSportsLab/OSL-ActionSpotting); [SoccerNet 2024 results](https://arxiv.org/html/2409.10587v1); [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1); licence and push dates from the GitHub API for each repo, e.g. [T-DEED API record](https://api.github.com/repos/arturxe2/T-DEED).

**Input requirements and compute, where stated**
- T-DEED: example inference command for SoccerNetBall uses 796x448 frames; frame size must match training resolution. README says the 1st place in the 2024 SNBAS challenge was achieved with this implementation. — [T-DEED repo](https://github.com/arturxe2/T-DEED)
- T-DEED paper (non-football datasets): RegNetY 200MF/800MF backbones, 100-frame clips reported as optimal. — [T-DEED paper](https://arxiv.org/html/2404.05392v2)
- dude.k (2025 winner): 224x224 px input, 80-frame clips with 68 frames overlap, regnety_008 backbone, T-DEED with 2 layers and displacement 4, single RTX 4090, pre-training 30 epochs x 5,000 clips on ~500 historical broadcasts with the "old" 17 action labels + background, final training 20 epochs. — [int8.io write-up](https://int8.io/team-ball-action-spotting-challenge-2025)
- lRomul 2023: 1280x736 grayscale frames, every second frame of 25 fps video (effective 12.5 fps), tuned for an RTX 3090; one training epoch 7 min 10 s on a 3090 Ti. — [lRomul repo](https://github.com/lRomul/ball-action-spotting)
- E2E-Spot: frames pre-extracted and resized to about 224 px high. — [E2E-Spot repo](https://github.com/jhong93/spot)
- ASTRA does not run on raw video: it needs precomputed Baidu soccer embeddings and log-mel audio spectrograms per half. — [ASTRA repo](https://github.com/arturxe2/ASTRA)

**Data and weight terms**
- SoccerNet video datasets with protected content are gated on Hugging Face; access requires accepting an NDA, with a personalised NDA copy emailed to the account. — [soccer-net.org/data](https://www.soccer-net.org/data)
- The SN-BAS-2025 dataset card lists licence "other", manual approval, and points to the SoccerNet NDA; 19.3 GB of videos. — [HF SoccerNet/SN-BAS-2025](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025)
- A third-party dataset built on SoccerNet video states the clips "remain subject to the SoccerNet NDA", with redistribution or commercial use not permitted and use limited to research/education. This is a derived dataset's reading of the NDA, not the NDA text itself. — [SimulaMet/SoccerChat card](https://huggingface.co/datasets/SimulaMet/SoccerChat)

### Inferences
- Mapping of the 12 classes to the product's event list (my mapping, not from a source):
  - Pass count: Pass + High Pass + Cross (+ Throw In, Free Kick if set-piece restarts are counted as passes; Header is ambiguous, it can be a pass, a clearance or a shot).
  - Shot: Shot. Goal: Goal.
  - Tackle: Player Successful Tackle only. Unsuccessful tackles/fouls are not labelled, so "tackle outcome" cannot come from this taxonomy.
  - Dribble/carry: Drive is a ball carry, not a take-on; it gives no "dribble success".
  - Interception: no class. Closest signal is a team change between two consecutive events (e.g. Pass by left team followed by Drive/Pass by right team) using the 2025 team head.
  - Duel: no class. Ball Player Block is a block of a shot/pass, not a duel.
  - Possession/turnovers: derivable from the per-event team label sequence (2025 baseline) plus Out events; not a model output in itself.
  - Pass completion: derivable only indirectly (next event by same team vs other team vs Out).
- For this prototype the practical "code + weights today" option is the sn-teamspotting T-DEED checkpoint (12 classes + left/right team). Both it and T-DEED are GPL-3.0 and trained on NDA-gated SoccerNet video, so commercial use of the released weights is at best unclear; retraining on one's own labelled footage with the GPL code is a cleaner route but GPL obligations still apply to distributed code.
- The 2025 winner's dude.k repo has no licence file detected, which by default means no usage rights are granted; treat it as read-only reference.
- Because the 2024/2025 scores are ensembles or tuned challenge entries, the single released checkpoint should be expected nearer the baseline numbers (53-58 mAP@1) than the winning ones.

### Gaps
- No per-class AP table for the 12 BAS classes was found in the 2024 or 2025 results papers or repos I read. The SoccerNet evaluation code can emit per-class AP, so the numbers must be generated by running the released checkpoint. The T-DEED extended journal paper may contain a per-class table; I did not retrieve it.
- Could not confirm whether dude.k weights, Intellindust, FSI-Tahakom, KIST or UniBW Munich 2025 solutions have public code or weights.
- Baidu (2021/2022 action-spotting) and other pre-2023 challenge solutions: not re-checked in this session. My recollection is that Baidu released precomputed features rather than an end-to-end event model, but I have no source read today.
- UGLF: nothing found in this session; unverified whether code/weights exist.
- Yahoo spivak, COMEDIAN/eztorch, E2E-Spot model repo: weight availability and SoccerNet scores not read from their pages. Note these target SoccerNet-v2's 17 broadcast "highlight" classes (goals, cards, corners, substitutions etc.), which is the "old 17 classes" set referenced by the dude.k author; they do not output passes, tackles or dribbles.
- Exact NDA wording on commercial use and on the status of weights trained on NDA video was not obtained (the NDA is only issued per applicant).
- Stated GPU inference speed (fps on L4/A10G) was not found for any of these models.

---

## Q2. Is there any open model that attributes an event to a specific player or team?

### Takeaway
Yes, as of 2026: FOOTPASS / SoccerNet Player-Centric Ball Action Spotting (PCBAS) outputs (frame, class, team, jersey number) for 8 classes with public baseline code, and the 2025 SoccerNet team-BAS baseline outputs left/right team with a public checkpoint. However the FOOTPASS baselines depend on high-quality tracking, jersey and role inputs, are released under non-commercial terms, and best published scores are about 59 macro-F1 with Tackle near zero.

### Cited Findings

**FOOTPASS dataset (Footovision Play-by-Play Action Spotting in Soccer)**
- Posted to arXiv 20 Nov 2025 (2511.16183); described as the first benchmark for play-by-play action spotting over entire matches in a multi-modal, multi-agent tactical context. — [FOOTPASS abstract](https://arxiv.org/abs/2511.16183v1)
- 54 full matches from 2023/24 (Ligue 1, Bundesliga, Serie A, La Liga, UEFA Champions League), 102,992 events; 1920x1080, 25 fps broadcast video. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- 8 classes with shares: Pass 49.9%, Drive 39.0%, Header 4.0%, Cross 2.3%, Throw-in 1.9%, Block 1.4%, Shot 1.2%, Tackle 0.3%. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- Each event is a tuple (frame, class, team [0 left / 1 right], jersey number), anchored to a single ball-touch frame; annotated manually by a professional team. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- Provided inputs besides video: player positions and velocities, team membership, jersey numbers, 13 roles, per-frame homography; produced by Footovision's tracking pipeline (player detector trained on a private dataset; described as "FIFA EPTS certified"). About 81.5% of events have an acting-player bounding box. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- No outcome (success/failure) labels are described; a cross is labelled "regardless of whether the pass is completed". — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- Benchmark videos are downsampled to 352x640; split 48 train / 3 val / 3 test matches. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)

**FOOTPASS baseline results (test set, precision / recall / F1 x100, confidence 0.15, tolerance 12 frames)**
- TAAD+DST (best): Pass 72.6/72.1/72.3; Drive 68.2/68.9/68.5; Header 30.7/26.3/28.3; Cross 62.8/51.8/56.8; Throw-in 65.9/58.6/62.0; Block 23.3/14.7/18.0; Shot 63.4/67.2/65.2; Tackle 0.0/0.0/0.0; Overall 68.2/66.8/67.5. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- TAAD (visual only): Pass 43.8/61.0/51.0; Drive 23.9/60.0/34.2; Shot 12.3/74.6/21.1; Tackle 1.5/16.7/2.8; Overall 25.6/59.9/35.9. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- TAAD+GNN: Overall 44.5/62.7/52.1; mean AP 32.8% (TAAD+DST mean AP 36.8%). — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- TAAD+DST trained without multi-agent/tactical context (ablation): Overall 45.0/54.0/49.1. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- When the acting player has no bounding box, recall falls to 3.9% (TAAD), 2.6% (TAAD+GNN), 33.4% (TAAD+DST). — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- Training compute: TAAD uses an X3D-L backbone, 50-frame clips, batch 6, 25 epochs on one RTX A6000; DST uses 750-frame sequences, batch 96, 10 epochs on one RTX A6000. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)

**FOOTPASS code and terms**
- Repo https://github.com/JeremieOchin/FOOTPASS contains training scripts for the TAAD baseline, GNN and DST, evaluation code, and a `models/` folder (contents not described on the page). Dataset hosted at https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026. — [FOOTPASS repo](https://github.com/JeremieOchin/FOOTPASS)
- Licence conflict: GitHub sidebar shows Apache-2.0, but the README states "Dataset annotations & baselines released under CC BY-NC 4.0"; videos are only available via the SoccerNet NDA and may not be redistributed. — [FOOTPASS repo](https://github.com/JeremieOchin/FOOTPASS)
- Last push 2026-06-17, not archived. — [GitHub API](https://api.github.com/repos/JeremieOchin/FOOTPASS)

**SoccerNet 2026 Player-Centric Ball Action Spotting (built on FOOTPASS)**
- Task: localise and classify 8 classes (Drive, Pass, Cross, Shot, Header, Throw-in, Tackle, Block) and assign each to a player via team side and jersey number. Metric Macro F1@0.15: true positive only if within +/-12 frames and class, team and jersey all match. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320v1)
- Leaderboard (teams with reviewed reports; 6 teams, 124 submissions): FSI-TAHAKOM 58.94; AISATSANZ 56.40; TeamKIST 55.69; UniBW Munich VIS 50.35; Baseline (TAAD+DST) 46.41; WRF320 46.06; Sarthi-GameChanger 44.63. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320v1)
- Tackle was the hardest class for all participants. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320v1)
- Winner "PAVE" adds a temporal transformer over ROI-aligned X3D player features and per-player attention over 26 role slots, and fuses four independently trained variants with agreement filtering. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320v1)
- An individual participant's report gives System-Test Macro F1 0.548 for an ensemble versus 0.493 for the TAAD+DST baseline using retraining and post-processing only. — [arXiv 2606.09679](https://arxiv.org/pdf/2606.09679)
- The 2026 Ball Action Anticipation task (observe 30 s, predict actions in the next 5 s) topped out at mAPavg 24.08 (baseline 16.76). Not relevant for counting events. — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320v1)

**Team-level attribution**
- SoccerNet 2025 team-BAS baseline predicts left/right team per action, with a public checkpoint; scores under Q1. — [sn-teamspotting repo](https://github.com/SoccerNet/sn-teamspotting)

**SoccerTrack v2 (player-linked events on wide footage)**: see Q5.

### Inferences
- FOOTPASS/PCBAS is the only open resource found that goes from broadcast video to (time, class, team, jersey). Its taxonomy matches the product better than SoccerNet BAS for attribution, but still has no interception, duel, dribble take-on or outcome field.
- The best baselines are not "video-in" models: they consume tracklets, jersey numbers, roles and pitch positions of near-certified quality. The prototype's own BoT-SORT tracks, kit-colour teams and rough calibration would be a much noisier input; the ablation without tactical context (F1 49.1) and the no-bounding-box recall figures (2.6-33.4%) indicate how fast the scores fall when that input degrades.
- CC BY-NC 4.0 on annotations and baselines rules out commercial use of FOOTPASS-trained weights without a separate agreement with Footovision/SoccerNet.

### Gaps
- Whether FOOTPASS-trained checkpoints are actually downloadable (the `models/` folder contents and any Hugging Face weights) was not confirmed.
- No public code or weights link was found for the 2026 PCBAS winner (PAVE) or other 2026 entries; the tail of the results paper (supplementary, last ~25k characters) was not read.
- Per-class F1 for the 2026 challenge entries is not reported in the part of the results paper read.
- I did not find an open model coupling SoccerNet-GSR (game state reconstruction) with event spotting beyond FOOTPASS's baselines.
- The FOOTPASS paper says class IDs run "from 0 to 9" while listing 8 classes; unresolved.

---

## Q3. Which open approaches derive events from tracking data rather than raw pixels?

### Takeaway
Published rule-based methods (Vidal-Codina et al.; Khaustov & Mozgovoy) report high detection rates but on provider-grade tracking with a reliable ball track, and I found no official code for either. The usable open code is PathCRF (KDD 2026, MPL-2.0, weights in repo), which infers possession and on-ball events from player trajectories alone; the PySport libraries (kloppy, databallpy, floodlight) load and synchronise data but do not detect events.

### Cited Findings
- Vidal-Codina, Evans, El Fakir, Billingham (MIT Sports Lab / FIFA), "Automatic event detection in football using tracking data": deterministic decision-tree algorithm in two steps, (1) a possession step assigning the ball to a player each frame and characterising player configurations at stoppages, (2) an event step using possession changes plus the laws of the game to detect in-game events and set pieces. — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804v3)
- Reported accuracy: "+90% detection rate" for most event categories against manually annotated events, across tournaments and tracking providers. No per-category figures on the abstract page. — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804v3)
- No code repository is linked from the arXiv page; paper text is CC BY 4.0. — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804v3)
- PathCRF, "Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories" (KDD 2026): repo https://github.com/hyunsungkim-ds/pathcrf. Input is player tracking trajectories only (ball track is never given to the model). Output is a per-frame possession state (a player in possession, or ball in transit between two players); state changes are logged as on-ball events, from which the repo builds possession shares, pass networks and event heatmaps. — [PathCRF repo](https://github.com/hyunsungkim-ds/pathcrf)
- PathCRF is trained/evaluated on the Sportec Open DFL dataset (7 Bundesliga 1st/2nd division matches), with events synchronised to tracking using ELASTIC; trained weights for five model variants are included under `saved/`. — [PathCRF repo](https://github.com/hyunsungkim-ds/pathcrf)
- PathCRF licence MPL-2.0; last push 2026-06-08. — [GitHub API](https://api.github.com/repos/hyunsungkim-ds/pathcrf)
- DataBallPy loads and synchronises event and tracking data; possession can be estimated when combined with event data (i.e. it needs event data, it does not create it). MIT, last push 2026-11-03. — [DataBallPy JOSS paper](https://www.theoj.org/joss-papers/joss.10223/10.21105.joss.10223.pdf); [GitHub API](https://api.github.com/repos/Alek050/databallpy)
- kloppy standardises vendor event and tracking data; it does not detect events. BSD-3-Clause, last push 2026-11-06. — [kloppy repo](https://github.com/PySport/kloppy); [GitHub API](https://api.github.com/repos/PySport/kloppy)
- floodlight is a framework for loading and processing tracking/event data from providers. MIT, last push 2026-05-04. — [floodlight repo](https://github.com/floodlight-sports/floodlight); [GitHub API](https://api.github.com/repos/floodlight-sports/floodlight)
- OpenSTARLab provides a preprocessing package, an Event modelling package (deep-learning event prediction) and a video labelling tool; Event repo is Apache-2.0, last push 2025-09-03. — [OpenSTARLab paper](https://arxiv.org/html/2502.02785v2); [GitHub API](https://api.github.com/repos/open-starlab/Event)
- PassNet (Sorano et al., 2020): detects passes in broadcast video with ResNet18 features + bidirectional LSTM + YOLOv3 ball/player positions; trained on four matches; code at github.com/jonpappalord/PassNet; dataset listed as research-only. Repo has no licence detected and last push 2020-11-07. — [PassNet paper](https://arxiv.org/pdf/2007.06475); [PassNet dataset page](https://ckan-d4s1.d4science.org/dataset/passnet); [GitHub API](https://api.github.com/repos/jonpappalord/PassNet)
- SoccerTrack v2's ball-action baseline is trained on player trajectories, with and without the ball track (search-result summary of the paper). — [SoccerTrack v2 paper](https://arxiv.org/pdf/2508.01802)

### Inferences
- PathCRF is the best fit for the prototype's situation (player tracks available, ball track unreliable): it needs only player trajectories in pitch coordinates and ships weights under a permissive-ish (file-level copyleft) licence. Its output is possession/receiver sequence, which directly yields passes, receptions, possession share and turnovers, but not shots, tackles, duels or dribbles as named classes.
- Rule-based detection in the Vidal-Codina style is what the prototype already attempts; the published 90% rates assume accurate ball tracking plus in-play/dead-ball flags, which is the likely reason the current rules "find almost nothing" on broadcast clips with rough calibration.
- A hybrid is implied by the evidence: video spotting (T-DEED-style) for when/what, tracking-based possession (PathCRF-style or nearest-player rules) for who and for outcome inference.

### Gaps
- Khaustov & Mozgovoy, "Recognizing Events in Spatiotemporal Soccer Data" (Applied Sciences 10(22):8046): the MDPI page returned HTTP 403, so the event list, precision/recall figures and code availability could not be read. Not cited.
- PathCRF numeric accuracy (event recall vs tolerance is only shown as a figure in the README) was not obtained; the KDD paper was not read.
- Sportec Open DFL dataset licence was not checked in this session.
- ETSY and "sloth" synchronisation tools were not found/checked; they synchronise existing event data to tracking rather than detect events, as far as I recall, but I have no source read today. ELASTIC is referenced by PathCRF for the same purpose.
- No independent implementation of Vidal-Codina's algorithm was found on GitHub in the searches run.
- The University of Zurich thesis "An Open Source Implementation of FIFA's Enhanced Football Intelligence" appeared in search results with a section on automatic event detection from tracking data, but I did not read it. — pointer only: https://capuana.ifi.uzh.ch/publications/PDFs/24098_An_Open_Source_Implementation_of_FIFA_s_Enhanced_Football_Intelligence.pdf

---

## Q4. Pass outcome, duel outcome, tackle outcome, dribble success: is there any open model or documented rule set?

### Takeaway
No open model or dataset found labels outcomes for passes, duels, tackles or dribbles from video. Open taxonomies carry outcome only implicitly (a "Player Successful Tackle" class, a "Goal" class, "Out"), so outcomes have to be inferred by rules from the event/possession sequence or supplied by human review.

### Cited Findings
- SoccerNet BAS has "Player Successful Tackle" and "Goal" and "Out" classes but no unsuccessful-tackle, duel, interception or pass-completed class. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- FOOTPASS describes no success/failure annotation; a cross is labelled regardless of completion. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- SoccerTrack v2 events were converted from BePro provider events into the same 12 classes, with manual checks; the 12 classes again contain only "Player Successful Tackle" as an outcome-bearing label. — [SoccerTrack v2 paper](https://arxiv.org/pdf/2508.01802); [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/)
- Tackle is the weakest class in the only per-class table found: 0.0 precision / 0.0 recall for TAAD+DST, 2.4 / 25.0 for TAAD+GNN (Tackle is 0.3% of FOOTPASS events), and was "the hardest class for all participants" in 2026. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1); [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320v1)
- The FOOTPASS authors state that current action recognition methods are not reliable enough to automate play-by-play extraction and are mostly used to assist annotation. — [FOOTPASS abstract](https://arxiv.org/abs/2511.16183v1)
- PassNet detects that a pass occurred, not whether it was completed. — [PassNet paper](https://arxiv.org/pdf/2007.06475)
- Vidal-Codina et al. derive events from possession changes plus the laws of the game, which is the documented rule-based route to outcome (who has the ball next). — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804v3)
- The 2025 runner-up corrected SHOT/GOAL team labels using the preceding correlated actions, an example of sequence rules layered on spotting output. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)

### Inferences
- Pass completion is the only outcome that can be derived reasonably from open components: next on-ball event by the same team (complete), by the opponent (incomplete/intercepted), or Out. This needs the team label from the 2025 baseline or a possession model.
- Interceptions and turnovers likewise fall out of team changes between consecutive events rather than from any trained class.
- Duel outcome, tackle outcome (won/lost/foul) and dribble (take-on) success have no open training labels in any dataset found; they would have to be defined by in-house rules over tracks (proximity of opponents at a drive, possession retained after N frames) and validated by human review. The product should present these as "proposed, needs review".

### Gaps
- I found no open dataset with explicit duel or take-on labels from video. Open provider event data (e.g. StatsBomb open data) carries such outcome fields but has no aligned public video; I did not verify this in the session, so it is not cited.
- No published accuracy for rule-derived pass completion from broadcast-derived tracking was found.

---

## Q5. What do fixed wide-angle / tactical-camera or amateur-footage projects use for events? Is there an open event dataset for full-pitch footage?

### Takeaway
SoccerTrack v2 is the one open full-pitch dataset with event labels: 10 amateur university matches of fixed 4K panoramic video with the 12 SoccerNet-style ball-action classes linked to player IDs, released under CC BY 4.0 (commercial use allowed) with MIT starter-kit code. No BAS baseline numbers or pretrained event weights were confirmed for it.

### Cited Findings
- SoccerTrack v2 (arXiv 2508.01802, Aug 2025; Atom Scott et al.): 10 full university-level matches, about 932 minutes of fixed-camera 4K panoramic video. — [SoccerTrack v2 paper](https://arxiv.org/pdf/2508.01802)
- Cameras: two matches with a BePro Cerberus camera, eight with BePro's 3-camera panoramic stitching system. — [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/)
- Annotations: per-frame 2D pitch coordinates (metres), persistent track IDs, player IDs, role, team side, jersey number; BAS events with match-clock time, millisecond timestamp, action label, team and player ID; MOT boxes for a subset. — [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/)
- BAS classes: Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal (same 12 as SoccerNet BAS). Events came from BePro and were converted to these classes with manual checks. — [SoccerTrack v2 paper](https://arxiv.org/pdf/2508.01802); [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/)
- Licences: code MIT; dataset (videos and annotations) CC BY 4.0; the project page says both "permit academic and commercial use, with attribution". No player names; university ethics approval. — [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/)
- Download: Hugging Face https://huggingface.co/datasets/atomscott/soccertrack-v2 (canonical), Google Drive mirror; repo https://github.com/AtomScott/SoccerTrack-v2 with `baselines/` starter kits for GSR, BAS and MOT. — [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/); [SoccerTrack v2 repo](https://github.com/AtomScott/SoccerTrack-v2)
- Repo last push 2026-10-05 (actively maintained); 139 commits. — [GitHub API](https://api.github.com/repos/AtomScott/SoccerTrack-v2); [SoccerTrack v2 repo](https://github.com/AtomScott/SoccerTrack-v2)
- In the paper, GS-HOTA degrades as sequence length increases, and the BAS baseline is trained on player trajectories with and without the ball track. — [SoccerTrack v2 paper](https://arxiv.org/pdf/2508.01802)
- The MOT subset was the basis of the SoccerTrack Challenge 2025. — [SoccerTrack v2 repo](https://github.com/AtomScott/SoccerTrack-v2)
- "Beyond the Premier" (Cabado et al., CVPR Workshops 2024) benchmarked transfer of SoccerNet-trained action-spotting models to four domains including amateur leagues and AI-piloted (automatic) camera footage. — [CVPRW 2024 paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.pdf)
- SoccerNet 2026 included a "Spiideo SoccerNet Synloc" task (Spiideo is a fixed-camera vendor); it is a localisation task, not an event task. — [SoccerNet 2026 Challenges Results](https://arxiv.org/abs/2607.07320)

### Inferences
- SoccerTrack v2 is the most relevant resource for the product's eventual target (fixed wide camera, full half): same 12-class taxonomy as the public T-DEED checkpoints, player-linked events, and a licence that permits commercial training. It is the natural fine-tuning/evaluation set for moving a SoccerNet-pretrained spotter to wide footage.
- Ten matches is small; class counts for rare events (shots, tackles, goals) will be in the tens to low hundreds, so expect weak performance on rare classes without extra labelled data.
- On a full-pitch panorama players are a few dozen pixels tall, so pixel-based spotters trained on broadcast close-ups will not transfer directly; the dataset authors' own choice of a trajectory-based BAS baseline points to tracking-derived events as the more workable approach for wide cameras.

### Gaps
- SoccerTrack v2 BAS baseline mAP figures, event counts per class and video fps were not present in the pages read (paper HTML summary, project page, repo README).
- Whether pretrained BAS or GSR weights are released in `baselines/` is not stated.
- GitHub reports the repo licence as NOASSERTION (not auto-detected) even though the project page says MIT; the LICENSE file text was not read.
- No published result was found for applying T-DEED or other SoccerNet-trained weights to SoccerTrack v2 panoramic video.
- TeamTrack, SoccerNet-Tracking, Veo/Pixellot-style research and Spiideo open research were not found to include event labels in the searches run; not individually verified.
- SoccerTrack v2 originates from the BePro capture pipeline at university level in Japan; domain match to the customer's camera hardware is unknown.

---

## Q6. Realistic expectations on unseen-league broadcast footage with the best public weights

### Takeaway
No published per-class precision/recall for 12-class BAS on unseen leagues was found; in-domain numbers are already modest (public T-DEED team baseline 53.59 mAP@1 / 47.18 Team-mAP@1 on the SoccerNet test split) and the one cross-domain study reports that transfer across leagues is hard. In the only per-class precision/recall table found (FOOTPASS, in-domain, with high-grade tracking inputs), passes and drives sit around 70% precision and recall, shots about 65%, headers about 30%, blocks about 20% and tackles at 0%.

### Cited Findings
- Public checkpoint, in-domain: SoccerNet 2025 T-DEED team baseline scores Team-mAP@1 47.18 and mAP@1 53.59 on the test split, 51.72 / 58.38 on the challenge split; the README warns reproduced results may differ slightly. — [sn-teamspotting repo](https://github.com/SoccerNet/sn-teamspotting)
- Best in-domain 12-class result ever reported: 73.39 mAP@1 (2024, T-DEED ensemble, no team); best with team: 60.03 Team-mAP@1 (2025). — [SoccerNet 2024 Challenges Results](https://arxiv.org/html/2409.10587v1); [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- In 2025, the entry with the best plain mAP@1 (UniBW Munich, 68.92) scored only 55.36 Team-mAP@1, i.e. team attribution costs about 13 points even in-domain. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182v1)
- Per-class precision/recall, in-domain, player-attributed (FOOTPASS TAAD+DST, confidence 0.15, +/-12 frames): Pass 72.6/72.1; Drive 68.2/68.9; Shot 63.4/67.2; Throw-in 65.9/58.6; Cross 62.8/51.8; Header 30.7/26.3; Block 23.3/14.7; Tackle 0.0/0.0. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- Vision-only baseline on the same test (TAAD): overall precision 25.6, recall 59.9; Pass 43.8/61.0; Shot 12.3/74.6. — [FOOTPASS paper](https://arxiv.org/html/2511.16183v1)
- Best 2026 player-centric result after a full challenge cycle: 58.94 Macro F1@0.15 (baseline 46.41). — [SoccerNet 2026 Challenges Results](https://arxiv.org/html/2607.07320v1)
- Cross-domain: Cabado et al. found transfer of SoccerNet-trained action-spotting models across leagues "proved hard", with broadcast editing quality the most impactful factor (per search-result summary of the paper; the full text was not retrievable). — [CVPRW 2024 paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.pdf)
- A 2025 survey lists poor generalisation and unrealistic evaluation schemes as open challenges for sports action spotting. — [Survey arXiv 2505.03991](https://arxiv.org/html/2505.03991v2)
- SoccerNet BAS training footage is English Football League broadcast at 720p; T-DEED's SoccerNetBall inference example uses 796x448 frames. — [SoccerNet 2024 Challenges Results](https://arxiv.org/html/2409.10587v1); [T-DEED repo](https://github.com/arturxe2/T-DEED)

### Inferences
- mAP is a ranking metric over all confidence thresholds; it is not the precision/recall the product would show at a fixed operating point. Counts displayed to users depend on a chosen threshold, and the FOOTPASS table is the only source found that reports such operating-point figures.
- A reasonable planning assumption from the cited evidence (not a published figure): on unseen-league broadcast clips with the public T-DEED team checkpoint, frequent classes (Pass, Drive) will be usable as proposals with a meaningful false-positive rate; Shot and Goal are rare and will be noisy in count terms; Player Successful Tackle and Ball Player Block will be unreliable; interceptions, duels and dribble outcomes cannot be produced by the model at all. Expect below the in-domain 47-54 mAP@1 baseline, since both league and broadcast style differ.
- 1080p 25 fps input is compatible after downscaling (the public models run at 448p or lower); a 120 s clip is 3,000 frames, small relative to the full halves these models are evaluated on. Nothing in the sources indicates these models are too heavy for an L4/A10G, but no source states inference throughput.
- All published evidence supports keeping a human review step: the dataset authors themselves say current methods mainly assist annotation rather than replace it.

### Gaps
- No cross-dataset per-class precision/recall for 12-class BAS (e.g. SoccerNet-trained model tested on FOOTPASS, SoccerTrack v2 or another league) was found.
- Quantitative drops from the Cabado et al. paper were not obtained (CVF page returned HTTP 403).
- No inference speed or GPU memory figures on L4/A10G were found for T-DEED, TAAD or the team baseline.
- No SoccerNet 2026 or later source reports a refreshed 12-class BAS leaderboard; the latest 12-class numbers are from 2025. I could not confirm any newer open 12-class BAS model released between mid-2025 and November 2026.
