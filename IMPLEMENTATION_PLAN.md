# Football AI Analytics App - End-to-End Implementation Plan

## 1. Product Goal

Build a coach-facing AI football analytics platform that turns raw match footage and structured football data into tactical intelligence:

- Upload match/training video and optional event/load/player datasets.
- Detect and track players, ball, referees, pitch keypoints, and zones.
- Convert broadcast footage into pitch coordinates and tactical-map views.
- Analyze team vs opposition across phases of play, not just static formations.
- Produce annotated video, event JSON/CSV, dashboards, scouting recommendations, injury risk indicators, and AI tactical reports.
- Present everything in a premium, smooth, fast UI suitable for coaches, analysts, scouts, and performance staff.

The PDF's core idea should be treated as the product north star: "Video -> Data -> Tactical Understanding", with computer vision, LLM tactical reasoning, and coach insight combined into one workflow.

## 2. Reference Repo Takeaways

Use these repositories as implementation references, not as direct copy-paste dependencies unless license and model-weight rights are confirmed.

| Source | Useful parts | Recommendation |
| --- | --- | --- |
| Somnath0707/FootBall-Analytics-MVP | Existing half-done YOLO11 pipeline: player tracking, KMeans team assignment, ball possession, interpolation, annotated output | Use as the starting conceptual MVP, but refactor into API workers and structured outputs. Add input/output folders, persistent jobs, tests, and UI. |
| nreHieW/Eagle | Strongest broadcast-to-tracking foundation: YOLO + BoTSORT, HRNet pitch keypoints, homography, UEFA 105x68 coordinates, processed_data.json schema | Use as the target architecture for pitch calibration and coordinate normalization. This is the best reference for serious tactical-map accuracy. |
| abdullahtarek/football_analysis | YOLO, ByteTrack, optical flow camera compensation, perspective transform, speed/distance, possession | Use for speed/distance and camera-motion module patterns. |
| Hmzbo/Football-Analytics-with-Deep-Learning-and-Computer-Vision and keshh22/FootyTacticalAnalysis | Tactical map Streamlit concept, pitch keypoint datasets, predefined team colors | Use as prototype inspiration only; replace Streamlit with production web UI. |
| TrishamBP/football_analysis_yolo and Ayan-OP/Soccer-Analytics | Clean pipeline documentation for tracking, camera motion, KMeans teams, possession, speed/distance | Use to validate module boundaries and early MVP outputs. |
| calvinyeungck/Football-Match-Event-Forecast-v2 | StatsBomb 360/event pipeline, PrefixSpan tactic extraction, NMSTPP, HPUS/HPUS+ metrics | Use for phase/tactic mining once structured event and coordinate data exists. |
| LaErre9/Player_Scouting_Recommendation_System | Player similarity, autocomplete search, Solr/vector retrieval, AI-generated scouting reports | Use for scouting module design; update LangChain/OpenAI stack and use modern vector search. |
| SeanZ509/InjuryPrevention | Supervised injury-risk example on weekly time series | Use only as a simple baseline pattern. Real injury model needs football-specific load, wellness, minutes, sprint, accel/decel, and recovery data. |
| matiasmascioto/awesome-soccer-analytics | Resource directory for datasets, providers, libraries, references | Use for sourcing datasets/providers and validating analytics methods. |
| Nature TacticAI paper | Graph representation of player states, retrieval of similar tactical situations, predictive/generative tactical recommendations | Use as the advanced roadmap for set-piece/tactical-situation retrieval and recommendation. |

## 3. Recommended Stack

### Frontend

- Next.js + TypeScript + React.
- Tailwind CSS plus shadcn-style primitives, but heavily customized for a premium dark/light sports analysis interface.
- Framer Motion for smooth transitions, upload progress, timeline scrubbing, and panel changes.
- Recharts/Plotly for charts; D3 for custom pitch/tactical visualizations.
- Video.js or custom HTML5 video player with synchronized overlay canvas.
- Canvas/SVG hybrid pitch renderer for heatmaps, formations, trajectories, Voronoi, pass maps, and phase shapes.

### Backend/API

- FastAPI Python service for analytics APIs, uploads, job control, and model serving.
- Celery/RQ/Arq workers for long-running video processing.
- Redis for job queue, progress, cache, and WebSocket pub/sub.
- PostgreSQL for users, teams, matches, players, events, model runs, reports.
- TimescaleDB extension or partitioned PostgreSQL tables for frame-level tracking/time-series data.
- Object storage: S3-compatible bucket or local MinIO for videos, annotated clips, model artifacts, JSON/CSV outputs.
- WebSockets or Server-Sent Events for live processing progress.

### ML/CV

- Python, PyTorch, Ultralytics YOLO11/YOLOv8, OpenCV, Supervision, BoTSORT/ByteTrack.
- HRNet or equivalent pitch keypoint detector for homography.
- Optional ONNX Runtime/TensorRT export path for faster inference.
- Pandas/Polars, NumPy, scikit-learn, scipy.
- PyTorch Geometric later for graph tactical models inspired by TacticAI.

### LLM/AI Reasoning

- LLM service wrapper with strict JSON outputs and citations to source metrics.
- RAG over match reports, generated metrics, team principles, coach notes, opposition profiles, and scouting docs.
- Store generated reports with provenance: each claim must link back to a metric, clip, event, or model output.

## 4. System Architecture

### Core Services

1. Web App
   - Upload center, match workspace, video analysis page, tactical board, scouting, player development, injury dashboard, report center.

2. API Gateway
   - Auth, teams/clubs, match metadata, uploads, analysis jobs, dashboards, report generation.

3. Video Processing Worker
   - Frame extraction, detection, tracking, pitch calibration, homography, coordinate projection, annotated video generation.

4. Analytics Worker
   - Possession, speed, distance, heatmaps, phases, formation states, transitions, event detection, team/opponent comparison.

5. AI Report Worker
   - Turns metrics and visual evidence into coach-facing summaries, tactical recommendations, and scouting reports.

6. Data Layer
   - PostgreSQL/TimescaleDB for structured data.
   - Object storage for raw and generated media.
   - Vector database for player/scenario similarity and report retrieval.

## 5. Data Model

### Main Entities

- Club
- Team
- Season
- Competition
- Match
- Player
- SquadMembership
- VideoAsset
- AnalysisJob
- FrameTracking
- Event
- PossessionSegment
- TacticalPhase
- PlayerLoadSession
- InjuryRiskSnapshot
- ScoutingProfile
- AIReport

### Tracking Schema

Use Eagle-style normalized pitch coordinates:

- Pitch: 105 x 68 meters.
- Frame row:
  - match_id
  - video_id
  - frame_idx
  - timestamp_ms
  - object_type: player, goalkeeper, referee, ball
  - track_id
  - team_id nullable
  - bbox_video: x1,y1,x2,y2
  - bottom_center_video
  - pitch_x, pitch_y nullable
  - confidence
  - homography_quality

### Event Schema

- event_type: pass, shot, carry, tackle, interception, turnover, goal, pressure, transition, phase_change
- actor_player_id nullable
- team_id
- start_time_ms / end_time_ms
- start_x/start_y, end_x/end_y
- outcome
- source: cv, imported_provider, manual, ai_inferred
- confidence

## 6. ML Pipeline

### Phase A: Video Ingestion

1. Upload raw video.
2. Extract metadata: duration, FPS, resolution, codec.
3. Create analysis job and preview thumbnails.
4. Optionally trim clips before full processing.

### Phase B: Detection and Tracking

1. Detect players, goalkeepers, referees, and ball using YOLO11/YOLOv8.
2. Track entities with BoTSORT or ByteTrack.
3. Normalize goalkeeper into player role where needed, but preserve role metadata.
4. Interpolate missing ball detections.
5. Detect team assignment using:
   - MVP: KMeans jersey clustering.
   - Later: manual correction UI plus ReID/team classifier.

### Phase C: Pitch Calibration

1. Detect pitch keypoints using HRNet-style model.
2. Propagate keypoints with optical flow between detection intervals.
3. Estimate homography from image points to UEFA 105x68 pitch.
4. Project player bottom-center points and ball positions to pitch coordinates.
5. Save per-frame quality scores and null coordinates when confidence is too low.

### Phase D: Metric Generation

1. Player trajectories and heatmaps.
2. Ball possession timeline and possession segments.
3. Speed, sprint counts, distance covered, acceleration/deceleration.
4. Team width, depth, compactness, line height, defensive block height.
5. Pressing indicators: distance to ball, nearest pressure, PPDA-style proxy when event data exists.
6. Zone control and Voronoi/pitch-control approximations.
7. Pass/carry/shot detection:
   - MVP: heuristic from ball/player proximity and trajectory.
   - Later: trained event classifier using event data and tracking windows.

### Phase E: Tactical Phase Analysis

The PDF specifically requires dynamic tactical shapes:

- Build-up
- Progression
- Final attack
- Out of possession
- Defensive transition
- Attacking transition
- Set pieces

Implementation:

1. Segment possessions from ball control and event stream.
2. Classify phase using rules first:
   - Ball zone, team possession, direction, speed of progression, opponent pressure, field third.
3. Calculate shape features per phase:
   - Formation skeleton, average player coordinates, width/depth, line distances, compactness, overloads.
4. Detect shape changes:
   - In-possession shape vs out-of-possession shape.
   - Phase-specific formations, e.g. 4-3-3 defending, 3-2-5 attacking.
5. Later train supervised phase classifier if labelled data is available.

### Phase F: Tactical Pattern Mining

Use event and coordinate sequences to identify:

- Fast progression.
- Possession recycling.
- Wide development.
- Direct attacking options.
- Press traps.
- Repeated buildup exits.
- Opposition weakness zones.

MVP implementation:

- Rule-based pattern templates plus dashboard visualizations.

Advanced implementation:

- PrefixSpan-style sequence mining from Football-Match-Event-Forecast-v2.
- HPUS/HPUS+ style possession value metrics.
- Graph embeddings for similar tactical scenarios inspired by TacticAI.

## 7. LLM Tactical Reasoning

The LLM should not invent metrics. It should explain and prioritize evidence already produced by the analytics layer.

### Inputs to the LLM

- Match metadata.
- Team principles and coach notes.
- Phase metrics.
- Event summary.
- Tracking-derived shape metrics.
- Clips and timestamps.
- Opponent profile.
- Historical/seasonal benchmarks.

### Outputs

- Coach summary: what happened, why it matters, recommended actions.
- Opposition report: strengths, weaknesses, triggers, threats.
- Player development report: top strengths, improvement areas, clips.
- Scouting report: similar players, fit, risk, value notes.
- Injury/performance alert: workload anomalies and recommended follow-up.

### Guardrails

- Every recommendation must cite metric IDs, event IDs, timestamps, or source files.
- Separate "observed" from "inferred".
- Provide confidence and data-quality warnings.
- Allow coach feedback to approve, reject, or annotate AI conclusions.

## 8. Product UI Plan

The app should feel like a high-end analysis studio, not a notebook or generic dashboard.

### Visual Direction

- Dense but elegant operational UI.
- Dark analysis workspace with high-contrast pitch graphics.
- Smooth timeline scrubber and synchronized video/pitch map.
- Minimal chrome, strong typography, subtle motion.
- No marketing-style landing page as the primary experience.

### Main Screens

1. Match Library
   - Search/filter by team, opponent, competition, date, processing status.
   - Compact match cards with score, xG, possession, key phase weakness.

2. Upload and Processing
   - Drag/drop video.
   - Optional team colors, lineups, pitch orientation, opponent, competition.
   - Real-time job progress: extraction, detection, calibration, metrics, report.

3. Match Analysis Workspace
   - Left: video with overlays.
   - Center/right: tactical pitch map synchronized to video.
   - Bottom: event and possession timeline.
   - Side panels: metrics, phase selector, clips, AI insights.

4. Tactical Phases
   - Tabs for build-up, progression, final attack, out of possession, transitions.
   - Show formation shape per phase, team width/depth, player average positions, opposition comparison.

5. Opposition Analysis
   - Benchmark our team vs opponent.
   - Threat players, dangerous zones, recurring patterns, pressing vulnerabilities.

6. Player Development
   - Player profile, trajectories, heatmaps, trend lines, role-specific KPIs, coach notes.

7. Scouting
   - Search/autocomplete players.
   - Similar-player recommendations.
   - Radar, percentile charts, role fit, AI scouting report.

8. Injury and Load
   - Weekly workload, acute/chronic load, sprint exposure, fatigue flags.
   - Risk alerts as decision support, not medical diagnosis.

9. Report Builder
   - Generate PDF/HTML/video-clip reports.
   - Coach can edit comments and export.

## 9. MVP Scope

Build the first useful version around the half-done MVP and add the missing product shell.

### MVP Features

- Upload a match clip.
- Run YOLO detection/tracking.
- Team assignment by jersey color with manual correction.
- Ball possession timeline.
- Annotated output video.
- Basic tactical pitch map from either simple perspective transform or integrated Eagle-style calibration.
- Heatmaps, trajectories, speed/distance.
- Basic phase tagging with rule-based classification.
- AI match summary grounded in generated metrics.
- Smooth web UI with processing progress and match workspace.

### MVP Non-Goals

- Fully reliable real-time live match analysis.
- Medical-grade injury prediction.
- Fully automated event detection for all events.
- Generative tactic optimization.
- Production scouting across all leagues without licensed data.

## 10. Implementation Phases

### Phase 0: Foundations

- Create monorepo:
  - apps/web
  - apps/api
  - workers/video
  - workers/analytics
  - packages/shared-types
  - infra/docker
- Add Docker Compose for PostgreSQL, Redis, MinIO, API, worker, web.
- Define database migrations.
- Define shared schemas for tracking, events, jobs, reports.
- Add auth placeholder and organization/team model.

### Phase 1: Video MVP

- Port/refactor the Somnath MVP into a worker module.
- Replace hardcoded paths with job inputs.
- Save structured tracking JSON, possession CSV, and annotated video.
- Add upload endpoint and job progress.
- Add match workspace UI with video playback and processing status.

### Phase 2: Pitch Coordinates

- Integrate Eagle-style calibration:
  - pitch keypoints
  - homography
  - 105x68 coordinate output
  - processed_data.json-compatible structure
- Add tactical pitch visualization synchronized to video.
- Add coordinate quality diagnostics and manual correction workflow.

### Phase 3: Metrics and Dashboards

- Generate heatmaps, trajectories, distance, speed, sprint approximations.
- Generate possession segments.
- Add team comparison KPIs.
- Add phase classification rules.
- Add phase dashboards and opposition comparison views.

### Phase 4: AI Reports

- Add report generation worker.
- Create strict report JSON schema.
- Generate:
  - match summary
  - phase weaknesses
  - opposition prep
  - player development notes
- Add report editor/export.

### Phase 5: Scouting and Seasonal Data

- Ingest player/team seasonal data from CSV/API/provider.
- Build player search and similarity engine.
- Add role-based player ranking.
- Add AI scouting reports with evidence and limitations.

### Phase 6: Injury/Load Module

- Ingest external load, training, minutes, wellness, recovery data.
- Build workload trend dashboards.
- Implement baseline risk scoring:
  - acute/chronic workload ratio
  - spikes in sprint load
  - high minutes + poor recovery
  - previous injury flag if available
- Add model experimentation only after enough labelled football data exists.

### Phase 7: Advanced Tactical Intelligence

- Add sequence mining for tactics.
- Add possession value metrics inspired by HPUS/HPUS+.
- Add graph embeddings for similar situations.
- Add set-piece retrieval and recommendation prototypes inspired by TacticAI.

## 11. API Surface

### Core Endpoints

- POST /videos/upload
- POST /matches
- POST /matches/{match_id}/analysis-jobs
- GET /analysis-jobs/{job_id}
- GET /matches/{match_id}/tracking
- GET /matches/{match_id}/events
- GET /matches/{match_id}/metrics
- GET /matches/{match_id}/phases
- POST /matches/{match_id}/reports
- GET /reports/{report_id}
- POST /players/import
- GET /players/search
- GET /players/{player_id}/similar

## 12. Verification Plan

### Unit Tests

- Bbox utilities.
- Ball interpolation.
- Team assignment.
- Possession assignment.
- Coordinate conversion.
- Phase classifier.
- Metric calculations.

### Integration Tests

- Upload video -> create job -> worker runs -> output files saved -> DB rows created.
- Tracking JSON validates against schema.
- Annotated video exists and is playable.
- Pitch visualization can load processed tracking.
- AI report generation refuses missing/low-quality metric context.

### E2E Tests

- Playwright:
  - upload a sample clip
  - watch job progress
  - open match workspace
  - scrub video and pitch map together
  - open phase analysis
  - generate report

### ML Quality Checks

- Track detection confidence, ID switches, ball missing-frame ratio.
- Homography quality and pitch coordinate null rate.
- Manual review UI for failed clips.

## 13. Key Risks

- Broadcast video is hard: ball detection, occlusion, camera cuts, and homography failures will happen.
- Team assignment by jersey color is brittle; manual correction is required in MVP.
- "Real-time" should be deferred unless GPU infrastructure is available.
- Injury prediction must be positioned as workload risk support, not medical diagnosis.
- Scouting and seasonal stats require reliable licensed or open data sources.
- Many reference repos have no explicit permissive license; reuse ideas and architecture unless licensing is cleared.

## 14. First Build Sprint

1. Scaffold monorepo, Docker Compose, API, worker, web app.
2. Implement upload and analysis job model.
3. Refactor MVP tracking pipeline into worker.
4. Store tracking JSON, possession CSV, annotated video.
5. Build match workspace:
   - video player
   - processing progress
   - possession timeline
   - basic metrics panel
6. Add first AI summary from generated possession and tracking metrics.
7. Add one sample video flow as a demo.

Success criterion: a user can upload a clip and receive an annotated video, possession stats, basic player/team tracking data, and a polished analysis workspace with a grounded AI summary.

