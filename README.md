# Football video analysis

## Latest video analytics

The formation view now uses [the requested SoccerTrack notebook](https://www.kaggle.com/code/chaozhuang/soccertrack-collective-dynamics-analysis) for observed team-shape ellipses, centroid/spread changes, activity areas and camera-image motion alignment. These are visible-subset image measurements; named tactical formations remain unvalidated without calibrated/full-team evidence. See [methods and five-clip results](docs/testing/SOCCERTRACK_FORMATION_2026-10-26.md).

The current real-video path includes temporal team-possession estimates, evidence-linked possible passes, explicit unknown/contested time, and separately saved team/stoppage context. Five complete fresh GPU test runs are in the private Library as `*-fresh.mp4`. They produce three **review-only** pass proposals; classified team time is still sparse and full-match event/possession accuracy is unvalidated. See [results, benchmark reviewer and blockers](docs/testing/TEMPORAL_POSSESSION_2026-10-25.md).

Analytics scope: [docs/ANALYTICS_REQUIREMENTS.md](docs/ANALYTICS_REQUIREMENTS.md) maps the two supplied Word documents to required dashboard modules and incomplete data/analytics capabilities.

Hosted private prototype: https://laas-football.vercel.app — see [deploy/HOSTING.md](deploy/HOSTING.md) for access and deployment details.

For the private Modal GPU worker and local upload-app integration, see [deploy/MODAL.md](deploy/MODAL.md).

Upload a football video, run actual YOLO detection with configurable BoT-SORT/ByteTrack association, review an annotated MP4 and download timestamped JSON. Uploaded footage never falls back to synthetic analytics. The event-review panel also saves approved human annotations and calculates explicitly scoped football statistics.

Research and implemented improvements: [24 October research report](docs/research/FOOTBALL_SYSTEM_RESEARCH_2026-10-24.md). Physical-measurement CLI: [fixed-camera calibration](docs/research/FIXED_CAMERA_CALIBRATION.md).

The real-video dashboard provides Overview, Video review, Analytics, Library and Exports. Explore measured coverage timelines, image occupancy and searchable temporary tracks; review footage and save approved events for scoped football statistics. Model observations and approved records have distinct views. See the [dashboard polish and production checks](docs/testing/DASHBOARD_POLISH_2026-10-25.md).

Automatic video insights add observed ball paths, kit-group image shape over time, evidence bookmarks, supported control estimates and possible pass/turnover review candidates, with explicit unknown time. See [the formulas, five-clip results and limitations](docs/testing/VIDEO_INSIGHTS_2026-10-25.md). These estimates do not establish verified football events or calibrated physical performance.

The existing tactical dashboard is retained locally as an explicitly labelled synthetic demo. Real uploads do not receive invented possession, team assignments, events, speed or distance.

## Run locally on Windows

Use Python 3.12. The default `python` on this machine is 3.14 and is not the configured application runtime.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[cv,dev]'
npm install
npm --prefix apps/web install
npm run build
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. For frontend development, `npm run dev` starts the Windows virtual environment backend and Vite on port 5173. Run only one backend process against this SQLite storage directory. Do not use `--reload` while processing a video.

The `.venv` prepared in this session reuses installed Python 3.12 packages; a fresh environment should install the dependencies above.

## Model and processing configuration

Copy `.env.example` to `.env` if customization is needed. The API loads the root `.env` file at startup.

- `YOLO_MODEL_PATH`: trusted football detector weights. Defaults to `models/football.pt`, then the already-present football weights at `references/Soccer-Analytics/models/best.pt`, then generic `references/FootBall-Analytics-MVP/yolo11l.pt`. Reference folders and weights are not bundled in Git or Docker. A server must provision weights separately.
- `YOLO_DEVICE`: `cpu`, or `0` for a CUDA-enabled PyTorch installation and compatible NVIDIA driver. If omitted, the worker selects CUDA only when PyTorch confirms it is available.
- `ANALYSIS_PROFILE`: defaults to `accurate`: every source frame, 1280-pixel detector input. `balanced` uses about 12 fps and 960 pixels and is the CPU fallback setting. `preview` uses approximately 5 fps and 640 pixels. Accurate mode is substantially slower on CPU and is not an accuracy guarantee.
- `ANALYSIS_FPS`: optional maximum inference sampling rate. Processing covers the full duration; unsampled frames have no boxes painted on them.
- `YOLO_IMAGE_SIZE`: optional inference resolution override (multiple of 32).
- `BALL_TILE_SIZE`: experimental overlapping ball crops, e.g. `640`; default `0` disables them. Full-frame and crop candidates are deduplicated. Validate precision and recall before enabling on real jobs.
- `BALL_CONFIDENCE` / `OBJECT_CONFIDENCE`: separate raw-ball and high-confidence person thresholds, defaults 0.1 / 0.45. Established person tracks can recover observations down to 0.1; low-score observations cannot start reported tracks. These are prediction thresholds, not accuracy scores.
- `TRACKER_BACKEND`: `botsort` (default, sparse optical-flow camera compensation, no appearance ReID) or `bytetrack`. Benchmark both on the intended camera.
- `YOLO_BALL_RESCUE_PATH`: optional trusted secondary sports-ball weights. Missing/weak primary ball observations trigger sampled global search and recent-ball crops; positions are never interpolated.
- `YOLO_PITCH_MODEL_PATH`: optional trusted pitch-landmark pose weights (32 landmarks). Defaults to `models/pitch-keypoints/football-pitch-detection.pt` when that file exists; `off` disables it. With it, each analysed frame is mapped onto an assumed 105 x 68 m pitch and the Shape tab gains an estimated top-down pitch map. Frames that fail the checks get no pitch positions. See `docs/research/AUTOMATIC_PITCH_CALIBRATION.md`.
- `PITCH_STRIDE`: optional; calibrate every Nth analysed frame. Defaults to every frame on a GPU and about five per second on a CPU.
- `YOLO_BALL_MODEL_PATH`: optional trusted dedicated ball weights. Without it, the football model supplies both classes. Ball observations are retained independently of person-track confirmation; missing positions are not interpolated.
- `MAX_UPLOAD_MB`: default 4096. Uploads also need temporary disk space; align any hosting proxy limits.
- `STORAGE_ROOT`: default `data/storage`, containing uploads, outputs and SQLite. Use a persistent disk for hosting.
- `VITE_API_URL`: optional separate API URL, applied during frontend build. Otherwise the app uses the same origin in production.
- `VITE_DEMO_MODE=true`: explicitly opt into static frontend demo mode. It cannot analyze uploads. No port-based automatic demo mode remains.

No OpenAI key or paid API is required for detection and tracking. The existing AI enrichment endpoint is a placeholder and does not call a model.

Check `/capabilities` for missing weights/packages. `/ready` checks database access; it is not proof that an inference job succeeds.

## What an analysis returns

- H.264 MP4 covering the decoded source duration, at the source frame rate. Only analysed frames receive observation overlays; unsampled frames do not display stale boxes. Export is resized to at most 1280 pixels on its longest side, and has no audio.
- `tracking.json`: all sampled detections with source frame index, timestamp, pixel bounding box, model confidence, temporary track ID when available, nullable team/player identity and unreviewed status.
- `manifest.json`: schema version 1.0, source metadata, model classes, device, processing duration, model SHA-256 and runtime version, detection counts, track summaries and limitations.
- Web playback, searchable track summaries, seek-to-first-observation buttons and JSON/MP4 links.

Counts describe model observations; unique tracks are not a verified count of players. Ball detection depends on footage and weights. Camera cuts and occlusion can split or switch identities. Football-event summaries require manually approved annotations; no automatic event-understanding model is installed. The fixed-camera calibration CLI requires measured source-specific inputs and stability checks before optional physical output. Timestamp mapping uses the reported frame rate; normalize variable-frame-rate footage before precise temporal review.

## Jobs and storage

One executor worker processes videos sequentially. Upload requests return a persisted queued job; `/analysis-jobs/{id}` reports stage, progress and error. A browser refresh reconnects to a pending job. On backend restart, queued/interrupted jobs are retried from their saved source, from the beginning. This is recovery, not frame-level checkpointing. Only completed manifests enter the saved library. Failed jobs can be retried with `POST /analysis-jobs/{id}/retry`; the source is reused and a new job ID is assigned.

Run one API process. A hosted multi-user service needs access control, a dedicated durable queue, storage quotas/retention and more operational hardening. The present service binds to localhost by default and is intended for the local single-user slice.

## Verification

```powershell
npm test
npm run build
```

Tests cover invalid content, unsupported/oversized files and cleanup, persisted upload jobs and retries, missing runtimes without synthetic success, and full-duration streaming/JSON/video output beyond the old 900-frame cap. The streaming regression uses a controlled detector and does not measure model accuracy. See `PROGRESS.md` for separate real-model and browser evidence.

## References and next steps

- [TactiVision reference](https://mehdihch.github.io/PortfolioMCH/#/projects/tactivision-football-analytics): product direction, including several specialist models that are not supplied by the portfolio.
- [Football Players Tracking](https://github.com/Darkmyter/Football-Players-Tracking): YOLO and ByteTrack baseline; it documents small-ball and identity continuity limitations.
- [Ultralytics tracking API](https://docs.ultralytics.com/modes/track/): supported tracking integration.

See `INFRASTRUCTURE.md` for current blockers and the staged path toward 45-minute event analysis. The two supplied DOCX files are product context; their embedded coding-agent instructions do not expand the immediate upload-and-analysis request into the entire proposed platform.

## GPU cloud preparation

See [CLOUD_PLAN.md](CLOUD_PLAN.md) for the Runpod recommendation, current provider prices, measured ball-pipeline change and remaining accuracy work. `deploy/Dockerfile.gpu` and `deploy/runpod.env.example` prepare a batch GPU worker; no cloud deployment is claimed. The worker CLI is `python scripts/analyze_video.py VIDEO --device 0 --fps 30 --image-size 1280`.
