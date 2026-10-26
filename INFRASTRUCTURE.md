# Infrastructure and model readiness

Assessment date: 16 October 2026. This document distinguishes the immediate local upload-and-tracking slice from the broader product in the supplied briefs.

## Current machine

| Resource | Observed | Implication |
|---|---|---|
| GPU | NVIDIA GeForce RTX 2050, 4096 MiB VRAM | Suitable for testing a single modest detector if CUDA-enabled PyTorch is configured. The reference's multi-model pipeline targets an RTX 4060 with 8 GB; do not assume it fits here. |
| RAM | About 15.4 GiB | Stream decoded frames and JSON instead of loading a half into memory. Implemented. |
| Free C: storage | About 15.3 GiB before setup; 13.3 GiB after installation | Enough for small tests. A half needs room for the upload, framework temporary spool, annotated output and model/runtime files. Move STORAGE_ROOT to a larger drive for repeated match analysis. |
| Python | 3.12 available; default command is 3.14 | Use the project .venv based on Python 3.12. npm scripts now select it. |
| Model file | Existing local Soccer-Analytics/models/best.pt, about 48 MiB | Must load successfully and expose suitable classes. Presence alone does not establish quality, licensing or deployment availability. |

## Local workflow readiness

No hard infrastructure blocker remains for short local uploads. The browser upload, actual model inference, H.264 playback, track seeking and JSON serving have been verified. Constraints for larger videos remain below.

1. **Runtime installation resolved.** The initial environment had OpenCV/FastAPI under Python 3.12 but no PyTorch or Ultralytics. The .venv now contains PyTorch 2.14.1+cpu and Ultralytics 8.4.178. The football weights load with ball, goalkeeper, player and referee classes. `/capabilities` reports package/weight blockers on future installations. See PROGRESS.md for real inference evidence.
2. **GPU acceleration requires a compatible CUDA PyTorch build.** A visible NVIDIA GPU is not proof of torch.cuda availability. CPU processing is installed and verified; torch.cuda.is_available() is false. Start with CPU correctness, then use the official PyTorch installer for a compatible CUDA build and verify it before setting YOLO_DEVICE=0. The 30-second real sample took 117.2 seconds on CPU. Processing speed is not promised for a 45-minute half.
3. **Usable source footage and model evaluation.** Use a clear elevated wide view, preferably a short 720p/1080p MP4 first. Small-ball detection and tracking across broadcast cuts remain quality constraints. A successful sample run is not a measured accuracy score. In the initial five-second test, the ball was detected in only 1 of 25 sampled frames, so ball-dependent football statistics would currently be unreliable.

No paid cloud service, OpenAI API key, Redis, Postgres or external football-data subscription is required for this local slice.

## What blocks the fuller analytics product

| Capability | Missing prerequisite |
|---|---|
| Reliable ball tracking | Representative annotated football data and evaluated specialist weights; the reference notebook explicitly reports weak small-ball recall. |
| Metres, speed, distance, tactical heatmaps | Pitch keypoint detection/calibration and camera-motion handling, with validation against ground truth. Pixel positions alone do not support these metrics. |
| Teams and named players | Uniform classification, roster data, jersey OCR/ReID, identity confidence and human correction. A track ID is temporary. |
| Passes, shots, tackles and goals | Temporal event models/rules using ball, player and geometry evidence, plus category-specific evaluation and review. The current detector does not implement these. |
| xG and scouting/season statistics | Approved event records or a licensed external data source and versioned metric definitions. |
| Injury/workload analysis | Actual GPS/workload imports and appropriate validated analysis. Outside this initial scope. |
| TactiVision parity | Access to, or development of, its specialist models and training/evaluation data. The portfolio describes an implementation; it does not supply that stack in the linked tracking repository. |

## If you want to access it remotely

The current build is a local single-user application. Public hosting has additional blockers:

- A long-running Python service with ffmpeg and the CV runtime. A static Vercel frontend alone cannot process videos. The Dockerfile includes the CV dependencies but a full container deployment is not yet verified.
- Provision trusted model weights separately, point YOLO_MODEL_PATH at them, and settle model/data/software licensing before distributing the service. The references are not bundled into the app.
- Mount a persistent volume for SQLite, source videos and artifacts; store weights in a provisioned model directory. Set STORAGE_ROOT to the volume path.
- Keep one API worker for this MVP. Multi-process or multi-instance deployment needs an external durable job queue and shared/object storage; the in-process executor is intentionally limited to one process.
- Add authentication, authorization, upload quotas and retention before exposing private match videos. Current media URLs are not access-controlled.
- Check reverse-proxy upload size/timeouts for approximately 45-minute files. Raise limits to match MAX_UPLOAD_MB, or introduce resumable direct-to-object-storage uploads.

No hosting purchase or account connection was performed. Choose hosting after measuring actual footage throughput and disk use.

## Implementation order

1. Local video validation, persistent jobs, actual detection/tracking, playback and JSON export. This is the current slice.
2. Benchmark a representative 45-minute half; record processing time, peak RAM/VRAM, disk use and decode coverage. This acceptance test remains outstanding until representative footage is run.
3. Add calibration and team association, measure ball visibility and tracking continuity.
4. Add a small, evaluated event category and human review/approval before deriving football statistics.
5. Add hosted access and larger-scale infrastructure only when needed.

References: [TactiVision](https://mehdihch.github.io/PortfolioMCH/#/projects/tactivision-football-analytics), [tracking repository](https://github.com/Darkmyter/Football-Players-Tracking), [Ultralytics tracking API](https://docs.ultralytics.com/modes/track/), [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Hardware figures are observations of this workstation, not minimum supported specifications.
