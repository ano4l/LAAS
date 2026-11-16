# Modal GPU setup

The frontend and API are now hosted as well; see [HOSTING.md](HOSTING.md) for the current public URL, access-key location, architecture and verified journey. The local CLI/app integration below remains available.

## Current status, 21 October 2026

Modal SDK 1.6.2 is installed and the `laas` profile is authenticated to the `ano4l` workspace. After the user added a payment method, `laas-football` deployed successfully. Deployment dashboard: https://modal.com/apps/ano4l/main/deployed/laas-football

A real five-second clip completed on NVIDIA L4: all 125 frames analysed at 1280px, 11.24 processing seconds, 70 frames with ball candidates. Run `dc79e0e7079f4a2da4a588d9fc3891b2` was downloaded into `data/storage/outputs`, with every artifact hash and the source-video hash verified. Processing time excludes upload, queue, cold start, and download overhead. Candidate coverage is not accuracy. No labelled precision/recall evaluation is implied.

The local `.env` now selects `ANALYSIS_COMPUTE=modal`; the API is running on `http://127.0.0.1:8000`. Tests are isolated from the configured cloud provider to prevent accidental GPU calls. Local tests: 19 passed; frontend build passed. The full 30-second browser-upload test is job `e21891b92d474ee0881c614a8d423291`; check its persisted status for completion. The function may wait in Modal's GPU queue even after a previous call succeeded. No billing limits or top-up settings were changed by the agent.

The local upload app can send clips to an authenticated, private Modal L4 function. Results are copied back into local storage with SHA256 verification and appear in the existing analysis library. This deployment hosts GPU processing; the web UI and its SQLite database continue running on the local machine. No unauthenticated public video-upload endpoint is created.

## Setup

```powershell
node scripts/python.mjs -m pip install -e '.[modal]'
node scripts/python.mjs -m modal token new --profile laas
$env:PYTHONIOENCODING='utf-8'
node scripts/python.mjs -m modal deploy deploy/modal_app.py
node scripts/python.mjs scripts/modal_analyze.py data/qa/football-test.mp4
```

The image includes only the video worker source and selected trusted football weights. If `models/pitch-keypoints/football-pitch-detection.pt` exists locally at deploy time (or `YOLO_PITCH_MODEL_PATH` points to one), the pitch landmark model is baked in too and runs gain estimated pitch positions; otherwise the worker runs without them. Reference weights are discovered locally at build time, or set `YOLO_MODEL_PATH` for a replacement before deploying. Videos, `.env`, API tokens, and the rest of the reference repositories are not baked into the image. Modal credentials live in the user profile's `.modal.toml`, outside the repository. The server process must use the same authenticated OS account/profile.

After a successful deployment and GPU smoke test, set `ANALYSIS_COMPUTE=modal` in `.env` and restart the API. Leave `ANALYSIS_PROFILE=accurate` for every-frame / 1280px inference. Remove `ANALYSIS_FPS`, `YOLO_IMAGE_SIZE`, and `YOLO_BALL_MODEL_PATH` overrides when using Modal; the deployed model and profile are authoritative. Set `ANALYSIS_COMPUTE=local` and restart to return to local processing. Cloud failures never silently run on the CPU or generate demo results. To allow a CPU run after a cloud failure (no token, no network, clip over the limits, run limit reached, timeout), set `ANALYSIS_FALLBACK=local`: the job then runs on this machine with `FALLBACK_PROFILE` (default `balanced`), and the run records `compute_fallback` with the reason and carries a warning. A run-limit refusal still uploads the clip and starts a container before it is refused.

## Prototype usage controls

- One GPU container maximum (an L4, or an A10G when no L4 is free); no warm minimum; two-second idle window. The run records which GPU it used.
- Two CPU cores requested, 8GiB RAM, ten-minute execution timeout and three-minute startup timeout.
- No configured exception retries. Infrastructure preemption can still restart work.
- Clips limited to 120 seconds, 250MiB, up to 4K and 60fps.
- At most five admitted attempts per UTC day and twenty per UTC calendar month, persisted in `laas-football-jobs/usage.json`. The daily allowance was raised from three while keeping the monthly limit unchanged, so deployment tests leave room for real use. Failed admitted attempts count. Requests refused by this ledger can still incur container startup/validation costs.
- Ledger safety relies on one sequential deployed function, one environment and one deployment version. Do not run overlapping old/new deployments or a second app against the same ledger. This is not an atomic distributed billing limit.

These are app controls, **not an account spending cap or a guarantee of a zero bill**. Modal meters compute, model/container loading, and storage; other apps in the account also consume credits. Check current credits and billing limits in Modal before larger workloads. No payment method or automatic top-up is configured by this project.

## Adding pitch maps to runs made before the pitch model

`backfill_pitch` adds pitch calibration to finished runs without re-tracking them. It reads the stored `source.mp4`, calibrates the frames the run analysed, writes `pitch-calibration.json`, and recomputes the insights. Detections and tracks are untouched. It has no web route; run it from a trusted machine:

```powershell
node scripts/python.mjs -m modal run deploy/modal_app.py::backfill_pitch
node scripts/python.mjs -m modal run deploy/modal_app.py::backfill_pitch --request-ids JOB_ID,JOB_ID
```

Runs that already have calibration are skipped unless `--force` is passed. Runs whose source video is no longer stored are skipped. It uses a GPU (an L4, or a cheaper T4 when no L4 is free), so it costs credits, but it does not count against the analysis attempt ledger. Do not run it while an analysis is in progress.

## Storage and recovery

Inputs and outputs are staged under `/requests/REQUEST_ID` in the private Modal Volume. After all artifacts are imported and verified locally, that request directory is removed from Modal. The usage ledger remains. Failed/interrupted imports retain their directory for investigation; an interrupted local API can leave a cloud call running. Check the Modal app dashboard before retrying or deleting its request directory. Retrying may start and count a new attempt. Inspect retained files with:

```powershell
node scripts/python.mjs -m modal volume ls laas-football-jobs /requests
```

Use the Modal dashboard to stop a running call. Never reset the ledger to bypass the prototype usage allowance. Full-match processing, multiple users and a public website need a durable dispatcher, authentication and storage design beyond this first setup.

References: https://modal.com/docs/guide/volumes and https://modal.com/docs/guide/trigger-deployed-functions
