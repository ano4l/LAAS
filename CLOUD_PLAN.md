# GPU cloud plan and accuracy work

Checked 16 October 2026. Recommendation: start with a Runpod Pod with one RTX 4090 (24 GB VRAM) for repeatable GPU experiments and the first half-match benchmark. Keep the current web app and use the same streaming Python worker on that GPU. Modal is a good later choice for jobs that start on upload and scale to zero.

## Current listed prices

| Service | GPU | Listed compute rate | Use here |
|---|---|---|---|
| Runpod Pods | RTX 4090, 24 GB | USD 0.89/hour | Recommended first benchmark target |
| Runpod Pods | A40, 48 GB | USD 0.59/hour | Alternative when extra model memory is useful |
| Modal Functions | L4, 24 GB | USD 0.7992/hour for GPU alone | Managed per-job execution; CPU and memory billed separately |

Runpod rates are advertised page values, not a reserved quote; the selected cloud tier, host, region and availability need checking in the console. Storage and other charges are additional. Its standard network storage lists USD 0.07/GB/month below 1 TB, so 100 GB is approximately USD 7/month. As a compute example, ten GPU-hours at USD 0.89 are USD 8.90; that is not a price per match. A cost per half requires a real GPU benchmark.

Modal's L4 calculation is 0.000222 USD/second multiplied by 3,600. Four physical CPU cores and 16 GiB of memory at its listed resource rates would add approximately USD 0.317/hour while allocated. Container startup and warm-idle billing also matter. Its documentation allows a configured function timeout up to 24 hours; video duration and processing duration must be treated separately.

Sources: [Runpod prices](https://www.runpod.io/pricing), [Runpod persistent storage](https://docs.runpod.io/pods/storage/types), [Modal prices](https://modal.com/pricing), [Modal timeouts](https://modal.com/docs/guide/timeouts).

## Proposed first machine

- One RTX 4090, 24 GB VRAM.
- At least 32 GB host RAM and 6 vCPUs as a starting configuration, to benchmark rather than a proven minimum.
- Approximately 40 GB container disk for the CUDA environment, plus 100 GB persistent workspace for weights, source footage, outputs and review evidence.
- CUDA-enabled PyTorch and a compatible NVIDIA host driver.
- Inference at the native video frame rate up to the configured target of 30 fps, at model input size 1280. These are candidate quality settings; they increase computation and need measured evaluation.
- A persistent mount at /workspace; provision trusted football weights at /workspace/models/football.pt. The optional /workspace/models/ball.pt can be a dedicated ball detector.

Do not purchase H100 or a multi-GPU cluster for the first benchmark. A single-GPU workload is the next measurement step. GPU speed and memory allow higher resolution and more frames; they do not establish recognition accuracy.

## What was taken from the references

The inspected reference commits are:

- probablyabdullah/Football-Tracking-with-YOLOv5-Bytetrack: 44ce3a32a82c5f1ef34944608f19071e408187a7.
- abdullahtarek/football_analysis: e4799632cdf271cf57f4eb0c4d872cf1b7ab0f17.

The YOLOv5 project is a notebook using externally downloaded custom weights and older YOLOv5/ByteTrack setup. It is useful as a detector experiment, but changing version names is not evidence of greater accuracy. Its trained weight file is not included in this application.

The modular football_analysis project is the stronger architecture reference. In its tracker.py, players/referees use ByteTrack while the ball comes from raw detections. That separation is now implemented in our own adapter. The current stream never requires a confirmed player-style track ID before retaining a ball detection.

The reference also interpolates and backward-fills missing ball positions and uses fixed sample-video perspective points. Those are not validated for arbitrary uploads. We retain missing ball observations, mark raw detections with provenance and unreviewed status, and keep metric analytics unavailable until calibrated.

References: [YOLOv5 notebook project](https://github.com/probablyabdullah/Football-Tracking-with-YOLOv5-Bytetrack), [modular analysis project](https://github.com/abdullahtarek/football_analysis), [ball handling source](https://github.com/abdullahtarek/football_analysis/blob/main/trackers/tracker.py), [sample perspective source](https://github.com/abdullahtarek/football_analysis/blob/main/view_transformer/view_transformer.py).

## Measured local change

On the same five-second clip, at 5 sampled fps:

- Earlier combined tracking path: ball retained in 1 of 25 sampled frames.
- New independent raw-ball path, input size 640: ball candidates in 14 of 25 frames, 16 candidate boxes (all 14 frames also contain a candidate at confidence 0.15 or higher); complete annotated MP4 and JSON produced in 18.83 seconds on CPU.
- Standalone raw-ball inference experiment at input size 1280: candidates in 16 of 25 frames, 18 boxes. Inference took 31.12 seconds on CPU, compared with 12.58 seconds for the raw 640 experiment. These inference-only timings differ from whole-pipeline time.

Candidate coverage is not precision, recall or proof that each box is the ball. The detector weights and source clip are the same ones already available locally. No new training or independent validation set has been introduced. The current sample does not justify a claim of super-accurate analysis.

## Runnable worker preparation

- deploy/Dockerfile.gpu builds a CUDA batch-worker image with a pinned PyTorch CUDA wheel version and Ultralytics runtime.
- deploy/runpod.env.example contains the proposed full-frame / 1280 settings.
- scripts/analyze_video.py invokes the same worker on local or mounted cloud videos and reports the manifest path.

On a Docker host with an NVIDIA GPU runtime:

```bash
docker build -f deploy/Dockerfile.gpu -t football-analysis-gpu .
docker run --rm --gpus all --env-file deploy/runpod.env.example \
  -v /absolute/persistent/workspace:/workspace football-analysis-gpu \
  /workspace/videos/match-half.mp4 --output /workspace/storage/outputs
```

For a CPU check from this Windows checkout:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_video.py data/qa/football-test.mp4 --device cpu --fps 5 --image-size 640
```

The prepared image runs a batch job and does not publish an unauthenticated web API. A hosted upload service still needs job dispatch, private media access and a way to deliver completed artifacts to the web app. Those integrations have not been deployed. The current application remains local and usable.

Docker is not available in the current environment, so the GPU image has not been built here. CUDA execution and cloud throughput have not been tested. The local worker and detector tests are separate evidence and must not be treated as a GPU deployment result.

## Acceptance before calling it accurate

1. Label representative clips from the user's actual camera angles, including tiny/occluded ball, advertising, crowd, close-ups, cuts and replays. Hold evaluation clips apart from training.
2. Compare the current weights and any YOLOv5 or dedicated ball weights on exactly the same held-out clips. Record ball precision/recall/localization and errors by object size, plus player identity switches and track fragmentation.
3. Add ball-specific high-resolution/tiled detection or a trained temporal ball model as needed. Tune thresholds using validation data, preserving confidence and missing observations.
4. Calibrate the pitch per camera view, handle cuts, and validate player/team identity before calculating movement or possession.
5. Process an entire 45-minute half on the proposed GPU. Record wall time, peak RAM/VRAM, storage, interruption/recovery behavior, video decode coverage and actual cloud cost.

The external steps needed next are a funded cloud account, an available GPU/volume, provisioned weights and representative footage/annotations. No cloud resource has been created or paid for in this session.
