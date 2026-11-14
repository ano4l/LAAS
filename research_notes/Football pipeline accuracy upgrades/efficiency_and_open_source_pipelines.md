# Efficiency of the GPU video-analytics pipeline and open-source football analytics projects (as of Nov 2026)

Reading guide: "Cited Findings" are things read from a source during this research. "Inferences" are my own arithmetic or engineering judgment and are NOT sourced; treat them as hypotheses to benchmark. Vendor numbers are flagged as such. Research budget was limited, so several sub-questions are listed honestly under "Gaps" rather than answered from memory.

## 1. TensorRT / ONNX Runtime / OpenVINO export of the Ultralytics detector

### Takeaway
On an L4, TensorRT FP16 is the cheapest reliable win: Ultralytics' own 2026 measurements show 1.35x (YOLO26x) to 4.8x (YOLO26n) lower latency than PyTorch FP16 at batch 1 / 640 px with no mAP change, while INT8 costs 3.4-7.6 mAP points on COCO and only pays off on large models (up to 1.67x over FP16 for YOLO26x on L4). No published L4/T4 measurement at 1280 px was found; it must be benchmarked locally.

### Cited Findings
- Measured (Ultralytics, vendor but with method stated: batch 1, 640 px, COCO val2017, warm-up excluded) latency on **L4**: YOLO26n PyTorch FP16 4.98 ms, ONNX FP16 2.04 ms, TensorRT FP16 1.03 ms, TensorRT INT8 1.09 ms; YOLO26x PyTorch FP16 7.76 ms, ONNX FP16 9.80 ms, TensorRT FP16 5.73 ms, TensorRT INT8 3.43 ms. — [Ultralytics blog: ONNX vs TensorRT](https://www.ultralytics.com/blog/onnx-vs-tensorrt)
- Across 19 NVIDIA GPUs, TensorRT FP16 vs ONNX FP16 median speedup: n 2.55x (1.79-3.28), s 2.19x, m 1.85x, l 2.03x, x 1.88x. TensorRT INT8 vs TensorRT FP16: n 0.90x (INT8 slower on 17 of 19 GPUs), s 1.04x, m 1.20x, l 1.08x, x 1.19x (max 1.67x, on L4). — [Ultralytics blog: ONNX vs TensorRT](https://www.ultralytics.com/blog/onnx-vs-tensorrt)
- Accuracy: TensorRT FP16 is within 0.001 mAP50-95 of ONNX FP16 and within 0.003 of PyTorch FP16. INT8 (calibrated on COCO128 via NVIDIA ModelOpt) loses 0.034-0.076 mAP50-95 (n -0.040, s -0.035, m -0.049, l -0.034, x -0.076). — [Ultralytics blog: ONNX vs TensorRT](https://www.ultralytics.com/blog/onnx-vs-tensorrt)
- The same blog did not test T4, did not report engine build times, and did not name the ONNX Runtime execution provider. — [Ultralytics blog: ONNX vs TensorRT](https://www.ultralytics.com/blog/onnx-vs-tensorrt)
- Older Ultralytics doc benchmark (YOLOv8n, ultralytics 8.2.4, TensorRT 8.6, A100, batch 8, 640): FP32 0.52 ms, FP16 0.34 ms, INT8 0.28 ms per image; COCO mAP50-95 0.37 for FP32/FP16 and 0.33 for INT8. The docs page has no T4 numbers. — [Ultralytics TensorRT docs](https://docs.ultralytics.com/integrations/tensorrt)
- Current export arguments: `format='engine'`, `quantize=16` (FP16) or `quantize=8` (INT8) which replace the deprecated `half` / `int8` flags; `dynamic`, `batch` (max batch for export and calibration), `nms` (None raw output, True embeds NMS, False uses the NMS-free head), `workspace`, `data` and `fraction` for INT8 calibration. — [Ultralytics TensorRT docs](https://docs.ultralytics.com/integrations/tensorrt)
- Pitfalls stated in the docs: engines are tuned to the build GPU and are not portable (build on the deployment GPU with matching TensorRT/CUDA); INT8 calibration differs per device so calibrate on the deployment device; NVIDIA suggests at least 500 representative calibration images; small calibration batches give poor scale estimates; INT8 lowers mAP and shifts confidence scores so the threshold should be re-chosen from the INT8 model's own F1 curve; with `dynamic=True` the first inferences and sizes different from the export size are slower; dynamic inputs span 32 px to 2x the export `imgsz`; batches larger than the export `batch` are rejected; a too-large `workspace` produces `UNSUPPORTED_STATE`. — [Ultralytics TensorRT docs](https://docs.ultralytics.com/integrations/tensorrt)
- TensorRT 11 removed the `IInt8Calibrator` interface; Ultralytics uses NVIDIA ModelOpt for INT8/FP16 there. — [Ultralytics TensorRT docs](https://docs.ultralytics.com/integrations/tensorrt)
- A community report in the docs comments: an RTX A6000 user saw TensorRT FP32 no faster than `.pt`; maintainer advice was FP16 plus warm-up. — [Ultralytics TensorRT docs](https://docs.ultralytics.com/integrations/tensorrt)
- Ultralytics v8.4.157 release notes claim up to 20% faster TensorRT inference for FP16 and INT8 engines relative to earlier builds (vendor claim, seen via search summary only). — [newreleases.io: ultralytics v8.4.157](https://newreleases.io/project/github/ultralytics/ultralytics/release/v8.4.157)
- Forum anecdote: about 2.25x over PyTorch FP16 described as typical for TensorRT FP16 on a GTX 1650 Ti Mobile class GPU (relevant to the RTX 2050 class; anecdotal). — [Ultralytics community thread](https://community.ultralytics.com/t/how-is-the-up-to-5x-speedup-with-tensorrt-achived/1432)
- Ultralytics' recommendation matrix: NVIDIA GPU -> TensorRT FP16 first, INT8 only on larger models after calibrating on your own data; CPU / mixed fleets -> ONNX Runtime; Intel hardware -> OpenVINO; a fixed shape gives TensorRT more room to optimise than dynamic shapes. — [Ultralytics blog: ONNX vs TensorRT](https://www.ultralytics.com/blog/onnx-vs-tensorrt)
- Modal Volumes are a documented pattern for persisting compilation artifacts across container restarts (vLLM / JIT kernel caches in Modal examples); Volume storage is $0.09/GiB-month with 1 TiB/month free. No Modal doc specific to TensorRT engines was found. — [Modal pricing](https://modal.com/pricing); [Modal memory snapshot guide](https://modal.com/docs/guide/memory-snapshots)

### Inferences
- The PyTorch-to-TensorRT gain on L4 is model-size dependent: 4.98/1.03 = 4.8x for nano but only 7.76/5.73 = 1.35x for x (FP16), 2.26x for x with INT8. A football detector at 1280 px is convolution-bound like the x case, so expect roughly 1.3-2x from FP16, not the "up to 5x" headline; the nano figure mostly reflects PyTorch per-call overhead.
- Rough scaling estimate (unsourced): 1280 px has 4x the pixels of 640, so YOLO26x-class TensorRT FP16 on L4 would be about 23 ms/frame (about 43 fps) at batch 1, and an m-class model roughly 10-12 ms. This must be measured with `yolo benchmark ... imgsz=1280 device=0`.
- Because engines are GPU- and TensorRT-version-specific, the practical Modal pattern is: build the engine once inside an L4 container (not at image build on a different GPU, and not on the RTX 2050), write it to a Volume keyed by `{weights hash}_{imgsz}_{batch}_{precision}_{trt version}_{gpu name}`, and load it on later cold starts. The RTX 2050 needs its own separately built engine.
- For batched offline processing, a fixed-shape, fixed-batch engine (for example batch 8 at the exact letterboxed size, such as 1280x736 for 16:9) is preferable to `dynamic=True`; rectangular export avoids wasting about 42% of pixels on padding relative to 1280x1280.
- INT8 is risky for the ball class (a few pixels wide); the COCO-level 3-8 point mAP loss is likely concentrated in small objects. Prefer FP16 and reduce cost elsewhere.
- NMS-free heads (YOLO26 `nms=False`, DETR family) remove the CPU-side NMS step that otherwise is excluded from quoted TensorRT latencies.

### Gaps
- No published, measured YOLO TensorRT numbers at 1280 px on L4 or T4, and no batch>1 throughput numbers on those GPUs, were found.
- Engine build time on L4 (FP16 or INT8) was not found in any source.
- ONNX Runtime TensorRT-EP vs CUDA-EP comparison and OpenVINO CPU numbers for the football model were not researched beyond Ultralytics' qualitative recommendation.
- Whether a 4 GB RTX 2050 can build a 1280 px batch>1 engine for an x-size model is unverified.

## 2. Detector alternatives to YOLOv8/11 (accuracy, latency, licence)

### Takeaway
At equal T4 latency, RF-DETR (Apache-2.0 for N/S/M/L), D-FINE and LW-DETR (Apache-2.0) beat YOLO11 by 5-9 COCO AP and match or beat YOLO26 in Roboflow's re-evaluation; YOLO26 is the best Ultralytics option and is NMS-free, but all Ultralytics code and trained weights are AGPL-3.0 unless an Enterprise licence is bought. The published comparisons come from Roboflow (an interested party) and from Ultralytics, and none report small-object AP.

### Cited Findings
- Roboflow's benchmark (COCO val2017, T4, TensorRT 10.4, FP16, batch 1, models re-evaluated in-house) — AP50:95 / latency / licence:
  - RF-DETR N 48.4 / 2.3 ms (384 px), S 53.0 / 3.5 ms (512), M 54.7 / 4.4 ms (576), L 56.5 / 6.8 ms (704): Apache-2.0. RF-DETR XL 58.6 / 11.5 ms and 2XL 60.1 / 17.2 ms: PML 1.0 (not Apache).
  - YOLO11 n 37.4 / 2.5, s 44.4 / 3.2, m 48.6 / 5.1, l 49.9 / 6.5, x 50.9 / 10.5: AGPL-3.0.
  - YOLO26 n 40.3 / 1.7, s 47.7 / 2.6, m 52.5 / 4.4, l 54.1 / 5.7, x 56.9 / 9.6: AGPL-3.0.
  - LW-DETR T 42.9 / 1.9, S 48.0 / 2.6, M 52.6 / 4.4, L 56.1 / 6.9, X 58.3 / 13.0: Apache-2.0.
  - D-FINE N 42.7 / 2.1, S 50.6 / 3.5, M 55.0 / 5.4, L 57.2 / 7.5, X 59.3 / 11.5: Apache-2.0.
  - The page has no RT-DETR rows, no AP-small column, and does not say whether NMS time is included for YOLO11. — [RF-DETR benchmarks](https://rfdetr.roboflow.com/develop/learn/benchmarks/)
- Ultralytics' own YOLO26 table (COCO, 640, T4 TensorRT10, NMS-free head): n 40.9 mAP / 1.7 ms, s 48.6 / 2.5, m 53.1 / 4.7, l 55.0 / 6.2, x 57.5 / 11.8; end-to-end (NMS-free) mAP is 40.1 / 47.8 / 52.5 / 54.4 / 56.9. Params 2.4M to 55.7M. — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26)
- YOLO26 design points: DFL removed, NMS-free inference via `nms=False`, Progressive Loss plus STAL (Small-Target-Aware Label assignment), MuSGD optimiser; claim of up to 43% faster CPU ONNX inference for YOLO26n vs YOLO11n (vendor claim); described as released January 2026; AGPL-3.0 or Enterprise. — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26)
- Conflict: Ultralytics lists YOLO26x at 11.8 ms on T4, Roboflow measures 9.6 ms; AP differs by 0.6 because Roboflow quotes the end-to-end figure. — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26); [RF-DETR benchmarks](https://rfdetr.roboflow.com/develop/learn/benchmarks/)
- Conflict: one Roboflow blog comparison states D-FINE at about 78 ms on T4 with 55.8 AP, contradicting the 5.4-7.5 ms in the benchmark table; treat the 78 ms as an error. — [Roboflow: RF-DETR vs alternatives](https://blog.roboflow.com/rf-detr-vs-alternatives/); [RF-DETR benchmarks](https://rfdetr.roboflow.com/develop/learn/benchmarks/)
- RF-DETR licensing: the `rfdetr` package and Apache-designated weights are Apache-2.0; the `rfdetr_plus` extension and the XL / 2XL detection models are under PML 1.0, and commercial self-hosting of PML models needs a Roboflow Enterprise add-on. — [RF-DETR benchmarks](https://rfdetr.roboflow.com/develop/learn/benchmarks/)
- Ultralytics licensing position (vendor page): an Enterprise licence is required for "SaaS platforms, APIs, or cloud systems that use YOLO behind the scenes"; all Ultralytics YOLO trained models are AGPL-3.0 by default, covering training code and the models produced with it. — [Ultralytics licence page](https://www.ultralytics.com/license)
- Roboflow's (competitor) reading: AGPL-3.0's network clause can require releasing the source of an application that serves an AGPL model over a network; custom-trained versions of AGPL models remain AGPL. — [Roboflow: Why AGPL-3.0 is a risk](https://blog.roboflow.com/why-agpl-3-0-is-a-risk-for-computer-vision-teams/)
- Evidence of practitioner choice in football: the SoccerNet GSR 2025 winner used YOLO-X (Apache-2.0 family) and third place used YOLOv12 for people plus a custom RT-DETR for field lines. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182)
- The roboflow `trackers` evaluation uses YOLOX detections for its MOT17 / SportsMOT / DanceTrack tables. — [Roboflow trackers results](https://trackers.roboflow.com/latest/evaluations/results/)

### Inferences
- For a hosted commercial product, the existing Ultralytics pipeline (code plus fine-tuned `.pt`, and the Ultralytics BoT-SORT/ByteTrack implementations in `ultralytics/trackers`) is AGPL-encumbered; the options are (a) buy the Enterprise licence, (b) open-source the service under AGPL, or (c) migrate to an Apache stack: RF-DETR (N-L) or D-FINE for detection plus the roboflow `trackers` / `supervision` libraries or original ByteTrack (MIT) for tracking. No independent legal opinion was found; both sources are interested parties.
- RF-DETR's published operating points use small inputs (384-704 px). Football needs high input resolution for the ball and far-side players, and DETR-style models with DINOv2 backbones scale worse than CNNs with resolution (attention cost), so the COCO latency advantage may not transfer to 1280 px. This needs a head-to-head on football frames (player/ball recall, not COCO AP).
- YOLO26 is the lowest-friction upgrade from 8.4.x (same API, NMS-free, STAL for small targets) if the AGPL question is resolved by licence purchase.

### Gaps
- No AP-small or football-specific (ball recall) comparison of YOLO26 vs RF-DETR vs D-FINE was found.
- RT-DETRv2/v3, DEIM and YOLOv12 T4 numbers were not retrieved from primary sources.
- Licence of the newest DEIM / D-FINE repos was taken from Roboflow's table (Apache-2.0 for D-FINE), not re-verified in the repos.

## 3. GPU video decode / encode and full-pipeline frameworks

### Takeaway
One third-party A100 benchmark shows moving decode + preprocess to the GPU raised a YOLOv8n pipeline from about 140 fps (OpenCV CPU) to 383 fps (DALI), 405-427 fps (PyNvVideoCodec 2.1) and 451-459 fps (PyNvVideoCodec + CV-CUDA); the dominant saving was CPU preprocessing (2.47 s to about 0.1 s per 1024 frames), not decode itself. No L4-specific decode numbers were found.

### Cited Findings
- Benchmark setup: A100-SXM4-40GB, PyTorch 2.6, YOLOv8n FP16 at 640x640, 1024 frames per run, 3 warm-up + 10 timed runs. Throughput at batch 16/32/64/128: OpenCV CPU 140.2 / 141.9 / 144.4 / 148.8 fps; FFmpeg+DALI 382.8 / 387.7 / 391.3 / 396.7; PyNvVideoCodec 2.1 405.0 / 414.0 / 427.3 / 420.4; PyNvVC + CV-CUDA 451.1 / 455.0 / 459.3 / failed at 128. — [Yash Jain: Beyond DALI benchmark](https://yashjain14.github.io/blogs/pynvvideocodec-vs-dali-vs-opencv-benchmark.html)
- Per-stage time for 1024 frames at batch 16: OpenCV decode 1752 ms, preprocess 2471 ms, inference 3065 ms; PyNvVideoCodec decode 1497 ms, preprocess 107 ms, inference 910 ms; DALI decode 1644 / preprocess 107 / inference 910; CV-CUDA decode 1444 / preprocess 86 / inference 728. GPU preprocessing was 23-29x faster than CPU; the author attributes OpenCV's slower inference to per-frame CPU handoffs. Source video resolution is not stated. — [Yash Jain: Beyond DALI benchmark](https://yashjain14.github.io/blogs/pynvvideocodec-vs-dali-vs-opencv-benchmark.html)
- NVIDIA's PyNvVideoCodec performance note includes an example of 1920x1080, GOP 30, 30 s clip, 3 NVDEC instances at 158.4 fps (context of the measurement was only seen in a search snippet). NVIDIA states PyNvVideoCodec 2.0 performance is on par with the native Video Codec SDK. — [NVIDIA PyNvVideoCodec performance note](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/latest/application-note.html); [NVIDIA blog: PyNvVideoCodec 2.0](https://developer.nvidia.com/blog/whats-new-in-pynvvideocodec-2-0-for-python-gpu-accelerated-video-processing)
- TorchCodec supports CUDA decoding (FFmpeg 4-8); the PyTorch team's benchmark (A100, 5 NVDEC decoders) positions it for decoding many videos in a training loader and says it is "not as competitive when there is no seeking", i.e. plain sequential decode from the start; high-resolution video gains most from CUDA decode + GPU transforms. No fps numbers were exposed in text. — [PyTorch blog: TorchCodec](https://pytorch.org/blog/torchcodec/)
- Savant (insight-platform) is an Apache-2.0 Python/YAML framework on top of DeepStream: 0.5.x production line on DeepStream 7.0, 0.6.x development line on a customised DeepStream 7.1 with TensorRT 10.9 (marked not recommended for production); supports custom Python pre/post-processing, OpenCV-CUDA, CuPy, PyTorch integration, file/RTSP/Kafka adapters; 869 stars, 1,020 commits; README gives no fps figures (links a performance dashboard). — [Savant GitHub](https://github.com/insight-platform/Savant)
- DeepStream community data points (uncontrolled forum reports): DeepStream-Yolo table about 110 fps vs about 97 fps for Ultralytics without display; a Jetson Orin sample went from about 30 fps with display to about 170 fps without; YOLOv5s on Orin Nano about 50 fps FP32 vs about 98 fps FP16. — [NVIDIA forum: Improved DeepStream for YOLO](https://forums.developer.nvidia.com/t/improved-deepstream-for-yolo-models/203368.md); [NVIDIA forum: Orin display thread](https://forums.developer.nvidia.com/t/deepstream-python-sample-apps-can-just-run-at-30fps-when-showing-display-on-orin/233088); [NVIDIA forum: YOLOv5s benchmark](https://forums.developer.nvidia.com/t/yolov5s-model-performance-testing-benchmark/329949)

### Inferences
- In the current pipeline the GPU is almost certainly not the bottleneck once TensorRT is in place: CPU `cv2.VideoCapture` decode of 1080p, CPU letterbox/normalise, CPU overlay drawing and CPU H.264 encode all run serially on a Modal container's few cores. The benchmark above shows that even with a nano model, CPU decode+preprocess capped throughput at about 140 fps on a large host; with 1080p-to-1280 resize and overlay rendering it will be far lower.
- Recommended minimal change (unsourced engineering judgment): decode with PyNvVideoCodec (NVDEC, frames stay as CUDA tensors via DLPack), do resize/letterbox/normalise in torch on the GPU, feed fixed-size batches to the TensorRT engine; only download detections to the CPU. This removes the host-memory round trip of full frames. If frames are needed on the CPU for sparse optical flow (GMC), download a downscaled grayscale copy only.
- If server-side rendering is kept, pipe frames to `ffmpeg -c:v h264_nvenc` (or PyNvVideoCodec's encoder) rather than libx264/OpenCV `VideoWriter`; better, drop server rendering entirely (section 6).
- DeepStream/Savant give the highest ceiling (zero-copy NVDEC -> nvinfer -> nvtracker -> NVENC) but impose GStreamer/DeepStream containers, their own tracker set (NvDCF/NvSORT/DeepSORT variants rather than BoT-SORT), and custom parsers; for an offline batch job on Modal with custom football logic, PyNvVideoCodec + TensorRT in plain Python is a much smaller migration with most of the gain.
- On the RTX 2050 (4 GB) the same NVDEC + TensorRT FP16 path works (laptop GPUs have NVDEC/NVENC), but batch size and 1280 px x-size models will be memory-constrained.

### Gaps
- No measured NVDEC / NVENC throughput specific to the L4 (decode fps for 1080p/4K H.264/HEVC, h264_nvenc fps) was retrieved; no decord benchmark was found.
- No football pipeline was found that reports before/after fps from switching to GPU decode.
- DALI video reader behaviour on long VFR broadcast files and PyNvVideoCodec licence terms were not checked.

## 4. Pipeline architecture: batching, frame skipping, chunked parallelism, tracker/GMC cost

### Takeaway
The published evidence here is thin: the strongest sourced facts are that leading football systems are offline, multi-pass designs (online tracklets, then split/merge and majority voting), that BoT-SORT's default GMC is sparse optical flow at 2x downscale, and that tuned tracker configs on SoccerNet all keep sparseOptFlow CMC enabled. No source gives per-frame GMC timings or validates frame-skipping or chunk-stitching for football, so those remain design hypotheses to test.

### Cited Findings
- Ultralytics' GMC class supports `orb`, `sift`, `ecc`, `sparseOptFlow`, `none`; default is `sparseOptFlow` with downscale factor 2 (downscale exists for computational efficiency). The ORB path uses FAST keypoints + ORB descriptors + brute-force Hamming matching. — [Ultralytics gmc.py (mirror)](https://huggingface.co/spaces/AkashDataScience/FastSAM/blob/main/ultralytics/tracker/utils/gmc.py); [Ultralytics GMC reference](https://docs.ultralytics.com/ja/reference/trackers/utils/gmc/)
- The BoT-SORT paper names ECC maximisation and ORB matching as registration options without reporting their speed. — [BoT-SORT paper](https://arxiv.org/pdf/2206.14651.pdf)
- Roboflow `trackers` docs describe sparseOptFlow as fast and robust for most footage and note CMC can be disabled on static cameras for maximum speed. — [Roboflow trackers: OC-SORT](https://trackers.roboflow.com/latest/trackers/ocsort/); [Roboflow trackers: McByte](https://trackers.roboflow.com/latest/trackers/mcbyte/)
- Tracker accuracy on SoccerNet-tracking with oracle detections (trackers v2.6/2.7), default -> tuned HOTA: SORT 81.6 -> 84.2; ByteTrack 84.0 -> 84.0; OC-SORT 78.4 -> 82.9; C-BIoU 82.6 -> 85.7; BoT-SORT 84.5 -> 85.0; BoT-SORT+ReID 84.3 -> 88.4 (IDF1 79.1 -> 84.4); McByte 85.0 (default only). On SportsMOT (YOLOX detections): ByteTrack 73.0/73.3, BoT-SORT 73.8/74.1, BoT-SORT+ReID 75.5/77.8, McByte 76.5. — [Roboflow trackers results](https://trackers.roboflow.com/latest/evaluations/results/)
- All tuned BoT-SORT configs in that evaluation keep `enable_cmc: true` with `cmc_method: sparseOptFlow`; C-BIoU is positioned for cases where CMC is "unavailable or harmful" and runs without it. The page gives no fps numbers. — [Roboflow trackers results](https://trackers.roboflow.com/latest/evaluations/results/)
- Offline multi-pass is how the best systems work: SoccerNet GSR 2025 winner KIST-GSR (GS-HOTA 63.90) uses YOLO-X + Deep-EIoU with OSNet ReID, then tracklet split-and-merge (split on identity-prediction consistency, merge on ReID similarity under spatio-temporal and identity constraints), majority voting within tracklets and filtering of unreliable trajectories. Third place (lianyou, 62.76) uses Farneback optical flow to improve Kalman prediction, CLIP-ReIdent features, and a 6-layer transformer over tracklet features. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182)
- Eagle exposes exactly the "run expensive steps less often" knobs: `--fps` for output granularity and `num_homography` / `num_keypoint_detection` for how many times per second homography and keypoint detection run. — [nreHieW/Eagle](https://github.com/nreHieW/Eagle)
- A community pipeline (AgabaEmbedded/Soccer-Analysis) explicitly runs three offline passes and resolves identities by per-track majority vote rather than deciding live. — [AgabaEmbedded/Soccer-Analysis](https://github.com/AgabaEmbedded/Soccer-Analysis)
- Modal plan limits relevant to fan-out: Starter allows 10 concurrent GPUs (100 containers), Team 50 GPUs; functions are preemptible by default and non-preemptible execution costs 3x. — [Modal pricing](https://modal.com/pricing)
- Modal GPU memory snapshots (alpha/experimental): enabled with `enable_memory_snapshot=True` plus `experimental_options={"enable_gpu_snapshot": True}`; snapshot includes GPU memory (weights), CUDA contexts, and `torch.compile` artifacts; measured cold starts: Parakeet about 20 s -> about 2 s, ViT with torch.compile 8.5 s -> 2.25 s, vLLM Qwen2.5-0.5B 45 s -> 5 s (vendor measurements); restore needs compatible GPU hardware; changes to Volumes do not update snapshots and deleting Volume files used at restore breaks restore. — [Modal blog: GPU memory snapshots](https://modal.com/blog/gpu-mem-snapshots); [Modal memory snapshot guide](https://modal.com/docs/guide/memory-snapshots)

### Inferences
(All items below are unsourced engineering judgment.)
- Decouple stages with bounded queues: decode thread (NVDEC) -> batcher -> TensorRT inference (batch 8-16) -> single-threaded tracker consuming detections strictly in frame order. Batched inference is compatible with sequential trackers because the tracker only needs per-frame detections in order, not synchronous inference. GMC needs consecutive frames, so compute it in the decode/preprocess worker on a downscaled grayscale copy and pass the affine matrix along with the detections.
- GMC cost: sparse optical flow at 2x downscale on 1080p is typically a few ms per frame on one CPU core, which becomes the bottleneck above roughly 100-200 fps. Cheaper options: (1) downscale by 4 instead of 2; (2) compute GMC every Nth frame and compose/interpolate; (3) reuse the pitch-keypoint homography already computed for calibration to derive inter-frame camera motion (free if calibration runs per frame); (4) disable CMC for static tactical/panoramic cameras (SoccerTrack-style footage). ECC is generally the slowest option; do not switch to it for speed.
- Detection at reduced fps: for 50 fps sources, detecting every 2nd frame (25 Hz) and interpolating boxes is low-risk for players (inter-frame motion is small relative to box size); for 25 fps sources, skipping to 12.5 Hz degrades association during camera pans and for the ball, which should be detected on every frame or recovered by a dedicated ball tracker/trajectory fit. Output positions at 10-25 Hz are standard for tracking-data consumers, so rendering needs can be met by client-side interpolation.
- ROI: cropping to the pitch bounding region (from a pitch mask or the calibration) and using rectangular inference typically removes 20-40% of pixels (stands, scoreboard) on broadcast footage; for static wide cameras, fixed tiles are an alternative to 1280 full-frame.
- Two-pass design: pass 1 online detection + cheap tracker (ByteTrack/BoT-SORT without ReID) emitting short, high-purity tracklets plus per-detection crops/embeddings sampled sparsely (for example 1-2 crops per second per track); pass 2 CPU-only global association (tracklet merge with team/jersey/ReID constraints, majority voting, interpolation, smoothing). This mirrors the SoccerNet winners and moves ReID/jersey cost from per-frame to per-tracklet.
- Chunked parallelism on Modal: split a 45-min half at keyframes into N segments (for example 9 x 5 min) with a 2-5 s overlap, `Function.map`/`starmap` over segments on separate L4 containers, then stitch by matching tracklets in the overlap window (IoU / pitch-coordinate distance with Hungarian assignment) and let the global pass-2 association resolve remaining identities. Cost is roughly unchanged (same GPU-seconds, plus N cold starts); wall-clock drops close to N-fold. Starter plan's 10-GPU limit bounds N. Byte-range/segment access needs the source in a seekable store (Volume or object storage) and segment-accurate cutting (`ffmpeg -ss` before `-i` with keyframe alignment, or decode-side seeking).
- Cold start: bake weights into the image or a Volume, load the prebuilt TensorRT engine from the Volume, and try GPU memory snapshots to skip CUDA/TensorRT initialisation; because snapshots are alpha, keep a non-snapshot fallback.

### Gaps
- No measured per-frame timings for sparseOptFlow vs ECC vs ORB GMC were found.
- No published study quantifying HOTA/IDF1 loss from detection frame-skipping on football footage was found.
- No published recipe or measurement for cross-chunk track stitching in a football pipeline was found.
- Modal `.map` / `@modal.batched` / `@modal.concurrent` documentation was not fetched in this session; API details should be checked in current Modal docs.
- Ultralytics BoT-SORT per-frame tracker CPU cost (with/without ReID) is unmeasured here.

## 5. Modal economics, GPU choice and alternatives

### Takeaway
Modal list prices (Nov 2026): T4 $0.59/h, L4 $0.80/h, A10 $1.10/h, L40S $1.95/h, billed per second and preemptible by default; at these prices a full half costs cents to low tens of cents on an L4 if the pipeline sustains 50-150 fps, so throughput per GPU-second (not GPU choice) is the main lever.

### Cited Findings
- Modal per-second prices: T4 $0.000164, L4 $0.000222, A10 $0.000306, L40S $0.000542, A100-40GB $0.000583, A100-80GB $0.000694, H100 $0.001097. CPU $0.0000131 per physical core-second (2 vCPU equivalent); memory $0.00000222 per GiB-second; Volumes $0.09/GiB-month with 1 TiB free. — [Modal pricing](https://modal.com/pricing)
- Starter plan: $30/month free credits, 10 GPU concurrency, 100 containers; Team: $100/month credits, 50 GPU concurrency. Non-preemptible execution costs 3x; region selection costs 1.15x-1.75x. — [Modal pricing](https://modal.com/pricing)
- A third-party page lists A10G at $0.000292/s, conflicting with the $0.000306 on Modal's own page. — [Beam: Modal pricing explained](https://www.beam.cloud/blog/modal-pricing-explained); contradicted by [Modal pricing](https://modal.com/pricing)
- L4 measured detector latencies (640 px, batch 1): YOLO26n 1.03 ms and YOLO26x 5.73 ms TensorRT FP16; 3.43 ms for YOLO26x INT8. — [Ultralytics blog: ONNX vs TensorRT](https://www.ultralytics.com/blog/onnx-vs-tensorrt)
- An arXiv study comparing T4 and L4 across FP32/FP16/INT8 exists but covers ResNet models, not detectors (seen in search results only). — [arXiv 2604.14552](https://arxiv.org/pdf/2604.14552)

### Inferences
- Cost arithmetic (mine): a 45-min half at 25 fps is 67,500 frames; at 50 fps, 135,000. On an L4 at $0.000222/s: 25 fps end-to-end -> 2,700 s -> $0.60; 60 fps -> 1,125 s -> $0.25; 150 fps -> 450 s -> $0.10 (for 67.5k frames; double for 50 fps sources unless detection runs on alternate frames). Add about $0.0000131 x cores x seconds for CPU (8 physical cores for 1,125 s is about $0.12), so CPU-heavy decode/render can cost as much as the GPU; GPU decode and no server render reduce both.
- GPU choice: T4 is 26% cheaper than L4 but generally more than 26% slower for FP16 detectors and has older NVENC/NVDEC, so L4 is normally the better price/performance; A10 costs 38% more than L4 and L40S 2.4x more, which only pays if the pipeline is GPU-bound after decode/preprocess are fixed (not the case today). Parallel L4 containers scale wall-clock linearly at constant cost, which is a better lever than a bigger single GPU. No source measuring YOLO price/performance across these four GPUs was found, so this is judgment.
- Alternatives such as Runpod serverless were not researched in this session.

### Gaps
- No sourced L4 vs T4 vs A10G vs L40S YOLO throughput comparison was found (the Ultralytics blog covers L4, 4090, A100, H100 only in the extracted table).
- Runpod / other serverless pricing not retrieved.
- Modal's current cold-start times for an L4 container with a multi-GB image were not found.

## 6. Output efficiency (tracking format, rendering, streaming)

### Takeaway
No sources were retrieved for this sub-question within the research budget; the items below are engineering judgment only and should be labelled as such in the final report.

### Cited Findings
- The only directly relevant sourced data points: Eagle writes tracking data in pitch coordinates (UEFA 105 x 68 m) plus an `annotated.mp4` used only for error checking, i.e. the data, not the rendered video, is the product; and a 2026 community extension of the SoccerNet GSR stack ships a local React replay UI over the tracking output. — [nreHieW/Eagle](https://github.com/nreHieW/Eagle); [hualian0927/soccer](https://github.com/hualian0927/soccer)

### Inferences
(Unsourced engineering judgment.)
- Skip server-side overlay rendering and re-encoding altogether: serve the original video (remuxed to fragmented MP4/HLS without re-encoding if the codec is browser-compatible) and draw boxes/IDs/radar on a `<canvas>` layered over the `<video>` element in React, synchronised with `requestVideoFrameCallback` (frame-accurate `mediaTime`). This removes the two most CPU-expensive stages (draw + H.264 encode) and makes overlays toggleable. If a downloadable annotated MP4 is needed, render it on demand with h264_nvenc.
- Replace the monolithic `tracking.json` with columnar data: one row per (frame, track) with int/float32 columns (frame, track_id, class, team, x1,y1,x2,y2 as int16 pixels, pitch x/y as float32, conf as uint8) in Parquet with zstd; typically 10-30x smaller than pretty JSON. 67.5k frames x about 25 objects is about 1.7M rows, i.e. a few MB to low tens of MB in Parquet versus hundreds of MB of JSON. For the browser, serve time-chunked files (for example 10-60 s segments as Parquet/Arrow IPC via `parquet-wasm`/`apache-arrow`, or gzip/brotli-compressed compact binary) fetched lazily as playback proceeds, mirroring HLS segmenting.
- Persist intermediate per-chunk detections (pre-tracking) so tracking/association can be re-run without re-inferencing.
- Vercel is a poor place to proxy large media; serve video and tracking chunks directly from object storage/CDN with range requests.

### Gaps
- No measured size/latency comparisons (JSON vs Parquet vs Arrow for tracking data), no browser overlay performance data and no HLS packaging guidance were retrieved; all of the above is uncited.

## 7. Realistic end-to-end throughput reported for football pipelines

### Takeaway
None of the open-source football pipelines or SoccerNet challenge write-ups examined report end-to-end fps or hardware; throughput for detection+tracking+ReID+calibration on T4/L4/A10 is effectively unpublished and must be measured.

### Cited Findings
- The SoccerNet 2025 challenge report gives no runtime or hardware details for the top GSR methods. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182)
- sn-gamestate's README gives no speed or GPU requirement (only "lower batch sizes if memory issues occur"). — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- roboflow/sports and Eagle READMEs contain no speed benchmarks; Eagle says a CUDA GPU or Apple Metal is "highly recommended" and falls back to a medium detector / ONNX on CPU. — [roboflow/sports](https://github.com/roboflow/sports); [nreHieW/Eagle](https://github.com/nreHieW/Eagle)
- The 2025 winner uses LLaMA-3.2-Vision for jersey number / role / colour recognition, and third place uses ViT-L/14 CLIP, i.e. top accuracy currently relies on heavy VLM components. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182)

### Inferences
- Research-grade GSR stacks (TrackLab baseline with per-detection ReID, MMOCR jersey reading, TVCalib optimisation; VLM-based winners) are almost certainly far below real time on a single L4 (low single-digit fps is plausible) and are unsuitable as-is for a 45-minute half at cents per match; their value is in the offline association logic and calibration models, which can be applied per-tracklet rather than per-frame.
- A detector-bound budget on L4 (estimate from section 1): 1280 px m/l-class TensorRT FP16 at roughly 60-100 fps batched, x-class about 40-50 fps; with GPU decode and no rendering, total pipeline throughput should approach the detector figure, giving about 12-30 minutes for a 25 fps half on one L4, or about 2-4 minutes wall-clock with 8-10 parallel chunks.

### Gaps
- No published end-to-end fps for football detection+tracking+ReID+calibration on T4/L4/A10 was found in any source consulted.

## 8. Survey of open-source football CV pipelines

### Takeaway
There is no production-ready, permissively licensed, maintained end-to-end project to adopt wholesale: roboflow/sports (MIT, 5.5k stars) is the most practical base to borrow from (pitch keypoints, team clustering, radar), while SoccerNet sn-gamestate + TrackLab (GPL-3.0, updated May 2026) is the reference for full game-state reconstruction and for evaluation (GS-HOTA) but is research code with heavy, slow components and a copyleft licence.

### Cited Findings
- **roboflow/sports**: MIT; 5.5k stars, 675 forks, 34 commits on main, 32 open issues, 21 open PRs; no Python package published (install from source). Links Roboflow Universe datasets: football-players-detection-3zvbc, football-ball-detection-rejhg, football-field-detection-f07vi (pitch keypoints). README lists unsolved challenges: ball tracking, jersey number reading, player tracking through occlusion, player re-identification, camera calibration. No speed benchmarks. — [roboflow/sports](https://github.com/roboflow/sports)
- A fork (Football PitchVision) describes the pitch model as detecting 32 keypoints and downloads about 400 MB of pretrained models. — [DesusLove/Football-PitchVision](https://github.com/DesusLove/Football-PitchVision)
- **SoccerNet/sn-gamestate** (+ TrackingLaboratory/tracklab): full GSR baseline: YOLOv11 detection, PRTReid (BPBreID also referenced) ReID, StrongSORT-style tracking (inferred from tracker-state filenames), TVCalib calibration by default with PnLCalib and "No Bells, Just Whistles" as alternatives (added July 2024), MMOCR jersey numbers, team affiliation; code GPL-3.0; TrackLab auto-downloads dataset and all weights on first run; news: uv install + new visualisation tool (May 2025), TrackLab 1.3.24 fixing GS-HOTA evaluation (1 May 2026); environment references Python 3.9 / PyTorch 1.13.1 / CUDA 11.7. — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- GSR benchmark context: baseline GS-HOTA 29.01; 2025 leaderboard KIST-GSR 63.90, Constructor.Tech 63.81, lianyou 62.76, Playbox & MIXI 61.64, tyler_durden 57.08; i.e. the public baseline is less than half the accuracy of the top systems. GS-HOTA multiplies localisation similarity by identity similarity. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182); [SoccerNet GSR paper](https://arxiv.org/pdf/2404.11335)
- Top-method recipes (useful to borrow): KIST-GSR: YOLO-X, Deep-EIoU + OSNet ReID, tracklet split/merge, multi-frame keypoint model for pitch mapping, LLaMA-3.2-Vision for jersey/role/colour, jersey-colour clustering with mean x-position for left/right, majority voting. lianyou: YOLOv12 persons, custom RT-DETR for 27 field-line classes, radial distortion + PnLCalib with pan/tilt/roll/focal parametrisation and temporal smoothing, IOF-Tracker with Farneback flow, CLIP-ReIdent (0.5 cosine threshold), K-means teams, fine-tuned ViT-L/14 CLIP for numbers. The report does not state whether their code is released. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182)
- 2024 winner Constructor Tech's method is described in arXiv 2504.06357 (GS-HOTA 63.81). — [arXiv 2504.06357 listing](https://www.emergentmind.com/papers/2504.06357)
- **hualian0927/soccer** (2026 newcomer, 0 stars, 3 commits): a local tactical-analysis extension "built on SoccerNetGSR and Broadcast2Pitch" with YOLOX detection, ByteTrack/EIoU trackers, tracklet refinement, CLIP appearance re-check, court segmentation + calibration optimisation, CLIP fine-tune for jersey/role, an RF-DETR + SAHI + TCN ball module, event candidates, Chinese reports and a React replay demo; README states accuracy is not validated; no project licence stated; weights downloaded by script. A search summary described it as the "official implementation" and 2025 challenge winner, but the repository page itself makes no such claim. — [hualian0927/soccer](https://github.com/hualian0927/soccer)
- **nreHieW/Eagle**: broadcast video -> tracking data (player and ball coordinates in 105 x 68 pitch space); YOLOv8 detectors (`detector_medium` 640, `detector_large`, `detector_large_hd` trained at 960 px, default, better ball recall), HRNet keypoint model for homography, BoT-SORT; team assignment is purely heuristic and is the most common failure; weaker with irregular angles/occlusion; needs at least 4 keypoints for homography; weights via `get_weights.sh` (PyTorch + ONNX); 58 stars, 34 commits; no licence section visible. — [nreHieW/Eagle](https://github.com/nreHieW/Eagle)
- **abdullahtarek/football_analysis**: tutorial project: YOLO (v5 weights on Google Drive) detection/tracking, K-means shirt-colour team assignment, optical-flow camera movement, perspective transform to metres, speed/distance, ball possession assignment; about 1.0k stars, 344 forks, only 3 commits; no LICENSE file visible in the listing (unconfirmed); unpinned requirements. — [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis)
- **Other community pipelines** (small, unvalidated): adityapgupta/Sports-Analysis "SoccArt" (YOLOv10 + ByteTrack + SigLIP colour clustering + homography minimap, sliding-window interpolation); Moiz005/SoccerVision (sn-gamestate variant with YOLOv11, StrongSORT+BPBreID, NBJW calibration; jersey OCR skipped on Colab due to MMOCR/mmcv incompatibility); Khoagoodkid/soccer-player-detection (YOLOv5 + ByteTrack + ResNet18 number classifier with temporal voting); AgabaEmbedded/Soccer-Analysis (three offline passes, YOLOv11 digit reader at 88.5% mAP@50, majority vote); JooZef315/football-tracking-data-from-TV-broadcast (MIT). — [adityapgupta/Sports-Analysis](https://github.com/adityapgupta/Sports-Analysis); [Moiz005/SoccerVision](https://github.com/Moiz005/SoccerVision-Player-Tracking-3D-Reconstruction); [Khoagoodkid/soccer-player-detection](https://github.com/Khoagoodkid/soccer-player-detection); [AgabaEmbedded/Soccer-Analysis](https://github.com/AgabaEmbedded/Soccer-Analysis); [JooZef315 repo](https://github.com/JooZef315/football-tracking-data-from-TV-broadcast)
- **BroadTrack** (playbox-dev; Playbox & MIXI placed 4th in GSR 2025): broadcast camera tracking (calibration over time) for soccer, arXiv 2412.01721. — [playbox-dev/BroadTrack](https://github.com/playbox-dev/BroadTrack); [BroadTrack paper](https://arxiv.org/abs/2412.01721)
- **OpenSTARLab**: not a video pipeline; a downstream analytics framework with PreProcessing (Unified and Integrated Event Data format), Event (deep event prediction), Spatial-Temporal Event label tool, and RLearn (deep RL) packages under github.com/open-starlab. Code licence not confirmed (an aggregator lists Apache-2.0 for the preprocessing package). — [OpenSTARLab paper](https://arxiv.org/html/2502.02785v2)
- **roboflow/trackers** library provides SORT, ByteTrack, OC-SORT, C-BIoU, BoT-SORT (with ReID from v2.7), McByte with published SoccerNet/SportsMOT evaluations (numbers in section 4). — [Roboflow trackers results](https://trackers.roboflow.com/latest/evaluations/results/)
- VLM dependence: top GSR entries rely on VLMs (LLaMA-3.2-Vision, CLIP ViT-L/14) for jersey/role recognition. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182)

### Inferences
- Adopt wholesale: none. The existing custom pipeline (already on Modal, already using SoccerTrack-style team shape per the latest commit) is a better base than any surveyed repo for a hosted product.
- Borrow from roboflow/sports (MIT, safe to copy): pitch keypoint model + homography/radar code, SigLIP-embedding team clustering approach, ball-track cleaning utilities. Caveat: its example models are Ultralytics YOLOv8 (AGPL) even though the repo code is MIT.
- Borrow ideas (not code, because of GPL-3.0) from sn-gamestate/TrackLab and the challenge reports: tracklet-level jersey voting, split/merge refinement, PnLCalib / NBJW calibration with temporal smoothing, team left/right assignment from mean x-position. Use TrackLab + GS-HOTA as an offline evaluation harness (internal use of GPL tooling does not affect the product).
- Borrow from Eagle: reduced-rate homography/keypoint scheduling and the tracking-data output schema; licence must be checked before copying code.
- abdullahtarek/football_analysis is a teaching project (no licence, 3 commits); nothing to adopt beyond what the current pipeline already does.
- OpenSTARLab is relevant only downstream (event/tracking data modelling), as would be format tooling such as kloppy (not researched here).

### Gaps
- Not researched within budget: Darkmyter/Football-Players-Tracking, SkalskiP's standalone examples, tryolabs/norfair soccer demo, PySport/kloppy; their status and licences are unverified in this session.
- Licences of Eagle, TrackLab itself, BroadTrack, and the OpenSTARLab repos were not confirmed from LICENSE files.
- Whether KIST-GSR ("Broadcast2Pitch"?) or Constructor.Tech released official code/weights is unconfirmed; the mapping of the name Broadcast2Pitch to the KIST-GSR entry is an unverified search-summary claim.
- Last-commit dates for roboflow/sports, Eagle and abdullahtarek/football_analysis were not visible.
- No independent quality evaluation (HOTA/GS-HOTA) exists for roboflow/sports, Eagle or the community pipelines.

## 9. Public pretrained football weights and datasets, and licences

### Takeaway
Usable open data exists (Roboflow Universe football datasets, SoccerNet GSR/tracking, SoccerTrack v2), but licence terms were only partially verifiable: Roboflow forks show CC BY 4.0, SoccerTrack v2's licence was not found, and any weights trained with Ultralytics inherit AGPL-3.0 regardless of dataset licence.

### Cited Findings
- roboflow/sports links three football datasets on Roboflow Universe: football-players-detection-3zvbc (classes include ball, goalkeeper, player, referee per derivative projects), football-ball-detection-rejhg, football-field-detection-f07vi; the repo README does not state their licences. — [roboflow/sports](https://github.com/roboflow/sports)
- Derivative/fork projects of football-players-detection on Universe display CC BY 4.0; the original project page's licence field was not retrieved. — [Roboflow Universe fork (Kick Data)](https://universe.roboflow.com/kick-data/football-players-detection-3zvbc-7ocfe-jzrw7); [Original project page](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- A Hugging Face re-upload of a YOLOv8 football-field keypoint model is tagged MIT by its uploader and requires a Roboflow API key; the tag reflects the uploader's choice, not necessarily the dataset's terms. — [Simon9/football-field-detection-roboflow](https://huggingface.co/Simon9/football-field-detection-roboflow)
- SoccerNet GSR dataset: version 1.3 (updated bbox-pitch annotations, May 2024), downloadable via the SoccerNet pip package (`gamestate-2024` splits) or automatically by TrackLab; baseline tracker states on Zenodo; the README states no dataset licence. — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- SoccerTrack v2: 10 full-length panoramic 4K university-level matches (about 900 minutes), per-frame GSR labels (2D pitch coordinates, jersey-based IDs, roles, teams) and Ball Action Spotting labels for 12 classes; captured with BePro Cerberus (2 matches) and BePro 3-camera stitching (8 matches); preprint Aug 2025 says release via GitHub and Hugging Face is forthcoming while the project page lists it as open access on GitHub/Google Drive (sources disagree on release status); licence not found. — [SoccerTrack v2 paper](https://arxiv.org/html/2508.01802v1); [SoccerTrack v2 project page](https://atomscott.github.io/SoccerTrack-v2/)
- SkillCorner open data: broadcast-derived tracking data for nine matches, MIT (useful as an output-format and validation reference, not video). — [SkillCorner/opendata](https://www.github.com/SkillCorner/opendata)
- Eagle ships downloadable football YOLOv8 detectors (640 and 960 px trained) and an HRNet pitch keypoint model in PyTorch and ONNX. — [nreHieW/Eagle](https://github.com/nreHieW/Eagle)
- Ultralytics states that models trained with its code are AGPL-3.0 by default. — [Ultralytics licence page](https://www.ultralytics.com/license)

### Inferences
- For a commercially clean stack: retrain on CC BY 4.0 Roboflow data (attribution required) plus own footage using an Apache-2.0 trainer (RF-DETR N-L, D-FINE, YOLOX). Publicly shared football YOLOv8 weights (Roboflow examples, Eagle) are effectively AGPL-encumbered.
- SoccerNet video data has historically required an NDA and is positioned for research; assume it cannot be used to train commercial models without clearing terms. This is not sourced in this session and must be verified.
- SoccerTrack v2 matches the existing pipeline's wide-angle use case (the repo's latest commit references SoccerTrack team shape) and is the most relevant evaluation set for static panoramic footage; confirm its licence on the Hugging Face dataset card before commercial use.

### Gaps
- Licence fields for the original Roboflow football datasets, SoccerNet GSR dataset terms (NDA/commercial use), and SoccerTrack v2 were not confirmed from primary pages.
- No inventory of Apache/MIT-licensed pretrained football detector weights (non-Ultralytics) was found.
