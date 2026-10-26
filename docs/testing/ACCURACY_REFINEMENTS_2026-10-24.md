# Accuracy refinements and provisional analytics — 24 October 2026

All five source videos and 3,512 exported frames were preserved. Primary model detections from the original cloud runs were filtered and re-tracked across every frame. New secondary COCO sports-ball inference ran at up to 2fps where primary confidence was below 0.3. This is not a fresh full-frame primary inference benchmark. Source SHA256 and full exported decode counts were verified.

## Before and after

| Clip | Player tracks before → after | Mean visible player detections before → after | Ball candidate coverage before → after | Selected ball coverage |
|---|---:|---:|---:|---:|
| input_vid | 1135 → 704 | 34.78 → 13.14 | 46.48% → 48.47% | 31.34% |
| untitled_design | 1268 → 448 | 47.15 → 20.26 | 58.35% → 59.14% | 22.40% |
| dfl_test_1 | 167 → 50 | 21.31 → 19.35 | 39.33% → 39.20% | 30.80% |
| dfl_test_10 | 343 → 160 | 28.60 → 21.46 | 72.27% → 70.13% | 56.67% |
| dfl_test_11 | 244 → 48 | 23.04 → 20.24 | 55.07% → 52.93% | 39.60% |

Fewer detections or track IDs do not establish precision, recall or correct identity. Ball candidate coverage is not accuracy. Selected-ball coverage intentionally leaves ambiguous/missing intervals unknown. These clips and their 5-second snapshots were used for tuning, not an independent held-out benchmark.

## Implemented changes

- Human confidence 0.45, IoU suppression across human classes at 0.40, and conservative containment suppression of aligned body fragments before tracking.
- Largest plausible grass-region geometry rejects off-field detections, with fail-open behavior when a field mask is unreliable. This can still exclude legitimate aerial/out-of-play balls; validate on reviewed footage.
- Hard-cut reset and per-scene track IDs prevent identities being joined across detected camera cuts. Gradual transitions and tracking identity switches remain possible.
- At most one observed ball candidate is selected using confidence/margin and short-gap motion continuity. No positions are interpolated.
- Optional secondary sports-ball model at 640px and up to 2fps recovers some missing candidates. A visually checked example at frame120 of input_vid shows the primary model missed the white ball and the secondary model detected it. This is an example, not a measured recall result.
- Anonymous per-scene jersey-colour groups, image-relative shape, near-ball observation seconds/share, and explicitly unreviewed possible transfer candidates. These are not verified teams, possession, completed passes or calibrated workload.

## Analytics and downloads

Each enhanced run contains metrics/report JSON plus CSV tables for temporary tracks, anonymous kit groups, timeline and review candidates. Empty candidate tables mean no heuristic candidate was produced; they do not establish zero real passes. Known kit-proximity support is shown explicitly because it may cover only a small fraction of a clip.

Local downloads are under ignored `data/qa/accuracy-v2/`: `enhanced-videos-and-analytics.zip`, `comparison.csv`, `comparison.json` and per-run artifacts. Original clips are in `data/qa/kaggle-2026-10-24/original-videos.zip`. Enhanced hosted Library entries have `-enhanced.mp4` labels.

The new worker and frontend were deployed. No additional Modal GPU inference jobs were submitted beyond the existing five daily test jobs; refinements and sampled secondary inference were performed locally. The new worker CUDA secondary path remains to be exercised on a future allowed job. A separate local GPU runtime installation was attempted but not completed; the CPU runtime remains in use.

## Remaining gates

Review ball boxes and player identities on independent footage to measure precision, recall and ID switches. Track counts are still high on the first two clips. Jersey groups can confuse colours or split after cuts, camera movement contaminates image shape/movement, and possession, actual passes, shots, goals, xG and physical distance/speed still require validated supporting models/data.
