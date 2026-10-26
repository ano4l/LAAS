# SoccerTrack formation and collective team shape — 26 October 2026

## Reference and adaptation

Requested source: [Chao Zhuang, SoccerTrack: Collective Dynamics Analysis](https://www.kaggle.com/code/chaozhuang/soccertrack-collective-dynamics-analysis), notebook version 21, listed Apache 2.0. The public Kaggle pull API returned its complete source; the notebook was inspected, not executed. Local reference, metadata and SHA256 are in `data/qa/soccertrack-2026-10-26/`.

The notebook describes collective spatial dynamics rather than recognising named tactical formations. Its relevant methods are team centroids, sample-covariance activity/characteristic ellipses, ellipse area and unit-vector movement alignment. Our independent implementation adapts those calculations to observed detector footpoints. It does not reuse the notebook's fixed 11-player ordering, hard-coded pitch drawing, gap interpolation or assumed frame timing. No timeout, possession, aggression or 4-3-3/4-4-2 labels are inferred from these signals.

`metrics.video_insights` is now `video-insights-3` and contains `collective_dynamics` with schema `soccertrack-dynamics-1`. Old real results backfill from their complete stored tracking when selected; new analyses calculate it during processing. Existing reviewed events, saved team/stoppage context and temporal possession remain separate.

## Calculations

- A supported instantaneous group contains at least three uniquely tracked, assigned players in the same scene, with confidence at least 0.45 and finite in-image footpoints. Duplicate identities are excluded. This is a visible subset, not a verified team roster; grouped goalkeepers may be included.
- The centroid is the arithmetic mean of the observed points. Covariance uses `ddof=1`. Ellipse radii are the square roots of the ordered covariance eigenvalues; angle is the major eigenvector's pixel-image direction. One-standard-deviation ellipse area is `pi * sqrt(lambda_major * lambda_minor)`, divided by image area for display. Zero/collinear spread is valid. This is neither a confidence region nor occupied pitch territory.
- One-second summaries weight accepted frames by their actual source-time overlap. Snapshot positions/ellipse come from a real frame closest to the window centre; they are not average player positions. A boundary frame can support adjacent time bins, while its exact observation timestamp remains disclosed. Unsupported seconds remain gaps. Full-source group coverage counts supported time against the complete clip.
- Per-track activity ellipses require at least eight observed positions and 0.5 observed seconds; IDs remain scene/kit scoped. At most 80 are returned per group, with total count disclosed. Camera movement and fragmented identities affect these areas.
- Camera-image alignment is the magnitude of the mean of supported mover unit vectors, between 0 and 1. Continuous 0.4–0.6-second track intervals and at least three sufficiently moving players are required. Stationary/noisy motion, missing observations, cuts and kit switches cannot create alignment support. Weighted means use only eligible alignment time; no supported movers means null, not zero. Camera panning can make alignment high, so it is not reported as tactical coordination or accuracy.

SVG diagrams use the original pixel aspect ratio and observed footpoints, with no pitch lines, metric distances, fabricated off-camera players or interpolated ball positions. Source-linked JSON/CSV includes summaries, windows, snapshot ellipses/player points and temporary-track activity. CSV text is formula-safe.

## Actual full-clip recalculation

All five fresh GPU tracking outputs from 25 October were recalculated locally on CPU. No new model inference or GPU attempt was submitted. Their 3,512 frames, hashes and full annotated videos are unchanged. Ten anonymous scene/kit groups provide 249 supported one-second group windows and 258 temporary-track activity ellipses.

| Full source | Kit A supported source time | Kit B supported source time | Mean visible players A / B | Group windows A / B |
| --- | ---: | ---: | ---: | ---: |
| dfl_test_10 · 30s | 71.87% | 98.80% | 6.006 / 8.310 | 30 / 30 |
| dfl_test_11 · 30s | 72.93% | 98.80% | 4.307 / 9.289 | 30 / 30 |
| input_vid · 31.37s | 71.71% | 2.39% | 6.228 / 3.111 | 27 / 8 |
| untitled_design · 16.967s | 98.04% | 97.64% | 6.878 / 8.217 | 17 / 17 |
| dfl_test_1 · 30s | 96.53% | 98.80% | 5.670 / 8.603 | 30 / 30 |

These are support/visibility diagnostics, not formation accuracy. A group window may contain only a fraction of a second of evidence. The dashboard discloses supported seconds; a window count alone does not establish continuous support. `input_vid` kit B has only 0.7499 supported seconds and no movement alignment. Missing players are not filled to make a complete formation.

## Named tactical formations

Named formations remain explicitly unavailable. The cited notebook is not a trained formation classifier. Reliable tactical lines/roles require pitch coordinates (camera calibration), sufficient full-team coverage, goalkeeper/role separation, team/identity continuity and validation against reviewed formation periods. The available five clips are panning broadcast views; no trusted fixed-camera calibration or formation labels were supplied. The existing calibration CLI retains its source-hash, held-out landmark and camera-stability gates. No assumed 105×68-m pitch is imposed on uploaded video.

## Verification

The full backend suite passed **85 tests**. New tests check analytic covariance/area, translation and image-scale units, collinear/zero spread, full-source missing-time denominators, real snapshot times, movement alignment and missing-track reset, duplicate identities, confidence gates, scene separation, partial-second time weights and full-tracking schema integration. Controlled fixtures are calculation tests, not detected events or measured formation accuracy. Evidence: `backend-tests.log`, `focused-tests.log`, `results.json` in the evidence folder.

Both Modal API and GPU-worker images were deployed with the new calculation layer. No new GPU job was submitted; read-only hosted checks and frontend verification are recorded after completion below.

Read-only hosted checks passed for all five fresh entries and the complete sample: schema v3, matching source hashes/full analysed frame counts/durations, finite ordered ellipse radii, bounded source coverage/alignment, metrics HTTP 200 and video byte ranges HTTP 206. Review/context revisions remained 0, approved event totals remained 0 and the job-ID set was unchanged. The sample remains 30 seconds / 750 frames. These checks cover saved-data backfill and delivery; the new worker image has not received a fresh inference job in this task. Proof: `hosted-verification.json`.

The frontend production build passed (1,587 modules). Implementer QA checked all five actual sources, source/group/time switching, native dfl_test_10 seeking to 29.84 seconds while paused, input_vid kit B's sparse/unknown seconds, source-bound JSON/CSV and 375px layout without page overflow, duplicate IDs or broken heading labels. Browser-only controlled fixtures covered missing legacy data, empty groups and valid zero-area/null-alignment states; supported zero-area points have explicit baseline markers rather than false positive-area bars. Activity truncation discloses returned versus total rows. No real user records were mutated. Scoped accessibility scans found zero violations; desktop SVG text contrast required a manual check, and mobile had no incomplete checks. Automated checks do not certify physical-device behavior or complete accessibility. See `frontend-qa.md` and screenshots in the evidence folder.

Independent same-provider UI evaluation passed with no blocking revisions (a different-provider evaluator was unavailable). It checked all five real sources, 1920×1080 ellipse geometry, native seeking at 29.84s while paused, input_vid kit B's actual sparse/unknown/null states and 9.207s snapshot, JSON/CSV provenance, 1440/768/375px layouts and isolated legacy/empty/zero-area cases. There were no page-overflow, duplicate-ID or broken-heading-label failures. Isolated hooks were cleared and the test browser closed. An optional future phone diagram zoom was noted; numeric snapshot positions remain available beside the compact mobile graphic. The report is retained in the design session's temporary directory; no actual review/context records or GPU jobs were created.

Vercel deployment `dpl_4T2jCBfupW2x4tHWLf42bkGK8txS` is live at https://laas-football.vercel.app, assets `index-BzsyCYq7.js` / `index-sD3nrjy6.css`. Authenticated production browser checks switched all five sources into Analytics, retaining exactly one visible bound shape panel, full-duration scrub maximum and correct source video. The dfl_test_10 Watch button sought to the actual 29.84s snapshot in its 30s video and kept it paused. Actual input_vid kit B retained 8 supported bars, no snapshot at second 0, null alignment and the full 31.37s denominator. At 390px the page had no overflow, duplicate IDs or broken heading labels. The published bundle contains the hosted API/new shape feature and no workspace access key; browser errors were empty. Proofs: `production-verification.json`, `production-seek.json`, `production-partial-group.json`, `production-mobile.json`, `production-bundle-scan.json`. Final actual diagram screenshots: `production-diagram-desktop.png`, `production-diagram-mobile.png`.

Download `data/qa/soccertrack-2026-10-26/soccertrack-shape-analytics.zip` for all five source-bound JSON/CSV outputs, reference metadata and verification summaries. The individual new CSV files also retain run ID, source hash/name, camera-image units and unreviewed/unavailable-formation status. This package contains calculated observations, not independent formation labels or accuracy scores.
