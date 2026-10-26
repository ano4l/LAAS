# Development progress

Updated 26 October 2026.

## SoccerTrack collective formation view

The requested Chao Zhuang SoccerTrack notebook version 21 was inspected and adapted into source-bound collective-shape calculations: covariance ellipses, centroids/spread, per-second changes, temporary-track activity ellipses and supported camera-image movement alignment. All five complete stored fresh outputs were recalculated on CPU, producing 249 supported group windows and 258 activity ellipses without GPU inference. No interpolation, fake full teams, pitch units or named tactical formations are claimed. The full backend suite passed 85 tests; frontend build and independent UI evaluation passed. Both Modal images were updated and Vercel deployment `dpl_4T2jCBfupW2x4tHWLf42bkGK8txS` is live; all-five-source production checks, native seeking, partial evidence and mobile layout passed. [Methods, actual support and limitations](docs/testing/SOCCERTRACK_FORMATION_2026-10-26.md).

## Fresh inference and temporal possession

All five full Kaggle clips now have fresh L4 inference (3,512 frames, 389 ROI search frames) and a temporal pass/possession layer. Short receptions, motion-supported touches and supported ball journeys replace the earlier pass-only strict-spell rule. Three partial-tracking pass proposals are exposed for review and excluded from possession flight totals. Unknown/contested time remains explicit; classified team coverage is only 0.66–11.87%, so reliable match possession is still blocked by ball/identity evidence. A separate source-bound team/stoppage editor and raw-footage annotation/scoring workspace are available. No approved events or human benchmark labels were fabricated. Both Modal apps were updated; 78 backend tests, frontend build and independent UI evaluation passed. Vercel deployment `dpl_4uP54ggKtTfVxVJkx9JdhFWQk77v` is live; production checks covered all five fresh sources, source-bound totals, full playback, candidate seeking and 390px layout. Current methods, real results and remaining blockers: [temporal possession report](docs/testing/TEMPORAL_POSSESSION_2026-10-25.md). Earlier sections below describe their corresponding older versions.

## Automatic insights from uploaded video

New `metrics.video_insights` calculates segmented observed ball journeys, image-third distributions, scene-scoped kit occupancy and five-second shape changes, supported control spells with explicit unknown time, and strictly supported possible pass/turnover review candidates. Automatic findings and video bookmarks link the calculations to footage. Fresh uploads calculate these during inference; selected older full-tracking runs backfill without another GPU job. Approved human records remain authoritative and separate.

All five full research clips were recalculated: 1,075 eligible ball observations, 56 supported kit-shape windows and 2.04 seconds of sustained control across 3,512 analysed frames. No pass/turnover candidate met the evidence rules; this does not establish zero real events. These data cannot yet support reliable full-match possession. Both Modal apps were deployed; read-only hosted checks confirmed source hashes, frame counts, full durations, metrics HTTP 200, video HTTP 206 and unchanged job IDs. The full backend suite passed 64 tests; independent UI evaluation and frontend build passed. Vercel deployment `dpl_4ygFJWMegsqKRJqmXrZbeRafy9Rj` is live, with production source switching, 15-second seeking, full-duration playback and 390px layout checks passed. Detailed formulas, results and verification: [automatic video insights](docs/testing/VIDEO_INSIGHTS_2026-10-25.md).

## Dashboard usability and visual polish

The real-video workspace now has Overview, Video review, Analytics, Library and Exports views with a coherent dark navy/mint treatment, labelled navigation, convenient clip switching and prominent full-duration video. Analytics foregrounds interactive measured coverage/image occupancy and offers a separate saved-approved-records view. The seven analytics areas and 42 requirements remain in an accessible secondary scope section. Tracks and event records have manageable search/filter/pagination controls; model/source details sit in disclosure rather than the primary flow.

Review data stays bound to its selected run and server-saved revision. Draft forms/records survive view and clip changes, while authoritative summaries and JSON exports exclude unsaved work. Background processing continues while browsing another completed clip, upload admission stays disabled while a job is active, and late upload/selection/poll responses cannot replace a newer clip choice.

Both APIs now expose compact `GET /library`. Hosted listing data fell from 28,321,960 bytes across the former two requests to 14,409 bytes (99.949% smaller; selected full analysis still loads separately). This is a response-size comparison, not a latency benchmark. The hosted selected-result route resolves that run directly, and browsing does not query provider status or backfill unrelated clips.

Validation: 50 backend tests, final frontend build and independent design evaluation passed. Repeated switching across six 17/5/30-second clips leaves exactly one correct analytics panel and track explorer; IDs/ARIA references remain unique. Video seeking, pause behavior, pagination, saved-only summaries/exports, revision conflicts, draft retention, background-job selection safety and 375/390px layouts were checked. No real QA events or GPU jobs were created. Vercel deployment `dpl_AB6CN7bBzKb125YBBwavkeYgps7m` is live; authenticated production checks passed across four research clips, with full video durations and source-bound analytics, no mobile overflow or duplicate IDs, and no workspace key in the published bundle. Detailed notes/screenshots: [dashboard refinement](docs/testing/DASHBOARD_POLISH_2026-10-25.md).

## Research implementation: association, ball search and event review

The [primary-source research report](docs/research/FOOTBALL_SYSTEM_RESEARCH_2026-10-24.md) maps ByteTrack/BoT-SORT, SAHI/TrackNet, SoccerNet reconstruction/calibration/action spotting and structured event analytics to implementation decisions. Low-confidence established-track recovery is restored; BoT-SORT sparse optical-flow camera compensation is configurable/default, with appearance ReID disabled. The secondary ball detector now has recent-ball crop search plus periodic global reacquisition. Anonymous colour grouping uses supported populations and reduced brightness weighting. Same-track reappearances can no longer generate transfer candidates.

A persisted event-review API/dashboard calculates team/player event summaries only from saved approved annotations, with explicit pass-outcome coverage, reviewed duration and known possession coverage. Drafts, validation errors and stale-revision conflicts remain distinct. A local fixed-camera calibration CLI checks held-out landmarks and camera stability before optional observed movement in metres; no real calibration was supplied, so no physical performance is claimed for the five clips.

Five complete clips were reprocessed from their existing real primary/secondary observations, preserving and decoding all 3,512 frames and verifying source hashes. Revised temporary player tracks are 476, 336, 46, 107 and 49 versus the preceding refinement's 704, 448, 50, 160 and 48. These are continuity diagnostics, not measured identity accuracy. A separate ten-frame fresh CPU inference smoke test exercised three ROI searches. Full five-clip crop inference and the updated CUDA path have not been exercised; the existing daily GPU allowance was already consumed and was not changed.

Validation: 47 automated tests passed, followed by 11 focused tests after inference-threshold/scheduling hardening. Frontend production build passed. Independent same-provider UI evaluation passed: per-run draft/form restoration, saved-only review JSON, controlled stale-save conflict/reload, 375px layout without overflow, scoped accessibility checks, and tab-wide unload protection even after navigating to a clean run or Demo. QA drafts were cleared; no invented approved football records were created.

The Modal web review API and final GPU worker image have been deployed. All five hosted research runs returned review HTTP 200, unauthorised HTTP 401 and video-range HTTP 206. An empty review persisted/reloaded at revision 1 and a stale save returned HTTP 409. The sample remains the complete 30 seconds / 750 frames. Outputs and comparison CSV/JSON: `data/qa/research-2026-10-24/`; bundle: `research-videos-and-analytics.zip`.

Frontend deployment `dpl_6jZeg5vxJBGa4enZnQsUghGBavRe` is live at https://laas-football.vercel.app. The production browser loaded the research-run Library, review panel and a full 30-second H.264 video (readyState 4, 1280×720), with no page overflow. Build-tool advisories found during deployment were patched within compatible ranges: both root/frontend npm audits now report zero known vulnerabilities, and the production build passed with Vite 7.3.7. Published HTML/JavaScript include the hosted API and review UI, with no workspace access key embedded. This does not establish general security certification or measured model accuracy.

## Accuracy refinements and provisional kit analytics

Five Kaggle clips were reprocessed across all 3,512 frames with shared human-box suppression, a 0.45 confidence threshold, plausible-field geometry, hard-cut resets, ball continuity selection and anonymous jersey-colour groups. A secondary COCO sports-ball model performed new inference at up to 2fps on weak/missing primary detections. All five enhanced videos retain their source duration/frame count and source hashes, and were published to the private hosted Library with `-enhanced.mp4` labels. This reused the primary cloud detections; it is not a new full-frame primary inference benchmark.

Temporary player tracks changed 1135→704, 1268→448, 167→50, 343→160 and 244→48 respectively. These reductions do not establish correct identity or measured detection accuracy. The secondary model recovered a visually checked ball missed by the primary model at input_vid frame120, but ball coverage remains limited. No approved passes or possession were fabricated. New analytics include per-scene anonymous kit observations, explicitly supported near-ball seconds/share, image-relative shape and unreviewed transfer candidates (none in these final five runs).

All 33 automated tests and the frontend build passed. Independent UI review passed, including mobile keyboard access and legacy data compatibility. The Vercel frontend and Modal worker were deployed; no additional cloud GPU inference attempts were submitted. The worker's new CUDA secondary-model path remains to be exercised on a future allowed job. See [the detailed refinement report](docs/testing/ACCURACY_REFINEMENTS_2026-10-24.md). Local downloads are `data/qa/accuracy-v2/enhanced-videos-and-analytics.zip` and the original video bundle under `data/qa/kaggle-2026-10-24/`.

## Measured upload analytics and complete sample

Real runs now calculate versioned `metrics.video_measurements` directly while streaming detector observations. The dashboard shows visible-player mean/max, ball candidate coverage and detection gaps, per-track observed seconds/visibility, image displacement in pixels, conservative ball-proximity candidates, time-segment coverage and a 12x8 image occupancy grid. Full tracking JSON backfills earlier completed runs; empty sampled frames count in denominators. These are unreviewed camera-relative measurements, not possession, touches, calibrated running distance, team or named-player statistics.

The reusable sample is the complete 30-second source `08fd33_4.mp4`: 750 analysed/exported frames, 23.41 mean visible player detections, ball candidates in 564/750 frames (75.2%). Existing verified Modal output is reused without another GPU inference call. Authenticated `GET /sample` and **Open sample · 30s** reopen stored results. Source hash, exported frame count/duration, signed metrics download and HTTP 206 video delivery were checked. The former five-second hosted upload also receives the new calculations. Automated suite: 26 tests passed; frontend production build passed.

## Required dashboard analytics

Both user-supplied Word documents have been reread and mapped in [docs/ANALYTICS_REQUIREMENTS.md](docs/ANALYTICS_REQUIREMENTS.md). The real dashboard now has seven requirement-driven areas and 42 requested analytics/capabilities: match, team, opponent, players/development, scouting, workload/injury indicators, and reports/exports. Module navigation and metric-name search work; missing values and their data dependencies are explicit. Competition/season/team/player/position/date/match dataset filters are visibly unavailable until validated datasets exist. This is a dashboard surface and requirements checklist, not completion of the analytics engines.

The dashboard explicitly retains the approximately 45-minute-half target and distinguishes the current 120-second hosted test limit. Approved-event analytics, human event review/approval, historical imports, stable player identity, GPS loads, calibrated physical metrics, suitable xG inputs/models, report composition and role-based access remain incomplete. No synthetic metrics are inserted into real results.

The frontend is on Vercel and the hosted API/GPU worker on Modal; actual GPU inference, browser upload, signed video playback and raw JSON download have been verified. See [deploy/HOSTING.md](deploy/HOSTING.md). The earlier 16 October notes below are historical and their cloud-deployment status is superseded by that hosting record.

## Ball detection update from the supplied repositories

- Inspected both new reference repositories. Implemented an independent raw-ball detection path based on the separation in abdullahtarek/football_analysis, keeping ByteTrack for people. No cached reference tracks, backward-filled ball coordinates or fixed sample homography are used.
- Actual five-second smoke test at 640: ball candidates retained in 14/25 sampled frames, compared with 1/25 under the earlier combined tracking path. The full updated pipeline produced valid JSON and MP4 in 18.83 seconds on CPU; a subsequent browser upload also completed with the same 14/25 candidate coverage in 30.2 seconds. Timings vary with initialization and machine load. This is candidate coverage, not measured accuracy.
- A raw-inference comparison at 1280 produced candidates in 16/25 frames; no precision or recall is claimed.
- Added separate ball/object thresholds, optional dedicated ball weights and detection provenance. Pinned Ultralytics 8.4.178 for the tested ByteTrack interface.
- Automated tests: 9 passed, including regression coverage that low-confidence ball detections survive without a confirmed tracker ID and when no player is detected.
- Prepared a GPU batch Dockerfile, full-frame/1280 cloud profile and worker CLI. Docker is unavailable locally; CUDA and cloud deployment remain unverified. See CLOUD_PLAN.md.

## Current implemented scope

The local application now accepts actual uploaded football footage and uses YOLO plus ByteTrack. Uploads cannot silently fall back to synthetic analytics. The older tactical dashboard remains available only as an explicitly labelled demo.

Implemented:

- Content validation, extension and size limits, and cleanup of rejected uploads.
- Persistent SQLite job records, one sequential worker, progress/error reporting, source-preserving retries and restart recovery from the beginning.
- Frame-by-frame video decoding, sampled inference across the full duration and streamed JSON output. The previous 450-frame inference / 900-frame export caps are removed from real uploads.
- Football weights discovered at `references/Soccer-Analytics/models/best.pt`, configurable through YOLO_MODEL_PATH. Loaded classes: ball, goalkeeper, player, referee.
- Actual ByteTrack IDs, source frame indexes, reported-frame-rate timestamps, pixel boxes, confidence and unreviewed status. Team and named-player identity remain unknown.
- Browser-compatible H.264 export, source frame rate, at most 1280 pixels on the longest side, no audio.
- Dedicated real-results screen, visible upload/processing progress and library, error/retry controls, searchable track summaries, seek-to-first-observation and artifact links.
- Same-origin production serving works as an API-backed application. Static demo mode now requires explicit VITE_DEMO_MODE=true.
- Root .env loading, a Windows/POSIX Python launcher, Docker CV dependencies and setup/infrastructure documentation.

## Verification

- `npm test`: 7 passed. Cases cover content/extension/size validation and cleanup, persisted upload/retry behavior, missing runtime rejection, no synthetic success on engine failure, restart recovery and persisted processing errors, and a controlled-detector streaming regression.
- The streaming regression covers 1,005 frames / 40.2 seconds, checks the final detection timestamp, unknown identities, valid JSON and complete output video beyond the old truncation point. It is pipeline verification, not a model-accuracy test.
- `npm run build`: passed.
- `git diff --check`: passed.
- Runtime: Python 3.12.10, PyTorch 2.14.1+cpu, Ultralytics 8.4.178. CUDA availability is false in the installed runtime.
- Browser upload of a real five-second 1920x1080 football clip completed. The detector processed 25 sampled frames and the encoder preserved all 125 source frames. Runtime: 24.83 seconds on CPU. Observations: 366 player, 42 referee, 3 goalkeeper and 1 ball. These are detections across frames, not verified unique people.
- That five-second run generated 43 temporary tracks. Ball detection appeared in only 1/25 sampled frames. No accuracy or tracking-continuity score is claimed.
- Browser playback loaded a 1280x720 H.264 video with duration 5 seconds and readyState 4. Filtering to track T34 and selecting Show in video moved playback to 0.8 seconds. Tracking JSON returned 412 records; media range requests returned HTTP 206.
- A second real upload processed the entire 30-second 1920x1080 sample: 750 decoded/exported frames and 150 inference frames, in 117.20 seconds on CPU. Observations: 2,307 player, 299 referee, 46 goalkeeper and 1 ball. It generated 191 temporary tracks; this fragmentation is a limitation, not a player count. Browser playback loaded the full 30-second video.
- Browser refresh reconnected to the running real job. Desktop and 390px viewport checks found no horizontal document overflow; at 390px the video fits in 360px. Track tables scroll within their panel.

Real smoke-test artifacts and screenshots are under ignored `data/qa` and `data/storage`; they are not committed source or a deployment.

## Outstanding acceptance and limitations

- A representative 45-minute half has not been processed or benchmarked. Full-duration streaming support is implemented, but successful half-match acceptance is not yet established.
- CUDA acceleration is not configured. The GPU is an RTX 2050 with 4 GB VRAM; the installed CPU runtime works but is slower than real time in the initial sample.
- There is no automatic football-event model, team assignment, player naming, approval workflow, calibrated pitch projection, distance/speed calculation or xG. Real results do not populate those statistics.
- Ball detection and identity continuity need representative labelled evaluation and likely model improvements before ball-dependent analytics.
- Timestamps assume the reported frame rate. Variable-frame-rate inputs need normalization before precise temporal review.
- Restart recovery reruns a job from its saved source; there are no frame-level checkpoints. Run one backend process against this storage.
- Hosting, authentication, remote private media access, object storage and a distributed queue have not been deployed or verified.

See README.md for startup and INFRASTRUCTURE.md for blockers and the staged implementation path. IMPLEMENTATION_PLAN.md records the earlier broader roadmap; its demo-era milestones are not proof of real event analytics.
