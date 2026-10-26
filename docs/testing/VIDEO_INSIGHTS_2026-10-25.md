# Automatic video insights, 25 October 2026

## Scope and evidence

`metrics.video_insights` (`video-insights-1`) is a new automatic calculation layer over full stored tracking observations. It is computed during fresh inference and backfilled on selecting a real completed run. It never derives analytics from the small preview, silently fills missing observations, maps anonymous kit groups to Home/Away, or writes approved football records. An incomplete legacy run remains readable without assumed insights. JSON/CSV exports preserve the source run and unreviewed status.

The approach is informed by the two-stage possession/event framework in [Vidal-Codina et al., Automatic event detection in football using tracking data](https://arxiv.org/abs/2202.00804), which separates possession inference from event inference and benchmarks against manual labels. Our implementation is a narrower, independent heuristic using image coordinates. The paper's event accuracy does not transfer to this application. A single nearby-player frame is inadequate evidence for a verified pass or possession; camera calibration, detection quality, temporal evidence and independent labels remain important.

## Implemented calculations

- Observed ball journey, with scene and continuity segment IDs; missing observations, cuts and implausible jumps split the path. Display data is bounded to 600 actual observations, with sampling stride disclosed. No interpolated ball locations are generated.
- Ball distribution in left/centre/right **camera-image** thirds, observed seconds and confidence-gated coverage. This uses a selected observed candidate with score at least 0.25, so it differs from basic all-candidate coverage. Scores are not accuracy measurements.
- Anonymous, scene-scoped kit occupancy, image-third distributions and five-second shape evolution. Each shape frame needs at least three grouped players. Width, depth, centroid, supported seconds and mean visible grouped players are calculated with actual interval weights. They are image percentages, not pitch metres, attack direction or identified formations.
- Sustained estimated ball-control spells. One ground-plausible selected ball must be near a clearly nearest tracked player foot point (within 0.65 box heights; runner-up separation at least 0.2), for at least three consecutive observations and 0.24 seconds. Ball discontinuities break spells. No unknown gaps are carried forward. Unknown-kit support remains separate from group-assigned control.
- Kit shares of **known assigned control**, alongside full-source clip shares, assigned coverage and unknown seconds. A group receiving 100% of a tiny supported interval does not establish match possession.
- Possible passes/turnovers between distinct sustained controllers, with the same scene, separated image positions, at most 1.2 seconds between spells and a fully observed, plausible continuous ball bridge. Same-track returns, near-identical positions, missing/aerial bridge observations, cuts and jumps cannot create these candidates. They remain review prompts, not approved event totals or completed passes.
- Automatic findings and timestamped bookmarks for the clearest ball window and widest supported kit-image window. Wording reports observations and coverage rather than unsupported coaching conclusions.

## Five-clip replay

Recalculation used the existing full research tracking files. This is new CPU analytics over real observations, not new neural detector inference or an accuracy benchmark. All 3,512 analysed frames contribute, including empty observations. The annotated videos and approved review records are preserved.

| Source | Duration | Eligible ball coverage | Sustained control | Kit shape windows |
| --- | ---: | ---: | ---: | ---: |
| input_vid | 31.37s | 26.43% | 0s | 12 |
| untitled_design | 16.97s | 16.31% | 0s | 8 |
| dfl_test_1 | 30s | 24.13% | 0s | 12 |
| dfl_test_10 | 30s | 48.27% | 1.16s | 12 |
| dfl_test_11 | 30s | 33.33% | 0.88s | 12 |

There are 1,075 eligible observed ball points and 56 kit-shape windows across these clips. No possible pass/turnover candidate met the strict temporal evidence rules. This does not mean there were zero real passes or turnovers. These clips do **not** establish reliable automated match possession or event accuracy. The limiting evidence is ball/identity continuity and ground-plane context, rather than an absent counting formula. Named player performance, calibrated physical measures, xG and historical comparisons remain outside this calculation layer.

Replay: `node scripts/python.mjs scripts/replay_video_insights.py data/qa/research-2026-10-24/outputs --summary data/qa/video-insights-2026-10-25/replay-summary.json`. Each output includes `video-insights.json`, `estimated-control.csv`, `automatic-candidates.csv`, `kit-shape-timeline.csv` and `ball-path.csv`, in addition to existing metrics/video artifacts.

## Validation and hosting

Tests cover known/unknown denominators, sustained control, fly-bys, competing players, ball ambiguity, confidence/ground gates, same-track returns, identity splits, cuts, missing bridge evidence, jumps, positive controlled pass/turnover scenarios, shape support, sampling/tail intervals, bounded paths, idempotent legacy backfill, preservation of approved records, and protected selected-result/signed-metrics delivery without inference. Controlled positive test events are fixtures, not claimed events from real footage.

The complete backend suite passed: **64 tests**, zero failures/errors. Evidence: `data/qa/video-insights-2026-10-25/backend-tests.xml` and `backend-tests.txt`. The existing FastAPI startup deprecation warnings remain. Frontend TypeScript/Vite production build passed (1,584 modules). Initial UI checks confirmed 15-second shape seeks, 29.92-second ball-position seeks, exact selected-run exports, explicit CSV unknown/source-duration columns, scene selection, optional-data states, and no 375px overflow. A scoped insights accessibility scan reported zero violations/incomplete checks and 29 passes. These are browser/automated checks, not real-device certification.

The Modal web API and GPU worker images were deployed. Read-only hosted checks selected all five research runs, verified source hashes/frame counts/durations, signed metrics HTTP 200 and signed video range HTTP 206, retained zero approved events, and confirmed the job-ID set did not change. The sample still provides 30 seconds / 750 frames and now has the new schema. These checks did not submit GPU inference or alter compute/billing limits. Evidence: `data/qa/video-insights-2026-10-25/hosted-verification.json`.

Independent same-provider frontend evaluation passed with no mandatory revisions (a different-provider evaluator was unavailable). Four successive actual clip switches retained one correctly bound insights panel, legacy measurements panel, track explorer and video; there were no duplicate IDs, broken heading references, document overflow or browser errors. Actual shape seeking at 20 seconds and keyboard ball-slider seeking at 29.92 seconds worked. JSON/CSV retained the selected source and unreviewed status. Controlled browser-only fixtures covered second scenes, empty/unknown/missing insights, optional bookmarks, and a 13-entry candidate queue with type/search filtering, 6/6/1 pagination and evidence seeking. Native browser functions were restored and actual footage reopened afterward. The whole-page automated accessibility scan returned zero violations, zero incomplete checks and 47 passes. Desktop, tablet, 375px and 390px layouts were inspected.

Screenshots/proofs are under `data/qa/video-insights-2026-10-25/`: `ball-journey-dfl10-desktop.png`, `kit-shape-dfl10-desktop.png`, `insights-desktop-dfl10.png`, `insights-mobile-input.png` and `eval1-*`. Files labelled CONTROLLED show temporary UI fixtures rather than real football results. The independent report is retained in the design session's temporary directory.

## Production frontend verification

Vercel deployment `dpl_4ygFJWMegsqKRJqmXrZbeRafy9Rj` is live at https://laas-football.vercel.app, publishing `index-BOP5pPu-.js` and `index-5sKEowXH.css`. Production TypeScript/Vite build passed (1,584 modules). The authenticated hosted browser loaded the new insights with one compact `/library` request, and switching dfl11 to dfl10 then input_vid kept exactly one new insights panel, one legacy measurement panel and one video with matching source duration. The dfl10 chart sought to 15 seconds in its 30-second video; input_vid reached readyState 4 at 31.37 seconds with 753 analysed frames and correctly unknown control throughout. The 390x844 layout had no document overflow, duplicate IDs or broken heading labels. Published bundle inspection confirmed the hosted API/new insights and absence of the workspace access key. No GPU work or review writes occurred.

Production screenshots: `production-analytics-desktop.png`, `production-ball-journey-desktop.png`, `production-analytics-mobile.png`. Proofs: `production-verification.json`, `production-seek-verification.json`, `production-bundle-scan.json`. These checks establish integration and the displayed calculations, not detector/event accuracy or physical-device certification.
