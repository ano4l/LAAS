# Temporal pass and possession estimates — 25 October 2026

## What changed

Real video analysis now includes `video-insights-2` with a separate `temporal-possession-1` sequence layer. This improves the earlier strict near-foot counting model without promoting model output to approved football records.

- Short stable receptions and motion-supported one-touch interactions can establish control anchors. Stable kit votes can recover a missing kit for the same tracked player in the same scene; they never merge player identities across tracks.
- Distinct-controller transitions allow up to four seconds for an observed, plausible ball journey. Scene cuts, contested interactions, other players, implausible jumps and suspected identity splits stop a transition.
- Strong transitions need at least 65% observed ball support with no individual missing gap above 0.25 seconds. Same-team supported flight can contribute to estimated possession. A same-player return never becomes a pass.
- Partial tracking proposals allow at least 35% support, three observed positions and no individual gap above 0.75 seconds. They are **review-only** and their flight time remains unknown. Cumulative missing time and the longest individual gap are both disclosed.
- Possession partitions the entire source into controlled estimates, ball in transit, bounded control gaps, contested, unknown and explicitly confirmed dead-ball intervals. No missing ball coordinates are created. The denominator retains unknown and contested time; only source-confirmed stoppages are excluded.
- Team shares disclose both classified-known and all-non-dead denominators. Anonymous kit groups remain anonymous until the user confirms names. Cuts still prevent events even when the same team is mapped across scenes.

The departure/control distinction follows the reasoning in the [ball-possession framework](https://arxiv.org/html/2202.00804v3), which also distinguishes player control, team possession and dead-ball states. Our implementation uses unvalidated image geometry, rather than that work's optical ground-coordinate data and evaluated system. Its reported accuracy does not apply here.

## Fresh full-clip inference

All five original Kaggle clips received fresh full-source CUDA inference on an NVIDIA L4, including primary detection, tracking, secondary ball reacquisition and recent-ball ROI search. This replaces the earlier exercise that mainly replayed saved detections. All **3,512 frames / 138.34 source seconds** were processed and exported; processing total was **237.38 seconds** across the jobs, excluding upload, startup and download. There were **389 actual ROI search frames**.

| Source | Full duration | Eligible selected-ball coverage, previous → fresh | Estimated assigned team time | Unknown | Contested | Pass proposals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dfl_test_10 | 30s | 48.27% → 49.33% | 2.24s | 27.68s | 0.08s | 0 |
| dfl_test_11 | 30s | 33.33% → 38.13% | 3.56s | 26.36s | 0.08s | 1 review-only |
| input_vid | 31.37s | 26.43% → 30.81% | 0.2083s | 30.9951s | 0.1666s | 0 |
| untitled_design | 16.967s | 16.31% → 23.58% | 0.4667s | 16.4667s | 0.0333s | 0 |
| dfl_test_1 | 30s | 24.13% → 25.87% | 0.84s | 29.16s | 0s | 2 review-only |

These coverage changes are **not measured accuracy**. Classified team coverage is only 0.66–11.87% per clip. The three proposals are unreviewed: dfl_test_11 at 10.24s; dfl_test_1 at 22.88s and 25.28s. Their ball support is 53.66%, 64.41% and 54.55% respectively. All three are excluded from possession flight totals. This does not establish complete pass counts, pass completion or reliable full-match possession.

Evidence under `data/qa/sequence-analytics-2026-10-25/`: `fresh-runs.json`, `fresh-inference.log`, `comparison.json`, `outputs/<run-id>/` and `published-runs.json`. Original hashes, analysed frame counts and annotated playback duration were checked before publication to the private Library with `-fresh.mp4` labels. The reusable sample remains its complete 30 seconds / 750 frames.

## Source-bound review and evaluation

The dashboard adds a source-time possession strip, interval filtering and video seeks, team support breakdowns, partial-tracking candidate evidence and expanded JSON/CSV exports. Team/stoppage context is saved separately from approved events using a protected, source-hash-bound API with revision conflicts. Drafts survive clip switches in the current tab; a successful save followed by a failed refresh retries the read without repeating the write.

The independent benchmark workspace contains the five **raw** source videos, blank sequence labels and 120 uniformly selected ball frames. Predictions are not copied into annotations. Run the reviewer at `http://127.0.0.1:8877/review.html` while its local server is running. Alternatively, serve `data/qa/sequence-analytics-2026-10-25/benchmark` locally. Export reviewed labels to JSON; do not call unreviewed/predicted labels ground truth.

Scoring validates source hashes, duration, explicit provenance and completely reviewed windows. Event matching is one-to-one with a default 0.5-second timing tolerance. Possession scoring retains unclassified predictions rather than removing them from the error denominator. Labels must be independently human-reviewed by default; a separate explicit visual-pilot option never reports human validation. The clips are development examples with unverified match grouping; they are not an independently held-out fixed-camera test set.

Example scorer:

```text
node scripts/python.mjs scripts/sequence_benchmark.py score reviewed-labels.json data/qa/sequence-analytics-2026-10-25/outputs/<run-id> --output scores.json
```

A five-frame **Codex visual diagnostic** on input_vid found visible ball misses near player feet. `ball-visual-pilot.json`, `ball-pilot-diagnostic.json` and raw frame images record this debugging exercise. It is neither an independent human benchmark nor a representative model accuracy score. No human labels were fabricated and no model fine-tuning is claimed.

## Remaining accuracy blockers

1. Independent human review of the prepared ball/event/possession annotations. This is needed to measure false positives, missed passes and possession error, and to build trusted fine-tuning data.
2. Actual fixed wide-angle full-pitch footage at the intended resolution. The current five clips are panning broadcast views. Camera calibration and held-out landmarks must come from the target footage; none were invented.
3. A ball model trained/evaluated on representative small/occluded balls, plus more consistent player/team association. The current detector still misses visible balls; a looser counting threshold cannot solve those evidence gaps.

Existing compute protections remain five GPU attempts per UTC day and twenty per month. The five fresh jobs used today's allowance; October's ledger is now 13/20. No additional GPU jobs are required for reviewing footage, scoring labels, changing context or viewing stored analyses. No quota or billing settings were changed.

## Verification

The full backend suite passed **78 tests**, including short control/one-touch cases, supported and partial gaps, cut/contested/identity barriers, exact state partitions, context authentication/revision conflicts/recalculation, approved-record preservation and benchmark matching/provenance. Frontend production build passed. Browser QA covered real sources, evidence seeks, 375px layouts, controlled save/refresh failures and conflicts, source switching, exports and benchmark import validation. See `backend-tests.log` and `frontend-qa.md` in the evidence folder. Hosting verification and final deployment references are recorded below after publication.

Both Modal app deployments succeeded. Read-only hosted verification checked all five fresh runs: matching source hashes/full frame counts/durations, signed metrics HTTP 200, video byte ranges HTTP 206, context revision 0, zero approved events and an unchanged job-ID set. The full sample remains 30 seconds / 750 frames. Unauthenticated context access returns HTTP 401. Evidence: `hosted-verification.json` and `additional-verification.json`. The fresh CUDA detector path was exercised on all five sources; the new temporal calculation was then applied to their complete tracking on CPU. No further GPU run has been submitted since the latest worker image deployment.

Independent same-provider UI evaluation passed with no blocking revisions (a different-provider evaluator was unavailable). Actual desktop 1440px, tablet 768px and mobile 375px sources were checked. All three partial proposals retained review-only/excluded-possession wording and correct evidence seeks. Isolated browser fixtures covered save success, a saved context followed by a failed refresh with no repeated POST, stale-revision conflict, retained per-source drafts and tab-wide unload warnings. Benchmark checks covered source SHA/import guards, explicit human confirmation, pilot provenance, export and unfinished-form restoration. No real human records were changed by QA. Accessibility scans found no scoped violations; raw-video caption checks remain manual, and offscreen mobile table contrast had incomplete checks.

The review server initially prevented seeking; `scripts/serve_sequence_benchmark.py` now serves byte ranges. Independent raw-video tests confirmed dfl_test_10 seeking/capture at 5 seconds and input_vid at 12 seconds after a source switch. Start this server instead of Python's basic `http.server` for annotation.

Screenshots: `temporal-desktop.png`, `temporal-mobile.png`, `context-desktop.png`, `context-mobile.png`, `candidates-desktop.png`, `candidates-mobile.png`, `benchmark-desktop.png`, `benchmark-mobile.png`. The downloadable `fresh-videos-and-analytics.zip` contains all five complete annotated videos and source-bound JSON/CSV results. The raw independent reviewer remains in the separate `benchmark/` folder.

Vercel deployment `dpl_4uP54ggKtTfVxVJkx9JdhFWQk77v` is live at https://laas-football.vercel.app, publishing `index-CimZiqsL.js` / `index-Co7llVfE.css`. Its production TypeScript/Vite build passed (1,586 modules). Both dependency install audits reported zero known vulnerabilities; this is not a security certification.

Authenticated production browser checks switched across all five fresh sources, preserving exactly one temporal panel/video and the correct duration, totals, candidate count and saved context revision. The dfl_test_1 evidence button sought natively to 22.1 seconds and kept playback paused. At 390×844 there was no document overflow, duplicate IDs or broken heading-label references. Published HTML/JavaScript contained the hosted API and temporal context features without the workspace access key. No browser errors were recorded. Proofs: `production-verification.json`, `production-seek.json`, `production-mobile.json`, `production-bundle-scan.json`; final screenshots: `production-temporal-desktop.png`, `production-temporal-mobile.png`. These are integration/browser checks, not model accuracy or physical-device certification.
