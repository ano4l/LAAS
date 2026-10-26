# Dashboard usability and visual refinement

User request: polish the dashboard, especially video analysis and analytics, into a friendly and attractive working product. This refines the existing React/TypeScript/Vite application; it does not replace real video measurements with synthetic dashboard data.

## Audit

The original page placed a large requirement catalogue and repeated tracking tables ahead of the actual video. Completed jobs retained a prominent progress banner, clip switching lived near the bottom, the icon rail had inert controls, and human review was a long form below unrelated content. Dense caveats and repeated technical metadata obscured the useful actions. The seven analytics areas and 42 requirements remain product scope, but an unavailable-value catalogue should not dominate active clip analysis.

Visual direction: a calm football analysis workspace using the existing dark navy foundation, readable numeric typography, restrained mint/teal actions and distinct draft/approved states. Real video, coverage charts and source-linked evidence take priority. Camera-image occupancy is not presented as a calibrated pitch map.

## Library loading foundation

Both APIs now expose `GET /library`. It returns compact clip metadata, explicitly nullable headline summaries and compact job state. Full tracking arrays, event histories and signed media URLs are retrieved only for the selected clip through existing detail routes. The hosted route requires the workspace bearer key. Existing routes remain available for compatible clients.

The hosted selected-run route now resolves that run directly rather than constructing all full results first. Browsing the new library does not backfill missing measurements for unrelated completed clips and never starts GPU inference. Local listing uses indexed run summaries instead of reopening every full manifest. Startup indexing and successful worker completion maintain the local index; external CLI outputs are indexed on server restart or the older `/runs` refresh path.

Automated checks cover compact payload size, null unknown values, authorization, exclusion of private fields and full tracking payloads, no GPU invocation, preservation of signed selected-result links and isolation from unrelated incomplete legacy measurements. Status resolution remains on selected-job polling: the compact library does not query provider calls for unrelated jobs.

Hosted measurement across 17 completed runs: the compact library returned 14,409 JSON body bytes. The two former listing requests returned 14,159,635 and 14,162,325 bytes, totalling 28,321,960 bytes. The new listing is 99.949% smaller. The chosen full analysis still loads separately (833,153 bytes for the measured clip). These are body sizes with `Accept-Encoding: identity`, not a latency benchmark or a claim of equivalent speed improvement. Authorization rejection and signed selected-detail media links were also verified; no GPU jobs were submitted. Evidence: `data/qa/dashboard-polish-2026-10-25/hosted-library-verification.json`.

The complete 50-test backend suite passed after these changes. Existing FastAPI startup-deprecation warnings remain; the intentionally invalid upload fixture also emits the expected decoder warning.

## Analytics integrity retained

- Detector observations, anonymous kit groups and temporary identities remain unreviewed.
- Camera-relative movement/occupancy is not distance run or calibrated pitch position.
- Ball candidate coverage and model confidence are not accuracy or possession.
- Football-event statistics use saved approved human annotations only. Drafts, unknown outcomes and reviewed-coverage denominators stay visible.
- Saving, revision conflicts, explicit discard/reload, saved-only export and tab-wide unsaved-work protection remain required behaviours.
- Browsing stored results must not consume GPU attempts. Hosted upload admission stays at 120 seconds / 250 MiB.

## Implemented workspace

The real-video path now uses `apps/web/src/RealWorkspace.tsx` with Overview, Video review, Analytics, Library and Exports views. The selected clip and duration stay clear, clip switching is near the top, and the default overview shows measured headlines and the actual full annotated video. The finished processing banner is compact instead of occupying a large permanent block. Technical model/source details remain available in disclosure.

Analytics foregrounds model measurements and provides a distinct saved-approved-records source view. Measured coverage bars support timestamp selection and video seeking; the camera-image occupancy grid provides exact cell counts. Track exploration is searchable, sortable and paginated. The full seven-area/42-item analytics scope remains accessible in a secondary disclosure, with unsupported data requirements and dataset filters explicitly identified.

The video and review controller persist across workspace views. Overview and Analytics use the same server-fetched/saved review snapshot, checked against the selected run ID. Editing a draft does not change approved statistics. Human review retains all event types/outcomes, source-time controls, event search/filtering, explicit approvals, possession/coverage intervals, stale-save retention/reload and saved-only JSON download.

Background processing is separated from the chosen completed clip: polling can continue while browsing, and completion must not overwrite a newer clip choice. Upload admission is disabled while an active job exists. Upload and selection responses have selection guards. Library and selected-detail loads have loading/error/retry states. These interaction paths are validated with controlled browser fixtures so they do not consume GPU jobs or create invented events on real footage.

## Validation

Final frontend production build passed (TypeScript + Vite, 1,583 modules), and `git diff --check` passed. Independent same-provider design evaluation completed its final round with PASS. A focused source-binding regression was corrected by keying the model/track container by run ID; measurement and track components no longer share duplicate sibling React keys. Repeated switches between six 17/5/30-second local legacy clips leave exactly one measurement panel and one track explorer, with the correct current duration and analysed-frame count every time. No stale graphs remain beneath a newly selected source.

Checks also confirmed:

- Unique element IDs and valid heading/ARIA references after repeated clip switches. Global Analytics accessibility scan: zero violations, zero incomplete checks, 44 passes; review scan: zero/zero/22. These are scoped automated/browser checks, not general accessibility certification.
- Segment selection seeks to the correct source at 20 seconds. Previous playing video pauses/disconnects on a clip change; one current video remains.
- Eight-row track pagination/search and exact image-cell inspection work. Null values remain unknown, not implied zero coverage.
- Controlled approved-record fixture preserves the distinction between 100% completion among known pass outcomes and only 50% known-outcome coverage (1/2 recorded passes).
- Controlled delayed-selection/upload/background-job fixtures retain the newer selected clip, keep polling active work, disable new uploads globally and offer the completed result without replacing current footage.
- Draft/forms restore across clips/views, stale-save conflicts retain edits until explicit reload, saved-only exports exclude unsaved changes, and tab-wide unload protection clears when QA drafts are cancelled. All fixtures were restored and no invented events were saved to real footage.
- 375px and 390px layouts have no horizontal document overflow. Navigation has visible scroll affordance; tables scroll within their own labelled regions.

Screenshots are under `data/qa/dashboard-polish-2026-10-25/`: `overview-desktop.png`, `video-review-desktop.png`, `analytics-desktop.png`, `analytics-compact-desktop.png`, `analytics-timeline-grid-desktop.png`, `tracks-paginated-desktop.png` and corresponding mobile views. `qa-*` images are controlled fixtures, not real football observations.

## Production deployment and smoke check

Vercel deployment `dpl_AB6CN7bBzKb125YBBwavkeYgps7m` is live at https://laas-football.vercel.app, publishing `index-C2pgC3Vc.js`. The authenticated production browser loaded `/library` once instead of the former two full listings, then selected only the requested result and review. Switching through untitled_design (509 frames / 16.97 seconds), input_vid (753 / 31.37), dfl_test_10 (750 / 30) and back to dfl_test_11 (750 / 30) retained exactly one measurement panel with the correct source. The final 30-second video reached readyState 4. At 390x844, document overflow, duplicate IDs and broken heading-label references were all absent. Desktop playback/review and mobile Overview/Analytics were visually inspected. These browser checks do not establish real-device behavior or model accuracy.

Published bundle inspection confirmed the hosted API and compact library route are present, and the workspace access key is absent. Root/frontend dependency audits reported zero known vulnerabilities during deployment. No GPU job was submitted and no approved events were added by these checks.

Production evidence: `production-verification.json`, `production-switch-1.json` through `production-switch-3.json`, `production-bundle-scan.json`, `production-overview-desktop.png`, `production-analytics-desktop.png`, `production-video-review-desktop.png`, `production-overview-mobile.png` and `production-analytics-mobile.png` in the QA directory above.
