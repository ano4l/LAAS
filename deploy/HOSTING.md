# Hosted private prototype, 21 October 2026

## Research update, 24 October 2026

The web API and GPU worker were redeployed with persisted source-bound human review and the revised association/ball search pipeline. Five verified complete reprocessed clips are in the private Library as `*-research.mp4`. Hosted checks confirmed review reads (200), missing-key rejection (401), empty-review persistence, stale-save rejection (409), signed video ranges (206) and the unchanged full 30-second sample. No additional GPU inference was submitted and no allowance was increased. New CUDA execution of the updated detector/tracker is still a subsequent acceptance check when the existing quota permits a real job. See [research evidence and limitations](../docs/research/FOOTBALL_SYSTEM_RESEARCH_2026-10-24.md).

Current frontend deployment: `dpl_6jZeg5vxJBGa4enZnQsUghGBavRe`, production alias https://laas-football.vercel.app. Production build uses Vite 7.3.7 after compatible build-tool patches; both npm audits reported zero known vulnerabilities. Browser verification loaded the review panel and full 30-second video (readyState 4); the published bundle contains the configured hosted API and no workspace key.

Saved review records live beside a completed output as `review.json`, outside the immutable worker artifact hash set. They require the workspace bearer key and use server revision checks; saving a review does not launch GPU compute. The browser downloads its last server-fetched/saved review separately from raw detection JSON. Unsaved per-run drafts are retained in memory during Library navigation within the tab; refresh/close clears them after a browser warning. Click **Save review** to persist applied records.

The dashboard now displays real camera-relative calculations for uploaded videos. Existing completed hosted jobs are backfilled from their full tracking JSON without GPU inference. **Open sample · 30s** calls authenticated `GET /sample`, reusing the complete 750-frame reference analysis. It does not submit a new GPU job. To prepare a new sample from a verified real run, use `node scripts/python.mjs scripts/publish_sample.py RUN_DIRECTORY SOURCE_VIDEO`; this verifies source hash and full playback duration before publishing existing artifacts to the private volume and writes the local sample registry. Avoid republishing duplicates; the currently installed sample is job `60a81c3d5aca41df95e7b347cbbe5840`.

- Frontend: https://laas-football.vercel.app
- Hosted API: https://ano4l--laas-football-web-web.us-east.modal.run
- GPU worker dashboard: https://modal.com/apps/ano4l/main/deployed/laas-football
- API dashboard: https://modal.com/apps/ano4l/main/deployed/laas-football-web

The Vite frontend is hosted on Vercel. Browser uploads go directly to the CPU API on Modal, which persists the source clip and job record in a private Modal Volume, then spawns the private L4 worker. The browser polls status and receives signed video/JSON download links after completion. This path does not require the local API or PC to remain running. CPU and GPU functions scale to zero when idle; cold starts and GPU capacity queues can delay processing.

## Access

This is one private shared workspace, not a multi-user account system. The owner access key is in the ignored local file `data/qa/web-access-key.txt` and the Modal Secret `laas-web-access`. Enter it on the Vercel sign-in screen. It is stored in that browser tab's session storage. No Modal account token or workspace access key is embedded in the frontend build. The API uses a bearer key; media URLs have path-bound signatures valid for one hour. Refresh a run to obtain fresh links. Anyone holding the workspace key can access that workspace's videos and incur its allowed compute usage. Rotate the Modal Secret and redeploy the web API to revoke a key, then update the local key file.

## Limits and persistence

Clips: at most 120 seconds, 250 MiB, 4K / 60fps. One GPU container, five admitted attempts per UTC day and twenty per UTC month. The monthly allowance was not increased for deployment tests. These controls are not an account spending cap. Other Modal apps, startup time, API use and persistent storage can consume credits too. No billing limits or top-up settings were changed by the agent.

The hosted API admits one outstanding web job per workspace. Job JSON lives under `/web-jobs`, while uploads and outputs live under `/requests/JOB_ID` in `laas-football-jobs`. Completed hosted results persist there until deliberately removed; they are not automatically deleted when the browser closes. Review storage in Modal and remove unwanted completed requests and their matching job JSON together. Do not remove directories for running calls. Interrupted submission and provider outages require checking the Modal dashboard before resubmitting; retries can consume another attempt. Function-call results are provider-managed, so poll jobs to persist completion while their call results remain available.

## Deployment

GPU: `node scripts/python.mjs -m modal deploy deploy/modal_app.py`

API: `node scripts/python.mjs -m modal deploy deploy/modal_web.py`

Frontend: `vercel deploy --prod --yes`

The production Vercel project stores `VITE_API_URL`, `VITE_REQUIRE_ACCESS_KEY=true`, and `VITE_DEMO_MODE=false`. They are public build configuration, not credentials. `vercel.json` builds only the frontend; `.vercelignore` excludes backend code, videos, weights, local secrets and Python environments. Current source changes remain in the workspace; future Git deployments require committing the corresponding code.

## Verification

Modal CLI test: five seconds, 125 analysed frames, NVIDIA L4, 11.24 processing seconds; imported artifacts and source SHA256 verified. Local-browser-to-Modal test: full 30 seconds, 750 analysed frames, 31.97 processing seconds, correct completed job. Vercel-browser test: five-second upload through the hosted API, all 125 frames analysed at 1280px, 9.5 processing seconds, browser video playback confirmed at five seconds / 1280x720 with readyState 4. Tracking JSON download returned HTTP 200 and 3,933 actual model observations. Processing timings exclude upload, queue, cold start and download time.

The hosted API rejects data requests without a key (401). Automated tests cover rejected uploads, job completion, access checks, signed media tampering/expiry and range downloads. The deployed HTML/JS were checked to contain the hosted API configuration and no access key. Frontend build and 21 tests passed. Ball candidate counts remain unreviewed detections; these checks do not establish football-analysis accuracy.

## Dashboard polish deployed 25 October 2026

Vercel deployment `dpl_AB6CN7bBzKb125YBBwavkeYgps7m` supersedes the frontend deployment above at https://laas-football.vercel.app. The production browser verified compact authenticated `/library` startup, selected-run loading, correct analytics through four research-clip changes, full-duration playback, and desktop/mobile views. The 390px layout had no document overflow, duplicate IDs or broken heading-label references. The published bundle contains the hosted API and no workspace key. The Modal API provides the compact listing without provider lookups or unrelated measurement backfills. No GPU job was submitted for dashboard QA; admission limits remain unchanged. Evidence: [dashboard refinement report](../docs/testing/DASHBOARD_POLISH_2026-10-25.md).

## Automatic video insights deployed 25 October 2026

The latest frontend is `dpl_4ygFJWMegsqKRJqmXrZbeRafy9Rj` at the same production alias. Both Modal API/worker images now include `video-insights-1`. Fresh uploads calculate the new metrics; selected older real runs backfill from full tracking, without a GPU call. All five research entries and the full 30-second sample have been checked. Anonymous kit shape, observed ball paths, supported control estimates, review candidates and source-linked exports remain unreviewed; approved human summaries are separate. Production clip switching/seeking, full video playback, mobile layout and key exclusion were checked. Compute admission/billing settings were not changed, and deployment checks did not submit GPU work. See [formulas, actual results and verification](../docs/testing/VIDEO_INSIGHTS_2026-10-25.md).

## Temporal analytics and fresh full-clip testing — 25 October 2026

Current Vercel deployment: `dpl_4uP54ggKtTfVxVJkx9JdhFWQk77v`, https://laas-football.vercel.app, assets `index-CimZiqsL.js` / `index-Co7llVfE.css`. Both Modal apps now include `video-insights-2` and source-bound team/stoppage context routes. Five new `*-fresh.mp4` Library entries contain full L4 inference across 3,512 frames; all three partial pass proposals remain unreviewed and are excluded from possession flight totals. Hosted source/hash/frame/duration/signed-media/context checks passed; the reusable sample remains 30 seconds / 750 frames.

Existing compute limits remain unchanged: five attempts per UTC day, twenty per month. The fresh runs used today's five attempts and October is now 13/20. Dashboard browsing, context changes and local benchmark review do not submit GPU jobs. No billing settings were changed. The independently reviewed fixed-camera dataset and reliable full-match possession accuracy remain unavailable. See the [current detailed report](../docs/testing/TEMPORAL_POSSESSION_2026-10-25.md) for results and verification scope. Earlier deployment sections above are historical.

## SoccerTrack collective shape — 26 October 2026

Current frontend: `dpl_4T2jCBfupW2x4tHWLf42bkGK8txS` at the same production alias, assets `index-BzsyCYq7.js` / `index-sD3nrjy6.css`. Both Modal images include `video-insights-3` / `soccertrack-dynamics-1`. Old real results recalculate from stored full tracking on selection, preserving source identity, review records and team/stoppage context. The live Analytics view adds observed group ellipses, source-time spread, activity and camera-image alignment. Named tactical formations/pitch metrics remain unavailable without calibrated/full-team evidence. All five saved fresh sources and the full sample passed API checks; production desktop/mobile binding, seeking and missing-evidence checks passed. No GPU jobs, quota changes or billing changes were made for this addition. [Methods and verification](../docs/testing/SOCCERTRACK_FORMATION_2026-10-26.md).
