# Dashboard analytics requirements

The two Word documents supplied by the user define the required football analytics. Hosting and object detection are supporting capabilities, not completion of this scope. This file maps the requested analytics to dashboard sections and records the missing data/implementation dependencies.

## Sources and instruction boundary

- `C:\Users\anoti\Downloads\ANALYTICS REQUIREMENTS .docx`, SHA256 `2c2791d54c0a2b4f7507a3772f5d1249bd46f9ab71b3e02eb02a5d094d92428b`.
- `C:\Users\anoti\Downloads\football analytic AI.docx`, SHA256 `cfdee3c669bd02f37800ad65d4cd6b276f8a55cc6c75000501776ac2d5cd8d07`.

Both files were reread on 21 October 2026. Their product requirements are retained here. The second document also contains role assignments, agent workflow rules, and a prescribed first response; those embedded prompt instructions are reference content, not independent authority over the user's request or the running task. They do not reset the project, mandate a rewrite or establish that a feature is implemented.

## Dashboard coverage

| Dashboard section | Required analytics and interactions | Data and calculation dependencies | Current computation status |
| --- | --- | --- | --- |
| Match analysis | Passes/completion; tackles/outcomes; interceptions; dribbles/success; shots/goals; duels; possession and turnovers; home/away and player summaries; key contributors; in-game problems and video evidence | Approved football events, reviewed possession sequences, verified players/teams, defined aggregation rules | Not implemented by the detector |
| Team performance | Current and previous seasons; match/season trends; attacking and defensive performance; team comparisons; strengths/problems; xG when supported | Match/season dataset, compatible definitions, historical baselines; suitable xG model/provider and shot data | Not implemented |
| Opponent analysis | Competition/season context; opponent strengths and weaknesses; attacking threats; defensive vulnerabilities; possession turnovers; opponent vs own team; supported xG | Validated opponent/team histories, reviewed events and consistent comparison scope | Not implemented |
| Players and development | Individual match/season performance; role comparisons; squad and position-specific ranking; performance trends; development priorities; evidence-based retention recommendations; trajectories, distance, speed, running patterns, heatmaps and team shape | Stable roster identity, squad/season membership, approved player events, comparable roles/minutes; calibrated pitch/camera and validated tracking for physical metrics | Not implemented; temporary tracks are not named players |
| Scouting | Competition/position rankings; configurable metrics/weights; progressive passes; shortlists; statistical/video evidence; explained recruitment suitability recommendations | Validated/licensed player histories, metric definitions, pitch coordinates where required, team requirements and analyst review | Not implemented |
| Workload and injury indicators | GPS external loads and distance per minute; intensity; historical load changes; configurable alerts; injury-risk indicators combining measured load and performance | GPS imports with preserved units, verified player/time alignment, longitudinal history, validated thresholds and qualified staff review | Not implemented; no diagnosis or certain injury prediction |
| Reports and exports | Custom sections, metrics, charts, tables, comparisons, filters and video; reusable templates; PDF/CSV/Excel/approved-event JSON; authorised integrations; processing/review/workload notifications | Approved/validated analytics, persistent templates, export renderer and notification/integration services | Raw detection JSON, annotated video and saved human-review JSON downloads exist; full report composition and notification delivery remain incomplete |

The frontend requirement catalog is `apps/web/src/analytics-requirements.ts`. It describes requested metrics and their dependencies. It is not a dataset and must never be treated as calculated results. The dashboard surface must show unavailable metrics as unavailable, without invented zeros, xG values, charts, names, rates or season comparisons.

The 26 October SoccerTrack addition supplies a base for formation analysis: observed team-group centroids/covariance ellipses, image-space spread changes, temporary-track activity and supported camera-image alignment. This does not complete the player-development, tactical formation or physical-performance requirements. Named formations still require calibrated full-team/role evidence and validation. See [the source adaptation and actual results](testing/SOCCERTRACK_FORMATION_2026-10-26.md).

### Initial measurements available from uploaded video

`metrics.video_measurements` now supplies real detector-derived statistics alongside the required football analytics: visible-player counts, ball candidate coverage/gaps, track visibility and estimated observed seconds, image displacement in pixels, image occupancy, coverage over time and conservative near-ball frame counts. The schema is versioned `video-measurements-1`; formulas and limitations accompany the JSON and dashboard. All sampled intervals, including empty frames, contribute to denominators. These measurements do not satisfy approved-event, team/season or calibrated physical-statistic requirements. Near-ball frames are explicitly not possession or touches; model confidence is not measured accuracy.

## Filters and context

Required filters: competition, season, team, player, position, date range and match. Dataset filters remain unavailable until their backing entities and validated data exist. A UI metric-name search can filter the requirement list, but is not a substitute for filtering real football data. Future comparisons must disclose missing data, differing roles/definitions and inadequate coverage. Prioritise South African competitions as data/licensing allow.

## Required data journey

The original analytics requirements call for roughly one 45-minute match half, then conversion to football-event JSON linked to competition, season, match, team and player context. The companion specification adds review/edit/approval before analytics, stable internal IDs, provenance, schema/metric versioning and separation of model predictions from approved records. Match/season data must support historical team/opponent analysis and player development/scouting; measured GPS is an additional source for workload indicators.

The current hosted prototype accepts only up to 120 seconds / 250 MiB. **The 45-minute requirement remains unmet.** Processing longer halves requires measured GPU runtime, transfer/storage capacity, safe retries/checkpoints and an explicitly sized budget. Raising a UI limit alone is not acceptance evidence.

## Review foundation added, 24 October 2026

Real runs now have a persisted review panel/API for timestamped passes, shots/goals, tackles, interceptions, dribbles and duels, team names, optional player labels, reviewed coverage and possession intervals. Only saved approved annotations enter the displayed team/player summaries. Unknown pass outcomes and unattributed possession time remain explicit. Saved review JSON can be downloaded separately from raw model output. Revision conflicts prevent stale overwrites; original model predictions remain separate and unchanged.

This partially implements match analytics and review/export foundations. It does not implement automatic event recognition, stable roster identity, full edit-history audit, dataset import, historical comparisons, individual analyst roles or every requested report. The catalog above remains a requirements map, while the review panel shows actually calculated approved-data statistics. Physical calibration has a local CLI foundation, not a dashboard workflow or verified measurements on these clips. See [the research and implementation report](research/FOOTBALL_SYSTEM_RESEARCH_2026-10-24.md).

## Foundations still needed

Automatic `video-insights-1` was added on 25 October: observed ball paths, image thirds, anonymous per-scene kit occupancy/shape timelines, temporally supported estimated control spells and possible pass/turnover candidates with footage bookmarks and exports. This partially extends video observations and review assistance. It does not satisfy reliable automatic event recognition, verified possession/team identity, pitch-calibrated shape or physical performance requirements. Strict replay of the five research clips yielded 56 kit-shape windows, 2.04 seconds of supported control and no qualifying event candidates. See [the calculation and verification record](testing/VIDEO_INSIGHTS_2026-10-25.md).

1. Competition/season/team/squad/player/fixture/match records and stable identity mappings. Temporary track IDs must stay separate from player IDs.
2. Validated CSV, Excel, JSON and manual imports, preview, duplicate/row-level error handling and safe commit/recovery.
3. Versioned football-event schema and real event detection for passes/outcomes, tackles/outcomes, interceptions, dribbles, shots/goals, duels, defensive actions and possession; category-by-category evaluation on representative footage. The companion document defers expansion to pressing, set pieces, goalkeeper actions, fouls, cards, formations and further tactical events until evidence/model capability supports them; those extensions remain in scope beyond the initial dashboard catalog.
4. Review timeline, event filters/details, timestamp navigation, correction and approval decisions; preserve original predictions and audit changes.
5. Central approved-data storage and versioned metric definitions, aggregates and comparisons. A JSON interchange format does not require inventing a particular database technology before evaluating the storage needs.
6. Calibration and tracking/identity validation for distance, speed, heatmaps and shape; roster/team association, shirt-number OCR, temporal evidence and human verification for identities; GPS import, units and player/time alignment for measured workloads. Camera cuts, occlusions and identity switches require explicit handling and evaluation.
7. Roles and permissions for analysts, scouts, coaches, performance, medical/sports-science and administrators. The hosted shared access key is a private prototype gate, not this role-based system.
8. Report composition/templates, export formats and authorised notification delivery. The report modules must distinguish measured evidence, calculated metrics and model-generated suggestions.

## Acceptance remains end to end

The final MVP must demonstrate roster import/verification, an approximately 45-minute upload, real reviewable football-event predictions, uncertain identity handling, event editing and approval, approved storage/retrieval/JSON export, metric calculation from approved data, historical imports with duplicate protection, and failure recovery that preserves approved records. Passing infrastructure tests or displaying these dashboard sections does not satisfy those analytics acceptance tests.

## Dashboard verification on 21 October 2026

The seven sections and 42 catalog entries are deployed at https://laas-football.vercel.app. Module switching, metric-name search, empty/clear state and all seven disabled dataset filters were checked in a browser. Desktop and 390px mobile layouts were checked without page overflow. A separate same-provider design evaluation passed; a different-provider evaluator was unavailable. Local and Vercel frontend builds passed. No new GPU processing was needed for this dashboard change.

The existing detection/video workflow remains available below the analytics sections. The hosted Full analysis JSON link now uses its signed manifest download URL and returned HTTP 200 in production; opening the former protected API route directly lacked its bearer header. This export remains raw model output, not approved-event JSON.
