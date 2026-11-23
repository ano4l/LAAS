# Open-source building blocks for reporting, export, data import and dashboards in a football analytics web app, plus a survey of open football-analytics platforms (as of 18 November 2026)

Saved by the coordinator from the researcher's returned text (the researcher's own file write was rejected by the harness). Content is the researcher's; lightly condensed.

Reading notes:

- **(V)** = verified on 2026-11-18 from the GitHub REST API (stars, SPDX licence, archived flag, last push, latest release), the PyPI JSON API, the npm registry, or a fetched documentation page.
- **(K)** = capability known from the library's documentation; link given but not re-fetched this session. Re-check before relying on it.
- npm "unpacked" figures are the published package size on disk, not the minified+gzipped browser bundle.
- Repositories that now resolve to a new owner (V): `statsbomb/open-data` → `hudl/open-data`; `statsbomb/statsbombpy` → `hudl/statsbombpy`; `great-expectations/great_expectations` → `fivetran/great_expectations`; `casbin/pycasbin` → `apache/casbin-pycasbin`; `jorisschellekens/borb` → `borb-pdf/borb`.

---

## 1. Football visualisation in the browser versus server-side rendering

### Takeaway
There is no maintained, full-featured football charting library for React. The only dedicated JS library, d3-soccer, is small (77 stars, last release December 2024) and covers pitch, rectangular heatmap and SPADL action plots only. Shot maps, pass networks, hexbin/KDE heatmaps and pizza charts must be composed from D3 modules or Observable Plot on a hand-written SVG pitch. The mature football chart library is Python's mplsoccer (MIT, v1.8.1, September 2026), the natural engine for PDF output.

### Cited Findings

**Football-specific libraries**

| Option | What it does | Licence | State (V) | Source |
|---|---|---|---|---|
| d3-soccer (probberechts) | D3 plugin: `d3.pitch()` (105×68 default), `d3.heatmap()` with rectangular bins, `d3.actions()` (SPADL actions), `d3.actionsTable()`, `d3.scoreline()`. No shot maps, pass networks, hexbin, radar or TypeScript typings mentioned; example loads D3 v5 | GitHub detects BSD-3-Clause; npm says MIT (conflict) | 77 stars; rel v0.3.0 2024-12-07; push 2025-03-03 | [GitHub](https://github.com/probberechts/d3-soccer), [npm](https://www.npmjs.com/package/d3-soccer) |
| react-soccer-lineup (giustini) | React pitch with two line-ups (formation display only) | npm MIT; none detected on GitHub | 57 stars; npm 1.0.0-beta.12 2026-06-13 | [GitHub](https://github.com/giustini/react-soccer-lineup) |
| mplsoccer (Python, matplotlib) | Nine pitch types, coordinate standardisation, radar, Nightingale/pizza charts, bumpy charts, arrows, heatmaps, hexbins, scatter, comet lines | MIT | 555 stars; PyPI 1.8.1 2026-09-14; Python ≥3.10 | [Docs](https://mplsoccer.readthedocs.io/en/latest/), [GitHub](https://github.com/andrewRowlinson/mplsoccer) |
| soccerplots (Python) | Older radar/pitch plotting | MIT | 88 stars; last push 2021-04-29 (unmaintained) | [GitHub](https://github.com/slothfulwave/soccerplots) |

- GitHub searches for "soccer pitch react", "football pitch d3" and "soccer pitch svg" returned no other library above 5 stars (V).

**General-purpose chart libraries (all permissive, commercial use allowed)**

| Library | Licence | Latest (V) | Relevant chart types | Source |
|---|---|---|---|---|
| D3 (modular) | ISC | v7.9.2, 2026-11-03 | Scales, shapes, `d3-contour` density contours (KDE), force layout; football charts are hand-built | [GitHub](https://github.com/d3/d3) |
| d3-hexbin | BSD-3-Clause | 0.2.2, 2017-03-28 | Hexagonal binning | [npm](https://www.npmjs.com/package/d3-hexbin) |
| Observable Plot | ISC | v0.6.19, 2026-11-05; 5,430 stars | Hexbin transform, density mark, dot/arrow/link/line marks, SVG output (K) | [GitHub](https://github.com/observablehq/plot) |
| visx | MIT | v4.0.0, 2026-06-11; 21,117 stars | React D3 primitives: shapes, heatmap, hexbin, network, radar (K) | [GitHub](https://github.com/airbnb/visx) |
| Recharts | MIT | v3.11.0, 2026-11-17; 27,722 stars | React SVG line/area/bar/scatter and RadarChart (K); no pitch or hexbin | [GitHub](https://github.com/recharts/recharts) |
| Apache ECharts | Apache-2.0 | 6.1.0, 2026-05-19; 67,752 stars | Radar, heatmap, graph, scatter, line; canvas or SVG; server-side SVG string rendering (K) | [GitHub](https://github.com/apache/echarts), [SSR](https://echarts.apache.org/handbook/en/how-to/cross-platform/server/) |
| Plotly.js | MIT | v4.3.0, 2026-11-10; 18,385 stars | Scatterpolar (radar), 2D histogram/contour, shapes for pitch lines (K); heaviest runtime | [GitHub](https://github.com/plotly/plotly.js) |
| Nivo | MIT | v0.99.0, 2025-05-23; 14,146 stars | Radar, heatmap, network, line, bump (K) | [GitHub](https://github.com/plouc/nivo) |
| uPlot | MIT | 1.6.32, 2025-03-14; 10,657 stars | Canvas time series only; raster in PDF | [GitHub](https://github.com/leeoniya/uPlot) |
| Chart.js | MIT | v4.6.1, 2026-10-22; 67,804 stars | Canvas charts incl. radar; raster in PDF | [GitHub](https://github.com/chartjs/Chart.js) |

- Open football apps (section 9) mostly use Python-side matplotlib/mplsoccer or Plotly in Streamlit rather than a JS football library (V, READMEs).

### Inferences
- Recommended for a small team: keep the existing hand-written SVG pitch as a React component and add only D3 modules (`d3-scale`, `d3-shape`, `d3-contour`, `d3-hexbin`). Shot maps, pass networks, heatmaps and pizza/radar charts are each roughly 50–150 lines of SVG on that base, stay accessible and print to PDF as vectors.
- Trend lines and bars: Recharts or Observable Plot are the lowest-effort options. ECharts is the best single library if server-side SVG of the same chart spec is wanted without a browser.
- Server-side mplsoccer is the lowest-risk route for PDF charts (one Python function per chart returning SVG), at the cost of no interactivity and duplicated styling.
- Two realistic architectures: (A) one chart codebase in React/SVG, PDFs by headless Chromium printing a report route — identical charts, heavy Chromium dependency; (B) React/SVG for interactive plus mplsoccer for PDF — no browser on the server, duplicated chart definitions.
- d3-soccer is worth reading for API ideas but is too small and stale to depend on.

### Gaps
- Minified+gzipped bundle sizes not measured. Accessibility features not re-verified. d3-soccer licence conflict (BSD-3 vs MIT) unresolved; both permissive.

---

## 2. PDF report generation from FastAPI (Windows and containers) and client-side options

### Takeaway
For HTML/CSS reports with embedded SVG charts, the two dependable server routes are Playwright/Chromium `page.pdf()` (Apache-2.0, browser-identical rendering, large download) and WeasyPrint (BSD-3-Clause, needs Pango — on Windows via MSYS2 or the standalone executable). ReportLab (BSD) and fpdf2 (LGPL-3.0) are pure Python but need programmatic layout and have limited SVG support. borb is AGPL/commercial and wkhtmltopdf is archived; exclude both.

### Cited Findings

| Option | Approach | Licence (commercial use) | State (V) | SVG / tables / Windows notes | Source |
|---|---|---|---|---|---|
| WeasyPrint | HTML+CSS → PDF in Python, no browser | BSD-3-Clause (yes) | v70.0, 2026-09-08; 9,799 stars | Needs Pango ≥1.44. Windows: use the release executable, or install MSYS2 and `pacman -S mingw-w64-ucrt-x86_64-pango`, optionally set `WEASYPRINT_DLL_DIRECTORIES`. Docs warn the executable "is regularly marked as malware by different antivirus companies". Own SVG renderer; no JavaScript | [First steps](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html), [GitHub](https://github.com/Kozea/WeasyPrint) |
| Playwright (Python) + Chromium | Headless browser prints HTML to PDF | Apache-2.0 (yes) | v1.65.2, 2026-11-14; 15,119 stars | `page.pdf()` uses print CSS media; `page.emulate_media(media="screen")` for screen styles. Runs JS, so React/SVG charts render as in the app | [Page API](https://playwright.dev/python/docs/api/class-page) |
| Gotenberg | Docker service wrapping Chromium/LibreOffice behind HTTP | MIT (yes) | v9.8.0, 2026-11-18; 13,627 stars | Container-only; needs Docker on a Windows server | [GitHub](https://github.com/gotenberg/gotenberg) |
| ReportLab (open-source edition) | Programmatic canvas + Platypus tables | BSD (yes) | 5.1.0, 2026-10-22 | Pure Python. SVG via separate `svglib` (LGPL-3.0-or-later) | [PyPI](https://pypi.org/project/reportlab/) |
| fpdf2 | Programmatic PDF, pure Python | LGPL-3.0-only (usable as a dependency) | 2.9.0, 2026-10-29 | "supports basic conversion of SVG paths into PDF paths"; unsupported SVG features can give "a completely blank PDF" | [SVG docs](https://py-pdf.github.io/fpdf2/SVG.html) |
| xhtml2pdf | HTML subset → PDF on ReportLab | Apache-2.0 (yes) | v0.2.28, 2026-10-25 | Pure Python; limited CSS (K) | [GitHub](https://github.com/xhtml2pdf/xhtml2pdf) |
| borb | Programmatic PDF | **AGPL-3.0-or-later or paid** | v3.0.9, 2026-08-03 | Exclude unless AGPL is acceptable | [LICENSE.md](https://github.com/borb-pdf/borb/blob/master/LICENSE.md) |
| wkhtmltopdf (+ pdfkit) | QtWebKit HTML → PDF | LGPL-3.0 | **Archived**; last release 2020-06-10 | Do not adopt | [GitHub](https://github.com/wkhtmltopdf/wkhtmltopdf) |
| Typst (+ `typst` Python binding) | Markup typesetter compiled from Python | Apache-2.0 (yes) | Typst v0.15.1, 2026-07-17; PyPI `typst` 0.15.2, 2026-11-05 | Images: png, jpg, gif, svg, pdf, webp; no browser or system libraries; templates are Typst markup | [Image docs](https://typst.app/docs/reference/visualize/image/), [typst-py](https://github.com/messense/typst-py) |
| Quarto | Pandoc-based publishing with parameterised documents | MIT | v1.11.9, 2026-10-30 | External CLI; `quarto render x.ipynb -P name:value` | [Parameters](https://quarto.org/docs/computations/parameters) |
| @react-pdf/renderer | React components → PDF | MIT | 4.9.0, 2026-08-27; 16,858 stars | Partial SVG; chart bridge `react-pdf-charts` reportedly does not support Recharts v3+ | [GitHub](https://github.com/diegomura/react-pdf) |
| pdfmake | JSON document definition → PDF | MIT | 0.3.14, 2026-11-08 | `svg` node takes an SVG string; native tables | [SVG docs](https://pdfmake.github.io/docs/0.3/document-definition-object/svgs/) |
| jsPDF (+ svg2pdf.js) | Imperative PDF in browser | MIT / MIT | jsPDF v4.2.1, 2026-03-17 | Vector SVG needs svg2pdf.js; html2canvas route rasterises | [jsPDF](https://github.com/parallax/jsPDF) |

- CairoSVG (LGPL-3.0-or-later, 2.9.1) converts SVG to PNG/PDF as a fallback. pypdf (BSD-3-Clause, 7.1.1) merges/stamps PDFs.
- Open football projects that already ship PDF match reports: [mostafasaad91/Football-Match-Analysis](https://github.com/mostafasaad91/Football-Match-Analysis) (MIT), [DanielTomaro13/WyScout-Tool](https://github.com/DanielTomaro13/WyScout-Tool) (MIT), [panospen/football-tactical-dashboard](https://github.com/panospen/football-tactical-dashboard) (no licence).

### Inferences
- Primary recommendation: render reports as HTML (Jinja2 or a React print route) and convert with Playwright/Chromium. It is the only option that embeds arbitrary SVG charts and CSS tables exactly as designed and behaves the same on Windows, Railway and Modal. Run it in a background job.
- Lighter fallback: WeasyPrint with server-generated mplsoccer SVG and HTML tables. No browser, but no JavaScript; the Windows server is the weak point (MSYS2 Pango DLLs).
- If neither a browser nor native libraries are acceptable: Typst via the `typst` wheel is the most self-contained vector-capable option.
- ReportLab is safest for simple tabular PDFs; for chart SVG, rasterise to 2–3× PNG.
- Client-side PDF duplicates layout code, has weaker SVG fidelity and cannot serve scheduled/emailed reports.
- Footprint ranking, smallest first: fpdf2 ≈ ReportLab ≈ Typst wheel < WeasyPrint (+Pango) < Playwright Chromium ≈ Gotenberg.

### Gaps
- No measured image sizes or render times. Whether WeasyPrint handles every matplotlib SVG construct was not tested. Tagged/accessible PDF support not researched.

---

## 3. Excel and CSV export/import libraries and safe-export concerns

### Takeaway
On the server, XlsxWriter (BSD-2-Clause) for exports and openpyxl (MIT) or python-calamine (MIT) for imports cover everything. In the browser, SheetJS CE is Apache-2.0 but its current version (0.20.3) is distributed only from cdn.sheetjs.com — npm `xlsx` is frozen at 0.18.5 (2022) — and ExcelJS has had no release since October 2023. All text exports need formula-injection neutralisation.

### Cited Findings

| Library | Role | Licence | State (V) | Source |
|---|---|---|---|---|
| XlsxWriter | Write .xlsx with formats, charts, validation | BSD-2-Clause | 3.3.1, 2026-10-28 | [GitHub](https://github.com/jmcnamara/XlsxWriter) |
| openpyxl | Read/write .xlsx | MIT | 3.1.5, 2024-06-28 (no release in over two years) | [PyPI](https://pypi.org/project/openpyxl/) |
| python-calamine | Fast Rust reader (xls/xlsx/ods) | MIT | 0.9.1, 2026-11-03 | [GitHub](https://github.com/dimastbk/python-calamine) |
| pandas | DataFrame I/O | BSD-3-Clause | 3.1.1, 2026-11-04 | [GitHub](https://github.com/pandas-dev/pandas) |
| polars | Fast DataFrame I/O | MIT | 2.1.0, 2026-10-25 | [GitHub](https://github.com/pola-rs/polars) |
| SheetJS CE (`xlsx`) | Browser/Node read/write | Apache-2.0 | 0.20.3 via CDN tarball only; npm registry "is out of date" at 0.18.5 (2022-03-24) | [Install docs](https://docs.sheetjs.com/docs/getting-started/installation/frameworks) |
| ExcelJS | Browser/Node .xlsx with styling | MIT | v4.4.0, 2023-10-19 (effectively unmaintained) | [GitHub](https://github.com/exceljs/exceljs) |
| PapaParse | Browser CSV parser | MIT | 5.7.1, 2026-11-11 | [GitHub](https://github.com/mholt/PapaParse) |
| defusedcsv | `csv` replacement neutralising formulas | Apache-2.0 | 3.0.0, 2025-09-02 | [PyPI](https://pypi.org/project/defusedcsv/) |

- OWASP: cells beginning with `=`, `+`, `-`, `@`, tab, carriage return or line feed may be read as formulas. Guidance: ensure no cell begins with those characters; or wrap each field in double quotes, prepend a single quote, and escape double quotes — [OWASP CSV Injection](https://community.owasp.org/attacks/CSV_Injection).

### Inferences
- Export on the server: CSV with Python's `csv` (UTF-8 with BOM) and XLSX with XlsxWriter. One code path serves downloads, scheduled reports and email attachments, and sidesteps SheetJS and ExcelJS.
- Injection rule: in CSV, prefix any string cell starting with a trigger character with a single quote (or use `defusedcsv`). In XLSX, use `write_string()` for names and notes; write numeric columns as numbers.
- For imports, read cached values only and never evaluate formulas.

---

## 4. Report templates and builders: how open BI tools model saved, parameterised reports

### Takeaway
Embedding a BI tool is a poor fit. Metabase and Grafana are AGPL, Superset is Apache-2.0 but a heavy separate service, and Evidence/Quarto/Streamlit/Dash/Panel are code-first with no end-user builder. What is worth copying is their data model: a saved report is a JSON document of ordered sections, each naming a metric/query id, a visualisation type and bindings to report-level filter parameters.

### Cited Findings

| Tool | Licence | State (V) | Saved/parameterised report model | Source |
|---|---|---|---|---|
| Metabase | AGPL outside `enterprise/` | v0.65.4, 2026-11-13; 50,402 stars | Saved questions on dashboards with filters mapped to parameters (K); static embedding in OSS with a "Powered by Metabase" badge, interactive embedding paid (search summary) | [LICENSE.txt](https://github.com/metabase/metabase/blob/master/LICENSE.txt) |
| Apache Superset | Apache-2.0 | 6.1.0, 2026-05-13; 75,560 stars | Charts (`params` JSON) on dashboards with layout JSON and native filters (K) | [GitHub](https://github.com/apache/superset) |
| Grafana | AGPL-3.0 | v13.3.1, 2026-11-10 | Dashboard JSON: `panels[]`, `templating.list` variables, `schemaVersion` (K) | [JSON model](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/view-dashboard-json-model/) |
| Redash | BSD-2-Clause | v26.9.0, 2026-09-24 | Saved SQL with `{{parameters}}`, widgets on dashboards (K) | [GitHub](https://github.com/getredash/redash) |
| Evidence | MIT | 40.1.8, 2026-02-06 | SQL in Markdown; templated pages; static site; no PDF export found | [GitHub](https://github.com/evidence-dev/evidence) |
| Quarto | MIT | v1.11.9, 2026-10-30 | Papermill-style `parameters` cell; CLI batch tool | [Parameters](https://quarto.org/docs/computations/parameters) |
| Streamlit / Dash / Panel | Apache-2.0 / MIT / BSD-3 | current in 2026 | Code-defined apps, not end-user builders (K) | [Streamlit](https://github.com/streamlit/streamlit) |

- Football precedent: Statlas (AGPL-3.0, Next.js 16 + FastAPI) keeps a `metric_registry.json` and `tiers.json` with "dated immutable snapshots", plus workspaces, query builder, "verified reports" and watchlists — [Statlas](https://github.com/themanoj-025/Statlas).
- UI building blocks: react-grid-layout (MIT, 2.3.0, 2026-10-05) and TanStack Table (MIT, 9.3.1, 2026-10-12).
- GitHub searches for "football report builder" and "soccer match report generator" returned zero repositories (V).

### Inferences
- Do not embed a BI tool. Each adds a second service with its own auth and styling, and none draws pitch charts. Build the report builder natively on FastAPI + SQLite.
- Lightweight template model (proposal distilled from Grafana/Metabase/Superset/Statlas):
  - `report_template`: `id`, `name`, `kind` (match | opposition | player | squad | workload), `schema_version`, `owner_id`, `visibility`, `parameters[]` (name, type, required, default), `sections[]`.
  - `section`: `id`, `title`, `layout`, `block_type` (chart | table | kpi_row | text | pitch_plot), `metric_ids[]`, `viz` (`type`: shot_map | pass_network | heatmap | pizza | radar | trend_line | bar | table; `options`), `filters`, `notes`.
  - `report_instance`: `template_id`, `template_version`, resolved `parameter_values`, `generated_at`, `generated_by`, `data_snapshot_hash`, output file references.
- A metric registry (id → label, unit, aggregation, resolver, allowed viz types, required role) is what makes templates safe: users pick metric ids and never write queries.
- Carry an integer `schema_version` with forward migrations, as Grafana does.

### Gaps
- Superset's embedding requirements not verified. Metabase's embedding terms came from a search summary. No open-source football project implements an end-user, section-based report builder.

---

## 5. Imports and validation: preview, duplicates and row-level errors

### Takeaway
The cleanest stack is a React wizard for upload → column mapping → preview, with authoritative validation on the server by pandera (MIT; lazy validation returns every failure with column, check, value and row index) and/or Pydantic v2 row models. HelloCSV is the actively maintained MIT importer; react-spreadsheet-import is tied to Chakra UI and unreleased since August 2024; TableFlow's csv-import has been quiet since October 2025.

### Cited Findings

**Server-side validation**

| Tool | Licence | State (V) | Fit | Source |
|---|---|---|---|---|
| pandera | MIT | 0.36.0, 2026-11-10; 4,530 stars | `lazy=True` raises `SchemaErrors`; `failure_cases` has `schema_context`, `column`, `check`, `failure_case`, `index`. Polars/PyArrow backends | [Lazy validation](https://pandera.readthedocs.io/en/stable/lazy_validation.html) |
| Pydantic v2 | MIT | 2.14.3, 2026-11-05 | Row models; already a FastAPI dependency; per-field error list (K) | [GitHub](https://github.com/pydantic/pydantic) |
| Great Expectations | Apache-2.0 | 1.25.0, 2026-11-12 | Pipeline-oriented; heavy for an upload wizard | [GitHub](https://github.com/fivetran/great_expectations) |
| frictionless-py | MIT | v5.21.0, 2026-10-15 | Table Schema descriptors and validation reports (K) | [GitHub](https://github.com/frictionlessdata/frictionless-py) |

**React / browser importers**

| Tool | Licence | State (V) | Features | Source |
|---|---|---|---|---|
| HelloCSV | MIT | v0.6.2, 2026-11-11; 951 stars | Frontend-only CSV importer workflow: custom columns, validations, transformations; upload, map, preview, confirm | [GitHub](https://github.com/HelloCSV/HelloCSV) |
| react-spreadsheet-import | MIT | v4.7.1, 2024-08-23; 560 stars | XLS/XLSX/CSV; mapping, validation/edit UI; built with Chakra UI | [GitHub](https://github.com/UgnisSoftware/react-spreadsheet-import) |
| TableFlow csv-import | MIT | npm 1.0.18, 2025-10-21; 1,789 stars | CSV/TSV/XLS/XLSX; `template` prop of columns | [GitHub](https://github.com/tableflowhq/csv-import) |
| react-csv-importer | MIT | **Archived** | Do not adopt | [GitHub](https://github.com/beamworks/react-csv-importer) |
| YoBulk | **AGPL-3.0** | Last push 2023 | Exclude | [GitHub](https://github.com/yobulkdev/yobulkdev) |

- football-data.co.uk columns an import profile should recognise: `Div, Date (dd/mm/yy), Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR`, plus `Attendance, Referee, HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR` and others — [notes.txt](https://www.football-data.co.uk/notes.txt).

### Inferences
- Recommended two-phase, server-authoritative flow:
  1. Upload and stage: store the file, compute a SHA-256 hash (warn on exact re-upload), parse to a staging table keyed by `import_batch_id` + `row_number`.
  2. Map: save column mappings as a named import profile per source.
  3. Validate (dry run): pandera for column checks; Pydantic or custom checks for entity rules (team in season, player in squad). Return valid / error / duplicate / will-update counts and a per-row error list (`row`, `column`, `code`, `message`, `value`), downloadable as CSV.
  4. Commit: transactional upsert of valid rows with a per-row outcome for audit and rollback.
- Idempotent upsert: natural-key unique indexes — fixture `(competition_id, season_id, home_team_id, away_team_id, kickoff_date)`; player-match line `(fixture_id, player_id, stat_source)`; crosswalk `(external_source, external_id)` — with SQLite `INSERT ... ON CONFLICT DO UPDATE`. A `row_hash` lets an unchanged re-import report "identical, skipped". Fuzzy name matches should be suggestions for review, never silent merges.
- A custom four-step wizard fits hand-written CSS better than any importer here.

### Gaps
- HelloCSV's Excel support and accessibility not verified. No open football project implements row-level errors with duplicate preview.

---

## 6. Football data model and interchange standards; a versioned "approved event JSON"

### Takeaway
No single open standard is adopted, but four references suffice: kloppy's vendor-independent model (BSD-3-Clause), SPADL (a 12-attribute action tuple), the Common Data Format v1.0.0 (Anzer et al., 2025, with an MIT JSON Schema validator) and StatsBomb's open-data layout. The Reep register (CC0) is the only open cross-provider ID crosswalk found.

### Cited Findings
- **kloppy** (BSD-3-Clause; v3.19.1, 2026-10-03; 572 stars) — [docs](https://kloppy.pysport.org/), [GitHub](https://github.com/PySport/kloppy): `EventDataset`, `TrackingDataset`, `CodeDataset`; metadata for Team, Player, Position, Period, Orientation, CoordinateSystem, Provider, Score. Event types: Generic, Pass, Shot, Take on, Carry, Clearance, Interception, Duel, Substitution, Card, Player on/off, Recovery, Miscontrol, Ball out, Foul committed, Goalkeeper, Pressure, Formation change. Providers include StatsBomb, Wyscout, Opta, Sportscode, Metrica, SkillCorner, TRACAB, SecondSpectrum, Sportec, StatsPerform. Exports to Polars/pandas and SportsCode XML.
- **SPADL** (socceraction, MIT; v1.5.3, 2024-08-15; PyPI requires Python `<3.13,>=3.9`) — [SPADL docs](https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html): attributes `game_id, period_id, seconds, player, team, start_x, start_y, end_x, end_y, action_type, result, bodypart`; 22 action types; "uniform field of 105m x 68m", origin bottom-left. Note: the prototype runs Python 3.12, inside socceraction's supported range.
- **Common Data Format:** arXiv:2505.15820 (Anzer et al.), v4 20 Aug 2025: "The CDF specifies a minimal schema for five types of match data: match sheet data, video footage, event data, tracking data, and match meta data." Version 1.0.0 — [arXiv](https://arxiv.org/abs/2505.15820). Tooling: `common-data-format-validator`, MIT, PyPI 0.1.1 (2026-09-09) — [GitHub](https://github.com/UnravelSports/common-data-format-validator).
- **FIFA EPTS format:** tracking-data standard with XML metadata and a line-parseable raw file — [FIFA](https://inside.fifa.com/en/innovation/standards/epts/research-development-epts-standard-data-format).
- **StatsBomb open data** (`hudl/open-data`; 3,825 stars): layout `competitions.json`, `matches/<competition_id>/<season_id>.json`, `events/<match_id>.json`, `lineups/<match_id>.json`, `three-sixty/<match_id>.json`; terms: "please state the data source as StatsBomb and use our logo"; licence is a `LICENSE.pdf`, not an OSI licence — [GitHub](https://github.com/hudl/open-data).
- **Reep register** (CC0-1.0; 246 stars) — [withqwerty/reep](https://github.com/withqwerty/reep): "Maps player, team, and coach identities across Transfermarkt, FBref, UEFA, Sofascore, and 25+ data providers"; 444,707 people, 45,337 teams, 212 competitions, 1,200 seasons; v0 frozen (last CSV 21 June 2026), v1 at reep.football. No match-level register.
- **Other open sources:** openfootball/football.json (CC0-1.0); transfermarkt-datasets (CC0-1.0); soccerdata scrapers (Apache-2.0); worldfootballR is archived.
- **OpenSTARLab** (arXiv:2502.02785): a "Unified and Integrated Event Data (UIED)" format; repos Apache-2.0 and small — [arXiv](https://arxiv.org/abs/2502.02785v1).
- **Tagging export precedent:** PlayTrace exports JSON, CSV and XML with four time references per event — [PlayTrace](https://github.com/MrValtancoli/PlayTrace-App).

### Inferences
- Stable entities: `competition` → `season` → `team` → `squad_membership` (team × season × player, shirt number, position, dates) → `player`; `fixture` → `lineup` → `event`. Keep provider IDs in a crosswalk table `(entity_type, internal_id, source, source_id, verified_by, verified_at)`.
- Proposed approved-event JSON:
  - Envelope: `schema`, `schema_version` (semver), `generated_at`, `generator`, `fixture`, `coordinate_system` (metres, 105×68, origin, attacking-direction rule), `periods[]`, `teams[]` with `players[]`, `provenance` (video source, model versions, reviewers).
  - Event: `event_id` (UUID), `period_id`, `clock_seconds`, `video_timestamp`, `team_id`, `player_id`, `type` (kloppy/SPADL-aligned), `result`, start/end coordinates, `body_part`, `qualifiers{}`, `related_event_ids[]`, and a `review` block (`status`, `approved_by`, `approved_at`, `source: model|manual`, `confidence`, `candidate_id`).
  - Publish a JSON Schema per version and validate every export in tests. Keep a SPADL/kloppy converter.
- Full CDF conformance can be a second export target once the validator's schemas are reviewed.

### Gaps
- CDF field-level details not verified. No open crosswalk for match identifiers. StatsBomb's `LICENSE.pdf` not read by this researcher (the event-data researcher did read it).

---

## 7. Notifications (in-app and email)

### Takeaway
A small FastAPI app needs no notification platform: a `notifications` table, an SSE or polling endpoint, and SMTP via fastapi-mail or aiosmtplib (both MIT) are enough. FastAPI now ships SSE natively (`fastapi.sse`, added in 0.135.0). Novu is MIT at its core but a multi-service platform with proprietary enterprise packages.

### Cited Findings

| Option | Licence | State (V) | Role | Source |
|---|---|---|---|---|
| FastAPI native SSE | MIT | Added in 0.135.0; `EventSourceResponse`, `ServerSentEvent` | In-app live updates, no extra dependency | [Tutorial](https://fastapi.tiangolo.com/tutorial/server-sent-events/) |
| sse-starlette | BSD-3-Clause | v3.7.0, 2026-11-07 | SSE for older FastAPI | [GitHub](https://github.com/sysid/sse-starlette) |
| fastapi-mail | MIT | v1.6.8, 2026-08-22 | SMTP with templates and attachments | [GitHub](https://github.com/sabuhish/fastapi-mail) |
| aiosmtplib | MIT | 5.1.3, 2026-09-08 | Minimal async SMTP client | [PyPI](https://pypi.org/project/aiosmtplib/) |
| Apprise | BSD-2-Clause | v2.2.0, 2026-11-08 | One API to many channels | [GitHub](https://github.com/caronc/apprise) |
| Novu | MIT outside enterprise directories | v3.20.0, 2026-11-16 | Full notification infrastructure (heavy) | [GitHub](https://github.com/novuhq/novu) |

### Inferences
- Minimal design: tables `notification` (`user_id`, `type`, `title`, `body`, `link`, `entity_type/id`, `created_at`, `read_at`, `dedupe_key`), `notification_preference`, `email_outbox`. Types: `analysis_completed`, `review_requested`, `review_overdue`, `workload_flag`, `report_ready`, `import_finished`. Write the notification and outbox rows in the same transaction as the state change; a background worker drains the outbox with retries. In-app: unread badge and list, SSE with 30–60 s polling fallback. Workload/medical emails should carry no health detail, only a link behind login.

---

## 8. Roles and permissions

### Takeaway
For a single-club deployment, in-app authentication (Authlib/PyJWT + pwdlib) with a small role-and-permission table enforced through FastAPI dependencies is enough; add pycasbin (Apache-2.0) only if policies must be data-driven. fastapi-users is in maintenance mode and Oso's open-source library is archived. For SSO, Keycloak (Apache-2.0) or authentik (MIT core) are the open options; ZITADEL is AGPL-3.0.

### Cited Findings

| Option | Licence | State (V) | Notes | Source |
|---|---|---|---|---|
| fastapi-users | MIT | v15.0.5, 2026-03-27; "maintenance mode… no new features" | Auth routes, OAuth2; no RBAC beyond superuser (K) | [GitHub](https://github.com/fastapi-users/fastapi-users) |
| Authlib | BSD-3-Clause | v1.9.9, 2026-11-17 | OAuth/OIDC, JOSE | [GitHub](https://github.com/authlib/authlib) |
| PyJWT | MIT | 2.15.1, 2026-09-28 | JWT | [PyPI](https://pypi.org/project/PyJWT/) |
| pycasbin | Apache-2.0 | v2.8.0, 2026-02-02 | ACL/RBAC/ABAC policy engine | [GitHub](https://github.com/apache/casbin-pycasbin) |
| Oso (OSS library) | Apache-2.0 | **Archived** | Do not adopt | [GitHub](https://github.com/osohq/oso) |
| Keycloak | Apache-2.0 | 26.8.2, 2026-11-05 | Full IdP; Java service | [GitHub](https://github.com/keycloak/keycloak) |
| authentik | MIT core | 2026.8.5, 2026-10-21 | Python IdP | [GitHub](https://github.com/goauthentik/authentik) |
| ZITADEL | **AGPL-3.0** | v4.20.4, 2026-11-17 | IdP | [GitHub](https://github.com/zitadel/zitadel) |

### Inferences
- Recommended: own the user table; hash with pwdlib; session cookie or short-lived JWT; optional OIDC via Authlib later. Authorise by permission strings (`events.review`, `reports.export`, `imports.commit`, `scouting.read`, `workload.read`, `medical.read`, `medical.write`, `users.manage`) through a `require()` dependency, scoped by a `membership(user_id, team_id, role)` table.
- Default roles: administrator (everything except medical by default), analyst (events, stats, reports, imports), scout (scouting data and player reports), coach (read reports, own-squad workload summaries), performance staff (workload read/write), medical staff (medical read/write).
- Medical/workload data: separate tables and permission, default-deny, field-level redaction, excluded from general exports and email bodies, an audit row for every read and export.

### Gaps
- Legal framing for health data not cited to a fetched source; confirm against the applicable law (for South Africa, POPIA — not researched).

---

## 9. Complete open-source football analytics platforms, dashboards and video-tagging tools

### Takeaway
No mature open-source equivalent of Sportscode, Nacsport or Wyscout exists. LongoMatch's open-source edition is no longer offered or maintained, Kinovea (GPL-2.0) is a generic motion-analysis desktop tool, and open football "platforms" on GitHub are almost all single-author projects under 100 stars. The reusable assets are libraries, not applications.

### Cited Findings

**Video tagging / analysis tools**

| Project | Licence | Stars / activity (V) | Stack and features | Source |
|---|---|---|---|---|
| LongoMatch (OSS edition) | GPLv2 (historical) | Vendor says it no longer offers or maintains the open-source version | C#/Mono + GStreamer desktop (K) | [Support article](https://support.longomatch.com/en/support/solutions/articles/103000249610-where-can-i-find-longomatch-opensource-) |
| Kinovea | GPL-2.0 | 517 stars; pushed 2026-11-12 | Windows .NET motion analysis; not event tagging or reporting | [GitHub](https://github.com/Kinovea/Kinovea) |
| PlayTrace | MIT | 3 stars; pushed 2026-11-17 | Expo/React Native live tagging: custom tag buttons, JSON/CSV/XML export, offline | [GitHub](https://github.com/MrValtancoli/PlayTrace-App) |
| Touchline | **AGPL-3.0** | 0 stars; pushed 2026-10-13 | Browser-only: tag over video → match reports (xG race, shot map, possession by thirds, field tilt), ratings, season tracking | [GitHub](https://github.com/ethnfshr-bot/touchline) |
| MyFootballLens | MIT | 0 stars; pushed 2026-10-07 | FastAPI + plain HTML/JS; tag, cut clips with ffmpeg, draw, stats | [GitHub](https://github.com/as13af/MyFootballLens) |

**Dashboards / platforms (small, single-author, not vetted)**

| Project | Licence | Stars / last push (V) | Stack and features | Source |
|---|---|---|---|---|
| panospen/football-tactical-dashboard | **None** | 4; 2026-04-07 | Streamlit + mplsoccer on StatsBomb data. Tabs: Overview, Passing Network, Pressing (PPDA, heatmaps), Build-up, Shots & xG, Player Roles, Player Radar; season view; similarity search; PDF match reports | [GitHub](https://github.com/panospen/football-tactical-dashboard) |
| DataKnight1/football-match-intelligence | MIT | 29; 2025-12-31 | Streamlit on SkillCorner tracking: pass networks, shot maps, momentum, heatmaps, speed zones, field tilt | [GitHub](https://github.com/DataKnight1/football-match-intelligence) |
| mostafasaad91/Football-Match-Analysis | MIT | 7; 2026-11-10 | WhoScored URL → visuals, PDF, player profiles | [GitHub](https://github.com/mostafasaad91/Football-Match-Analysis) |
| DanielTomaro13/WyScout-Tool | MIT | 0; 2026-09-21 | Streamlit: PIN gate, CSV upload with column matching, PDF/PNG report | [GitHub](https://github.com/DanielTomaro13/WyScout-Tool) |
| themanoj-025/Statlas | **AGPL-3.0** | 3; 2026-11-16 | Next.js 16 + FastAPI; ingestion adapters, metric registry, archetypes, scouting workspaces, verified reports | [GitHub](https://github.com/themanoj-025/Statlas) |
| KikeRev/smart_scout_app | MIT | 3; 2025-11-06 | Scouting: ratings, radar/pizza comparisons, PDF reports | [GitHub](https://github.com/KikeRev/smart_scout_app) |
| talal-atiq/scout-reports | **None** | 3; 2026-04-27 | TypeScript: pizza chart, xT map, touch heatmap, shot map | [GitHub](https://github.com/talal-atiq/scout-reports) |
| tom-draper/pldashboard | MIT | 51; 2026-10-02 | SvelteKit + TypeScript + Plotly + Python/MongoDB; hosted | [GitHub](https://github.com/tom-draper/pldashboard) |
| jakeyk11/football-data-analytics | Apache-2.0 | 356; 2025-03-20 | Python analysis/visualisation scripts | [GitHub](https://github.com/jakeyk11/football-data-analytics) |

**Libraries and learning resources:** databallpy (MIT, v0.8.1, 2026-08-11), floodlight (MIT, 1.2.0, 2026-05-04), penaltyblog (MIT), roboflow/sports (MIT, 5,506 stars), tryolabs/soccer-video-analytics (MIT, last push 2023), SoccerNet/sn-gamestate (**GPL-3.0**), abdullahtarek/football_analysis (**no licence**, reference only), LaurieOnTracking (MIT, last push 2022), eddwebster/football_analytics (**no licence**, largest curated resource list), devinpleuler/analytics-handbook (MIT).

**Commercial reference points**
- Hudl Sportscode: interactive dashboards with shot maps, heat maps and momentum charts "all linked direct to video" — [Hudl blog](https://www.hudl.com/blog/limitless-dashboard-functionality-with-hudl-sportscode).
- Hudl Wyscout: "Current Season Report, Match Day Report, Team Report", selectable stat groups, PDF from a preview — [guide](https://www.hudl.com/support/wyscout/guides/analysis-reports).
- Nacsport: user-configurable dashboards — [AnalysisPro KB](https://support.analysispro.com/kb/article/15/dashboards).

### Inferences
- Nothing to adopt wholesale. Statlas is the closest architectural match but is AGPL-3.0 with three stars; read it for ideas, do not copy code.
- Best ideas per project: panospen's tabs as a match-report section list; Touchline's tag → report → season loop; PlayTrace's export fields; WyScout-Tool's import profiles.
- Commercial features to match: charts linked back to video clips (Sportscode); fixed report types with selectable stat groups and PDF preview (Wyscout); configurable dashboard grids (Nacsport); a dense multi-page PDF (Driblab).
- Licence hygiene: several popular repositories have no licence (eddwebster, abdullahtarek, panospen, talal-atiq), so their code cannot be reused. GPL/AGPL applies to Kinovea, sn-gamestate, Statlas and Touchline.

### Gaps
- "ScoutLab" could not be identified as an open-source project. Feature details for Metrica Play, Veo, Spiideo and Catapult Pro Video not obtained from vendor documentation. None of the small projects' code was run or audited.

---

## 10. What professional match reports and opposition/scouting reports contain

### Takeaway
Public sources agree on a stable core, though no industry-standard template exists. A post-match report runs: summary statistics → xG/momentum timeline → shot map → pass network and progression → defensive actions/pressing → set pieces → player tables. An opposition report runs: overview → formation and shape → build-up → pressing and defensive structure → transitions → set pieces → key players → weaknesses → game plan.

### Cited Findings
- **Driblab match report PDF:** 16 pages. Page 1 has 25 statistics grouped as offensive, defensive and contribution, plus a match-flow chart; then player tables with a per-match score; then 13 pitch-map pages: shot maps with xG, key-pass maps with xA, goalkeeper save maps — [Driblab](https://www.driblab.com/blog/driblabpro-match-report-pdf).
- **Wyscout:** Current Season, Match Day and Team reports, filterable by attacking/defending/passing — [guide](https://www.hudl.com/support/wyscout/guides/analysis-reports).
- **Practitioner post-match sets:** shot maps, xT, pass networks, xG timelines, defensive duels, average positions — [Total Football Analysis](https://totalfootballanalysis.com/match-analysis/uefa-champions-league-2021-22-malmo-vs-chelsea-post-match-data-viz-and-stats).
- **Pass-network conventions:** successful passes before the first substitution, or the longest unchanged-line-up spell — [Soccerment](https://stppa.soccerment.com/game_analysis).
- **Opposition reports:** a commercial dossier lists "Formation and shape analysis with frequency breakdown; Player-by-player threat assessment with key stats; Set piece patterns - corners, free kicks, throw-ins; Pressing triggers and build-up tendencies; Substitution patterns and impact analysis" — [footballanalytics.ai](https://footballanalytics.ai/reports/opposition-dossier). A published example has: introduction, statistics, playing style, formation, crossing analysis, weaknesses, set pieces, conclusion — [Motherwell FC report](https://de.slideshare.net/MaxRogers2/motherwell-fc-analysis-report).
- **Player/scouting reports in open projects:** percentile pizza across 14+ metrics, xT map, touch heatmap, season shot map — [scout-reports](https://github.com/talal-atiq/scout-reports).

### Inferences
Default templates to ship (section → chart type):

1. **Post-match team report:** header (fixture, score, line-ups); key statistics home vs away (comparison bars: goals, xG, shots, possession, pass accuracy, final-third entries, PPDA, duels, corners, cards); xG race / momentum timeline; shot map per team plus shot table; pass network per team; progression; defensive actions and pressing heatmap; set pieces; player table and analyst notes with video links.
2. **Opposition report:** overview (form, typical XI, formation frequency); in possession; out of possession (pressing height, block, PPDA trend); transitions; set pieces; key players (pizza/radar, heatmap); substitution patterns; strengths, weaknesses and game plan.
3. **Player/scouting report:** profile, percentile pizza or radar vs position peers, trend lines for 3–5 metrics, heatmap, shot map, match log, recommendation.
4. **Squad/workload report (restricted):** minutes and availability, load trend lines, flags — only for users with workload/medical permissions.

- xG, xT, PPDA and field tilt need event locations; templates should hide a section with a note when the approved events do not yet cover it.

### Gaps
- No club-published official template was found; sources are vendor pages, template sellers, blogs and small open projects. No authoritative source for performance/medical report contents; the workload template is an inference to validate with practitioners.
