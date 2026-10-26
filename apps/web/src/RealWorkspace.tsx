import { useEffect, useRef, useState } from "react";
import { ArrowDownToLine, ArrowRight, Check, ChevronRight, Clock3, Film, FolderOpen, LayoutDashboard, Play, Search, SlidersHorizontal, Video } from "lucide-react";
import type { AnalysisResult } from "./main";
import { AnalyticsDashboard } from "./AnalyticsDashboard";
import { VideoMeasurementsPanel } from "./VideoMeasurements";
import { ApprovedSummary, ReviewPanel, type Review } from "./ReviewPanel";
import { VideoInsightsDownloads, VideoInsightsPanel } from "./VideoInsights";
import { SequenceContextPanel } from "./SequenceContextPanel";

export type LibraryRun = { run_id: string; engine?: string; created_at: string; metadata: AnalysisResult["metadata"]; metrics: { data_quality?: Record<string, string | number> }; summary?: { analyzed_frames?: number | null; temporary_tracks?: number | null; mean_visible_players?: number | null; ball_candidate_coverage_pct?: number | null; selected_ball_candidate_coverage_pct?: number | null } };
type View = "Overview" | "Video review" | "Analytics" | "Library" | "Exports";
type Props = { result: AnalysisResult | null; runs: LibraryRun[]; onOpen: (id: string) => Promise<void>; onSample: () => void; onUpload: () => void; uploadDisabled: boolean; api: string; authHeaders: () => Record<string, string>; selectionLoading: string; libraryLoading: boolean; libraryError: string; onRefresh: () => void; onRefreshResult: (runId: string) => Promise<void>; openRequest: number };
const fmt = (value: number | null | undefined, digits = 1) => value == null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
const clock = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
const tabs: { title: View; icon: typeof Video }[] = [{ title: "Overview", icon: LayoutDashboard }, { title: "Video review", icon: Video }, { title: "Analytics", icon: SlidersHorizontal }, { title: "Library", icon: FolderOpen }, { title: "Exports", icon: ArrowDownToLine }];

export function RealWorkspace({ result, runs, onOpen, onSample, onUpload, uploadDisabled, api, authHeaders, selectionLoading, libraryLoading, libraryError, onRefresh, onRefreshResult, openRequest }: Props) {
  const [view, setView] = useState<View>("Overview");
  const [savedReview, setSavedReview] = useState<Review | null>(null);
  const [reviewUnsaved, setReviewUnsaved] = useState(false);
  const [analyticsSource, setAnalyticsSource] = useState<"model" | "reviewed">("model");
  const [libraryQuery, setLibraryQuery] = useState("");
  const video = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<number | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const m = result?.metrics.video_measurements;
  const summary = result?.detection_summary;
  const saved = savedReview?.run_id === result?.run_id ? savedReview : null;
  useEffect(() => { pendingSeek.current = null; }, [result?.run_id]);
  useEffect(() => { setView("Overview"); }, [openRequest]);
  useEffect(() => { if (view !== "Overview" && view !== "Video review") video.current?.pause(); }, [view]);
  const media = (path: string) => /^https?:\/\//.test(path) ? path : `${api}${path}`;
  const chooseView = (next: View) => { setView(next); window.requestAnimationFrame(() => stage.current?.scrollIntoView({ block: "start", behavior: "instant" })); };
  const seek = (seconds: number) => {
    if (!result) return;
    pendingSeek.current = Math.max(0, Math.min(seconds, result.metadata.duration_seconds)); setView("Video review");
    if (video.current && video.current.readyState >= 1) { video.current.currentTime = pendingSeek.current; pendingSeek.current = null; }
    window.requestAnimationFrame(() => video.current?.scrollIntoView({ block: "start", behavior: "smooth" }));
  };
  const matchingRuns = runs.filter(run => `${run.metadata.filename} ${run.run_id}`.toLowerCase().includes(libraryQuery.trim().toLowerCase()));
  const open = async (id: string) => { pendingSeek.current = null; await onOpen(id); setView("Overview"); };
  const downloadReview = () => {
    if (!saved) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(saved, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `${saved.run_id}-saved-review-revision-${saved.revision}.json`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return <div className="studio" ref={stage}>
    <div className="studio-context"><div className="studio-context__clip"><span className="studio-clip-icon"><Film size={22} /></span><div><p className="eyebrow">{result ? "Selected footage" : "Your footage workspace"}</p><h2>{result?.metadata.filename ?? "Start with a match clip"}</h2><div className="studio-context__meta">{result ? <><span><Clock3 size={13} /> {clock(result.metadata.duration_seconds)}</span><span>{result.metadata.width} × {result.metadata.height}</span><span className="studio-status approved"><Check size={12} /> Analysis ready</span></> : <span>Upload once. Watch, measure and review.</span>}</div></div></div>
      <label className="studio-clip-picker">Switch clip<select aria-label="Switch selected clip" value={result?.run_id ?? ""} onChange={e => e.target.value && open(e.target.value)}><option value="" disabled>{libraryLoading ? "Loading clips…" : "Choose a saved clip"}</option>{runs.map(run => <option key={run.run_id} value={run.run_id}>{run.metadata.filename} · {clock(run.metadata.duration_seconds)}{run.engine && run.engine !== "yolo_bytetrack" ? " · Synthetic demo" : ""}</option>)}</select></label>
    </div>
    <nav className="studio-nav" aria-label="Analysis workspace">{tabs.map(({ title, icon: Icon }) => <button type="button" key={title} aria-current={view === title ? "page" : undefined} onClick={() => chooseView(title)}><Icon size={17} /><span>{title}</span></button>)}</nav>
    {selectionLoading && <p className="studio-loading" role="status"><span className="studio-loading__dot" />{selectionLoading} Your current clip stays available until the selection is ready.</p>}
    {libraryError && <p className="studio-error" role="alert">Could not load saved clips. {libraryError} <button className="ghost" onClick={onRefresh}>Retry library</button></p>}

    {!result && view !== "Library" && <section className="studio-empty"><span className="studio-empty__icon"><Video size={30} /></span><p className="eyebrow">Evidence starts with the video</p><h2>Your next match review starts here.</h2><p>Upload footage to detect visible players and ball candidates. Watch the evidence, add human-reviewed events and compare only what has been verified.</p><div className="studio-empty__actions"><button className="primary" onClick={onUpload} disabled={uploadDisabled}>Upload video <ArrowRight size={17} /></button><button className="ghost" onClick={onSample}>Explore the full 30s sample</button></div><small>Hosted test clips: up to 2 minutes / 250 MiB. Target: a 45-minute match half.</small></section>}

    {result && <>
      <section hidden={view !== "Overview"} className="studio-headlines" aria-label="Selected clip observations">
        <article><span>Frames analyzed</span><strong>{fmt(summary?.analyzed_frames, 0)}</strong><small>{m?.analyzed_frame_pct == null ? "Source coverage not recorded" : `${fmt(m.analyzed_frame_pct)}% of source frames`}</small></article>
        <article><span>Ball candidate coverage</span><strong>{m?.ball_coverage_pct == null ? "—" : `${fmt(m.ball_coverage_pct)}%`}</strong><small>Observed candidates · not accuracy</small></article>
        <article><span>Mean visible players</span><strong>{fmt(m?.mean_visible_players)}</strong><small>Detections per analyzed frame</small></article>
        <article className="studio-headlines__review"><span>Approved event records</span><strong>{saved ? saved.summary.approved_events > 0 ? fmt(saved.summary.approved_events, 0) : "None approved" : "Loading…"}</strong><small>Saved human annotations only</small></article>
      </section>

      <div className={`studio-evidence-layout ${view === "Video review" ? "is-review" : ""}`} hidden={view !== "Overview" && view !== "Video review"}>
        <div className="studio-evidence" id="analysis">
          <header><div><p className="eyebrow">Video evidence</p><h3>Watch what was observed</h3></div><span className="studio-status model">Model output · unreviewed</span></header>
          {result.files.annotated_video ? <video key={result.run_id} ref={video} controls preload="metadata" src={media(result.files.annotated_video)} onLoadedMetadata={() => { if (pendingSeek.current !== null && video.current) { video.current.currentTime = pendingSeek.current; pendingSeek.current = null; } }} /> : <p className="studio-empty-media">An annotated video is not available for this run.</p>}
          <div className="studio-video-footer"><span><span className="studio-live-dot" /> Full clip · {clock(result.metadata.duration_seconds)}</span><span>Boxes are predictions. Review against the footage.</span></div>
          {view === "Video review" && <div className="studio-video-controls"><button className="ghost" onClick={() => seek(Math.max(0, (video.current?.currentTime ?? 0) - 5))}>−5 seconds</button><button className="ghost" onClick={() => seek((video.current?.currentTime ?? 0) + 5)}>+5 seconds</button><button className="ghost" onClick={() => chooseView("Analytics")}>See observations <ArrowRight size={15} /></button></div>}
          <details className="studio-technical"><summary>Source & model details</summary><p>Source: {result.metadata.width} × {result.metadata.height} · {fmt(result.metadata.fps)} fps · {fmt(result.metadata.frame_count, 0)} frames. Analysis sampling: {fmt(summary?.sample_fps)} fps.</p><p>Model: {summary?.model ?? "Unknown"} · Tracker: {summary?.tracker ?? "Not recorded"} · Detector input: {summary?.image_size ?? "Unknown"} px · Device: {summary?.device ?? "Unknown"}.</p><p>{result.report.summary}</p>{result.warnings?.length ? <ul>{result.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul> : null}</details>
        </div>
        <aside className="studio-next" hidden={view !== "Overview"}><p className="eyebrow">Your analysis path</p><h3>From footage to evidence.</h3><p>The model gets you started. Your review makes the football record.</p>
          <button onClick={() => chooseView("Video review")}><span className="studio-next__number">01</span><span><strong>Watch & review events</strong><small>Add timestamped annotations. Approve explicitly.</small></span><ChevronRight size={17} /></button>
          <button onClick={() => chooseView("Analytics")}><span className="studio-next__number">02</span><span><strong>Explore observations</strong><small>Ball visibility, image occupancy and temporary tracks.</small></span><ChevronRight size={17} /></button>
          <button onClick={() => chooseView("Exports")}><span className="studio-next__number">03</span><span><strong>Take the evidence with you</strong><small>Annotated video, model data and saved review JSON.</small></span><ChevronRight size={17} /></button>
          <div className="studio-next__scope"><span className="studio-status draft">Review scope matters</span><p>{saved && saved.summary.approved_events > 0 ? `${saved.summary.approved_events} saved approved events. Reviewed coverage: ${fmt(saved.summary.reviewed_seconds)}s (${fmt(saved.summary.reviewed_video_pct)}% of clip).` : "No saved approved events yet. Passes, shots and possession are not inferred from detection counts."}</p></div>
        </aside>
        <div hidden={view !== "Video review"} className="studio-review-controller"><ReviewPanel key={result.run_id} runId={result.run_id} duration={result.metadata.duration_seconds} api={api} authHeaders={authHeaders} onSaved={setSavedReview} onUnsaved={setReviewUnsaved} onSeek={seek} getVideoTime={() => video.current?.currentTime ?? 0} hideSummary /></div>
      </div>

      <section hidden={view !== "Analytics"} className="studio-analytics" id="football-analytics" aria-label="Clip analytics">
        <header className="studio-section-header studio-analytics-heading"><h2>Clip analytics</h2><div className="studio-source-switch" role="group" aria-label="Analytics data source"><button aria-pressed={analyticsSource === "model"} onClick={() => setAnalyticsSource("model")}>Model observations <span>Unreviewed</span></button><button aria-pressed={analyticsSource === "reviewed"} onClick={() => setAnalyticsSource("reviewed")}>Approved records <span>{saved ? saved.summary.approved_events > 0 ? `${saved.summary.approved_events} saved events` : "Awaiting review" : "Loading…"}</span></button></div><button className="ghost" onClick={() => chooseView("Video review")}>{saved && !saved.summary.approved_events ? "Start event review" : "Review footage"} <ArrowRight size={16} /></button></header>

        <div className="studio-approved" hidden={analyticsSource !== "reviewed"}><span className="studio-status approved">Human review</span>{saved ? saved.summary.approved_events > 0 || saved.possession_intervals.some(item => item.review_status === "approved") ? <ApprovedSummary saved={saved} unsaved={reviewUnsaved} /> : <div className="studio-approved-empty"><div><h3>No approved event records yet</h3><p>Review footage and approve annotations to build scoped football statistics. {saved.summary.reviewed_seconds > 0 && `Reviewed footage: ${fmt(saved.summary.reviewed_seconds)}s (${fmt(saved.summary.reviewed_video_pct)}%).`}</p>{reviewUnsaved && <p className="studio-unsaved-note">Unsaved annotations and unfinished forms are excluded.</p>}</div><button className="primary" onClick={() => chooseView("Video review")}>Start event review <ArrowRight size={16} /></button></div> : <p role="status">Loading saved review records…</p>}</div>
        <div key={result.run_id} hidden={analyticsSource !== "model"}><VideoInsightsPanel insights={result.metrics.video_insights} runId={result.run_id} onSeek={seek} /><SequenceContextPanel runId={result.run_id} sourceSha={result.metadata.sha256} duration={result.metadata.duration_seconds} kits={[...new Set([...(result.metrics.video_insights?.kits.map(kit => kit.kit) ?? []), ...(result.metrics.video_measurements?.provisional_kit_analytics?.map(kit => kit.kit) ?? [])])]} api={api} authHeaders={authHeaders} onRefresh={onRefreshResult} onSeek={seek} /><details className="insights-observation-detail"><summary>Observation detail <span>Frame coverage, image occupancy & sampling</span></summary>{m ? <VideoMeasurementsPanel measurements={m} onSeek={seek} showTracks={false} /> : <section className="studio-panel"><h3>Model observations</h3><p>Extended camera-image measurements are not present in this older run. Its available detections can still be explored below and downloaded.</p></section>}</details>
        <TrackExplorer result={result} onSeek={seek} /></div>
        <details className="studio-scope"><summary><span>Analytics scope <small>All 7 areas · 42 required metrics</small></span><span>Explore requirements</span></summary><AnalyticsDashboard filename={result.metadata.filename} hasDetections status="complete" onReview={() => chooseView("Video review")} /></details>
      </section>

      <section hidden={view !== "Exports"} className="studio-exports"><header className="studio-section-header"><div><p className="eyebrow">Keep the source with the result</p><h2>Export your evidence</h2><p>Model output and reviewer-approved records are separate files.</p></div></header>
        <div className="studio-export-list">{[
          { label: "Annotated video", description: "Full-duration MP4 with unreviewed model detections.", path: result.files.annotated_video, type: "MP4" },
          { label: "Tracking observations", description: "Timestamped raw model observations and temporary track identities.", path: result.files.tracking_json, type: "JSON" },
          { label: "Measurement data", description: "Computed model measurements with their scope and definitions.", path: result.files.metrics_json, type: "JSON" },
          { label: "Full analysis", description: "Analysis manifest, model measurements, limits and provenance.", path: result.files.manifest_json, type: "JSON" },
        ].map(item => <article key={item.label}><span className="studio-filetype">{item.type}</span><div><h3>{item.label}</h3><p>{item.description}</p></div>{item.path ? <a href={media(item.path)} target="_blank" rel="noreferrer" aria-label={`Download ${item.label}`}><ArrowDownToLine size={18} /> Download</a> : <span className="studio-export-unavailable">Unavailable in this run</span>}</article>)}
          <article><span className="studio-filetype reviewed">JSON</span><div><h3>Saved human review</h3><p>Last server-saved revision {saved?.revision ?? "—"}. Includes approval states and saved summary; excludes unsaved edits and unfinished forms.</p></div><button className="ghost" onClick={downloadReview} disabled={!saved}><ArrowDownToLine size={18} /> Saved review JSON</button></article>
          {result.metrics.video_insights && <article><span className="studio-filetype">DATA</span><div><h3>Automatic video insights</h3><p>Observed ball paths, kit shape windows, supported control estimates and possible events. Unreviewed, bound to this selected clip.</p></div><VideoInsightsDownloads insights={result.metrics.video_insights} runId={result.run_id} /></article>}
        </div><p className="studio-footnote">Downloads use the existing protected/signed file paths. Missing files are shown as unavailable.</p>
      </section>
    </>}

    <section hidden={view !== "Library"} className="studio-library"><header className="studio-section-header"><div><p className="eyebrow">Your footage, ready to revisit</p><h2>Clip library <span className="studio-count">{runs.length}</span></h2><p>Opening a saved result uses existing analysis. It does not reprocess the video.</p></div><button className="ghost" onClick={onRefresh} disabled={libraryLoading}>Refresh library</button></header><label className="studio-library-search"><Search size={17} /><span className="studio-sr-only">Find a saved clip</span><input type="search" value={libraryQuery} onChange={e => setLibraryQuery(e.target.value)} placeholder="Search clip names" /></label>
      {libraryLoading && !runs.length && <p role="status">Loading your clip library…</p>}
      {!libraryLoading && !matchingRuns.length && <div className="studio-panel" role="status"><h3>{runs.length ? "No matching clips" : "Your library is ready for its first clip"}</h3><p>{runs.length ? "Try a different filename." : "Upload footage or open the sample to explore a completed analysis."}</p></div>}
      <div className="studio-library-list">{matchingRuns.map(run => <button key={run.run_id} className={`studio-library-item ${run.run_id === result?.run_id ? "selected" : ""}`} onClick={() => open(run.run_id)}><span className="studio-library-icon"><Film size={22} /></span><span><strong>{run.metadata.filename}</strong><small>{clock(run.metadata.duration_seconds)} · {run.metadata.width} × {run.metadata.height} · {new Date(run.created_at).toLocaleDateString()}</small></span><span className="studio-library-type">{run.run_id === result?.run_id ? "Selected" : run.engine === "yolo_bytetrack" || run.metrics.data_quality?.mode === "yolo_bytetrack" ? "Model observations" : "Synthetic demo"}</span><ChevronRight size={18} /></button>)}</div>
    </section>
    <footer className="studio-footer"><span>LAAS · Footage to evidence</span><span>Hosted test: 2 min / 250 MiB <span aria-hidden="true">·</span> Match-half target: 45 min</span></footer>
  </div>;
}

function TrackExplorer({ result, onSeek }: { result: AnalysisResult; onSeek: (seconds: number) => void }) {
  const [query, setQuery] = useState(""); const [sort, setSort] = useState("observations"); const [page, setPage] = useState(0);
  const measured = new Map(result.metrics.video_measurements?.tracks.map(track => [track.track_id, track]) ?? []);
  const matches = (result.detection_summary?.tracks ?? []).filter(track => `${track.track_id} ${track.object_type}`.toLowerCase().includes(query.trim().toLowerCase())).sort((a, b) => sort === "first" ? a.first_seen_ms - b.first_seen_ms : sort === "near" ? (measured.get(b.track_id)?.near_ball_frames ?? 0) - (measured.get(a.track_id)?.near_ball_frames ?? 0) : b.observations - a.observations);
  const totalPages = Math.max(1, Math.ceil(matches.length / 8)); const current = Math.min(page, totalPages - 1);
  return <section className="studio-tracks studio-panel"><header className="studio-section-header"><div><p className="eyebrow">Explore model observations</p><h3>Temporary tracks</h3><p>Track IDs can split or change. They do not identify verified players.</p></div><div className="studio-track-tools"><label>Search tracks<input type="search" value={query} onChange={e => { setQuery(e.target.value); setPage(0); }} placeholder="Track ID or class" /></label><label>Sort tracks<select value={sort} onChange={e => { setSort(e.target.value); setPage(0); }}><option value="observations">Most observations</option><option value="first">First seen</option><option value="near">Near-ball frames</option></select></label></div></header>
    <div className="studio-table-scroll" tabIndex={0} role="region" aria-label="Temporary track observations table"><table><thead><tr><th scope="col">Track / class</th><th scope="col">Observed</th><th scope="col">Visibility</th><th scope="col">Image displacement</th><th scope="col">Near-ball frames</th><th scope="col">Evidence</th></tr></thead><tbody>{matches.slice(current * 8, (current + 1) * 8).map(track => { const detail = measured.get(track.track_id); return <tr key={track.track_id}><th scope="row">{track.track_id}<small>{track.object_type}</small></th><td>{detail ? `${fmt(detail.observed_seconds)}s` : `${track.observations} frames`}</td><td>{detail?.visibility_pct == null ? "—" : `${fmt(detail.visibility_pct)}%`}</td><td>{detail ? `${fmt(detail.image_displacement_px)} px` : "—"}</td><td>{fmt(detail?.near_ball_frames, 0)}</td><td><button className="ghost" onClick={() => onSeek(track.first_seen_ms / 1000)}><Play size={13} /> {fmt(track.first_seen_ms / 1000)}s</button></td></tr>; })}</tbody></table></div>
    {!matches.length && <p role="status">No tracks match this search.</p>}<div className="studio-pagination"><span>{matches.length} matching tracks · page {current + 1} of {totalPages}</span><button className="ghost" disabled={current === 0} onClick={() => setPage(current - 1)}>Previous tracks</button><button className="ghost" disabled={current >= totalPages - 1} onClick={() => setPage(current + 1)}>Next tracks</button></div><details className="studio-technical"><summary>What these track measures mean</summary><p>Observed seconds estimate sampled visible intervals, not continuous play. Image displacement is in pixels and includes camera motion and tracking error. Near-ball frames are proximity observations, not possession or touches. Full detections remain available in Exports.</p></details>
  </section>;
}
