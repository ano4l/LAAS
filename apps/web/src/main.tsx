import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Bot,
  CalendarDays,
  Clock3,
  Database,
  Gauge,
  Layers3,
  Menu,
  Play,
  Radar,
  Search,
  Shield,
  Sparkles,
  Target,
  Upload,
  Users,
  Video,
  Wand2,
} from "lucide-react";
import "./styles.css";
import { AnalyticsDashboard } from "./AnalyticsDashboard";
import type { VideoMeasurements } from "./VideoMeasurements";
import { ReviewPanel } from "./ReviewPanel";
import { RealWorkspace, type LibraryRun } from "./RealWorkspace";
import type { VideoInsights } from "./VideoInsights";

const configuredApiUrl = import.meta.env.VITE_API_URL?.trim();
const API = configuredApiUrl || (window.location.port === "5173" ? "http://127.0.0.1:8000" : window.location.origin);
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";
const REQUIRE_ACCESS_KEY = import.meta.env.VITE_REQUIRE_ACCESS_KEY === "true";
const authHeaders = (): Record<string, string> => {
  const key = sessionStorage.getItem("laas-access-key");
  return key ? { Authorization: `Bearer ${key}` } : {};
};

type Job = {
  id: string;
  status: "queued" | "running" | "complete" | "failed";
  stage: string;
  progress: number;
  created_at: string;
  source_video?: string;
  filename?: string;
  result?: AnalysisResult;
  error?: string;
};
type LibraryJob = Pick<Job, "id" | "status" | "stage" | "progress" | "created_at" | "source_video" | "error"> & { engine?: string; filename?: string; run_id?: string };

type Track = {
  frame_idx: number;
  timestamp_ms: number;
  track_id: string;
  object_type: "player" | "ball";
  team: "Home" | "Away" | null;
  x: number;
  y: number;
  confidence: number;
};

export type AnalysisResult = {
  run_id: string;
  engine?: string;
  warnings?: string[];
  detection_summary?: { tracker?: string; analyzed_frames: number; decoded_frames: number; sample_fps: number; image_size?: number; ball_processing?: string; unique_tracks: number; ball_detected_frames: number; processing_seconds: number; device: string; model: string; detections_by_class: Record<string, number>; tracks: {track_id: string; object_type: string; observations: number; first_seen_ms: number; last_seen_ms: number}[] };
  metadata: { filename: string; fps: number; frame_count: number; duration_seconds: number; width: number; height: number; sha256?: string };
  tracking: Track[];
  metrics: {
    video_insights?: VideoInsights;
    video_measurements?: VideoMeasurements;
    possession: { home_pct: number; away_pct: number; timeline: { timestamp_ms: number; team: string }[] };
    teams: Record<string, { avg_line_height: number; avg_width: number; compactness: number; distance_estimate_km: number; sprint_count_estimate: number }>;
    players?: { track_id: string; team: string; distance_m: number; max_speed_kmh: number; avg_x: number; avg_y: number; influence: number }[];
    events?: { timestamp_ms: number; type: string; team: string | null; description: string; confidence: number }[];
    data_quality: Record<string, string | number>;
  };
  phases: { phase: string; start_ms: number; end_ms: number; confidence: number }[];
  report: { title: string; summary: string; recommendations: { label: string; detail: string; confidence: number }[] };
  files: { annotated_video: string; tracking_json?: string; metrics_json?: string; report_json?: string; manifest_json?: string };
};

export type RunSummary = {
  run_id: string;
  created_at: string;
  metadata: AnalysisResult["metadata"];
  metrics: AnalysisResult["metrics"];
  phases: AnalysisResult["phases"];
  report: AnalysisResult["report"];
  files: AnalysisResult["files"];
  engine?: string;
};

type Capabilities = {
  engines: {
    fallback: { available: boolean; description: string };
    yolo: { blockers?: string[]; available: boolean; compute_provider?: "local" | "modal"; model_exists?: boolean; ultralytics_installed?: boolean; opencv_available?: boolean; model_path?: string };
  };
};

function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [activeJobError, setActiveJobError] = useState("");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [libraryRuns, setLibraryRuns] = useState<LibraryRun[]>([]);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [activeModule, setActiveModule] = useState("Overview");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const selectionEpoch = useRef(0);
  const libraryEpoch = useRef(0);
  const [selectionLoading, setSelectionLoading] = useState("");
  const [selectionFailure, setSelectionFailure] = useState("");
  const [libraryLoading, setLibraryLoading] = useState(true);
  const [libraryError, setLibraryError] = useState("");
  const [workspaceOpen, setWorkspaceOpen] = useState(0);
  const activeProcessing = activeJob?.status === "queued" || activeJob?.status === "running";

  useEffect(() => {
    if (DEMO_MODE) {
      const result = createDemoAnalysis();
      setJob(demoJob(result, "Static Vercel demo"));
      setRuns([demoRun(result)]);
      setCapabilities(demoCapabilities());
      return;
    }
    refreshRuns(true);
    request<Capabilities>("/capabilities").then(setCapabilities).catch(() => setCapabilities(null));
  }, []);

  useEffect(() => {
    if (DEMO_MODE) return;
    if (!activeJob || activeJob.status === "complete" || activeJob.status === "failed") return;
    let cancelled = false;
    let polling = false;
    const timer = window.setInterval(async () => {
      if (polling) return;
      polling = true;
      try {
        const next = await request<Job>(`/analysis-jobs/${activeJob.id}`);
        if (!cancelled) { setActiveJob({ ...next, filename: activeJob.filename }); setActiveJobError(""); setJob(current => current?.id === next.id ? next : current); }
      } catch (err) { if (!cancelled) setActiveJobError(String(err)); }
      finally { polling = false; }
    }, 900);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [activeJob?.id, activeJob?.status]);

  useEffect(() => { if (activeJob?.status === "complete") refreshRuns(); }, [activeJob?.status]);

  async function refreshRuns(selectInitial = false) {
    if (DEMO_MODE) {
      const result = createDemoAnalysis();
      setRuns([demoRun(result)]);
      setLibraryLoading(false);
      return;
    }
    const currentLibraryRequest = ++libraryEpoch.current;
    setLibraryLoading(true); setLibraryError("");
    try {
      const library = await request<{ runs: LibraryRun[]; jobs: LibraryJob[] }>("/library");
      if (currentLibraryRequest !== libraryEpoch.current) return;
      setLibraryRuns(library.runs);
      if (selectInitial && selectionEpoch.current === 0) {
        const next = library.jobs.find(item => item.status === "running" || item.status === "queued") ?? library.jobs.find(item => item.engine === "yolo_bytetrack" && item.status === "complete");
        if (next) {
          const epoch = ++selectionEpoch.current; setSelectionLoading("Loading selected clip…");
          try { const detail = await request<Job>(`/analysis-jobs/${next.id}`); if (next.status === "running" || next.status === "queued") setActiveJob({ ...detail, filename: next.filename }); if (epoch === selectionEpoch.current) setJob(detail); }
          catch (cause) { if (epoch === selectionEpoch.current) { setError(String(cause)); setSelectionFailure(next.run_id ?? "sample"); } }
          finally { if (epoch === selectionEpoch.current) setSelectionLoading(""); }
        } else { const firstReal = library.runs.find(item => item.engine === "yolo_bytetrack"); if (firstReal) await openRun(firstReal.run_id); }
      }
    } catch (cause) { if (currentLibraryRequest === libraryEpoch.current) setLibraryError(String(cause)); }
    finally { if (currentLibraryRequest === libraryEpoch.current) setLibraryLoading(false); }
  }

  async function startDemo() {
    selectionEpoch.current++;
    if (DEMO_MODE) {
      const result = createDemoAnalysis();
      setBusy(true);
      setJob({
        id: "demo-vercel-job",
        status: "running",
        stage: "Preparing demo analysis",
        progress: 0.42,
        created_at: new Date().toISOString(),
        result,
      });
      window.setTimeout(() => {
        setJob(demoJob(result, "Loaded static demo"));
        setBusy(false);
      }, 550);
      return;
    }
    setBusy(true);
    try { setJob(await request<Job>("/demo-job", { method: "POST" })); }
    catch (err) { setError(String(err)); }
    finally { setBusy(false); }
  }

  async function openRun(runId: string) {
    if (DEMO_MODE) {
      const result = createDemoAnalysis();
      setJob(demoJob(result, "Loaded static demo"));
      return;
    }
    const epoch = ++selectionEpoch.current; setSelectionLoading("Loading clip…"); setSelectionFailure(""); setError("");
    const result = await request<AnalysisResult>(`/runs/${runId}`).catch(err => { if (epoch === selectionEpoch.current) { setError(String(err)); setSelectionFailure(runId); } return null; });
    if (epoch !== selectionEpoch.current) return;
    setSelectionLoading(""); if (!result) return;
    setJob({
      id: `run-${runId}`,
      status: "complete",
      stage: "Loaded run",
      progress: 1,
      created_at: new Date().toISOString(),
      source_video: undefined,
      result,
    });
  }

  async function selectModule(moduleName: string) {
    setActiveModule(moduleName);
    if (moduleName !== "Overview" && !job?.result) {
      if (runs[0]?.run_id) {
        await openRun(runs[0].run_id);
      } else {
        await startDemo();
      }
    }
    window.setTimeout(() => {
      document.getElementById(moduleName === "Overview" ? "overview" : "analysis")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 80);
  }

  async function refreshSelectedResult(runId: string) {
    if (job?.result?.run_id !== runId) return;
    const expectedSelection = selectionEpoch.current;
    const result = await request<AnalysisResult>(`/runs/${runId}`);
    if (expectedSelection !== selectionEpoch.current) return;
    setJob(current => current?.result?.run_id === runId ? { ...current, result } : current);
  }

  async function upload(file: File) {
    if (DEMO_MODE) { setError("This is a demo. Connect the analysis API to upload your footage."); return; }
    if (busy || activeProcessing || libraryLoading) { setError("Wait for the current upload, analysis or library check to finish. You can browse existing clips while analysis runs."); return; }
    const uploadSelection = ++selectionEpoch.current; setSelectionLoading(""); setBusy(true); setError(""); setUploadPct(0);
    try {
      const body = new FormData(); body.append("file", file);
      const next = await new Promise<Job>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API}/videos/upload`);
        for (const [name, value] of Object.entries(authHeaders())) xhr.setRequestHeader(name, value);
        xhr.upload.onprogress = event => { if (event.lengthComputable) setUploadPct(Math.round(event.loaded / event.total * 100)); };
        xhr.onerror = () => reject(new Error("Cannot reach the analysis API. Check that the backend is running."));
        xhr.onload = () => {
          try { const payload = JSON.parse(xhr.responseText); if (xhr.status >= 400) reject(new Error(payload.detail || "Upload failed")); else resolve(payload); }
          catch { reject(new Error("The API returned an invalid response.")); }
        };
        xhr.send(body);
      });
      setActiveJob({ ...next, filename: file.name }); if (uploadSelection === selectionEpoch.current) setJob(next); setActiveModule("Match Review");
    } catch (err) { setError(String(err)); }
    finally { setBusy(false); setUploadPct(null); if (fileRef.current) fileRef.current.value = ""; }
  }

  async function retry() {
    if (!job) return;
    if (activeProcessing || busy) return;
    const expectedSelection = selectionEpoch.current;
    try { setError(""); const next = await request<Job>(`/analysis-jobs/${job.id}/retry`, {method: "POST"}); setActiveJob(next); if (expectedSelection === selectionEpoch.current) setJob(next); }
    catch (err) { setError(String(err)); }
  }

  async function openSample() {
    const epoch = ++selectionEpoch.current; setSelectionLoading("Opening full sample…"); setSelectionFailure(""); setError("");
    try { const sample = await request<Job>("/sample"); if (epoch === selectionEpoch.current) setJob(sample); }
    catch (err) { if (epoch === selectionEpoch.current) { setError(String(err)); setSelectionFailure("sample"); } }
    finally { if (epoch === selectionEpoch.current) setSelectionLoading(""); }
  }

  return (
    <main className={`app-shell ${!job?.result || job.result.engine === "yolo_bytetrack" ? "real-app" : ""}`}>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">LAAS / ANALYSIS WORKSPACE</p>
            <h1>Match Intelligence Studio</h1>
          </div>
          <div className="actions">
            {!DEMO_MODE && <button className="ghost" onClick={openSample} disabled={busy}>Open sample · 30s</button>}
            <input ref={fileRef} type="file" aria-label="Upload football video" accept=".mp4,.mov,.avi,.mkv,.webm,.m4v" onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])} />
            <button className="primary" onClick={() => fileRef.current?.click()} disabled={busy || activeProcessing || libraryLoading} title={activeProcessing ? "One video is processing. Browse saved results while it finishes." : libraryLoading ? "Checking existing analyses before accepting another upload." : "Upload football footage"}>
              <Upload size={18} /> {uploadPct !== null ? `Uploading ${uploadPct}%` : "Upload video"}
            </button>
            {!REQUIRE_ACCESS_KEY && <button className="ghost demo-action" onClick={startDemo} disabled={busy}>
              <Play size={18} /> Demo analysis
            </button>}
          </div>
        </header>

        {uploadPct !== null && <p className="notice" role="status">Uploading footage · {uploadPct}%. Another upload is disabled until this job finishes. You can keep browsing saved clips.</p>}
        {activeJob && <div className={`active-job-banner ${activeProcessing ? "processing" : "finished"}`} role="status"><div><strong>{activeProcessing ? "Analysis running in the background" : activeJob.status === "complete" ? "Your analysis is ready" : "Analysis needs attention"}</strong><span>{activeJob.filename ?? activeJob.result?.metadata.filename ?? "Uploaded video"} · {activeProcessing ? `${activeJob.stage} · ${Math.round(activeJob.progress * 100)}% · One analysis at a time; saved clips remain available.` : activeJob.status === "complete" ? "Open the full result to review it." : activeJob.error}</span>{activeJobError && <span>Could not refresh status. Retrying automatically. {activeJobError}</span>}</div><button className="ghost" onClick={() => { selectionEpoch.current++; setSelectionLoading(""); setJob(activeJob); setWorkspaceOpen(value => value + 1); }}>Open {activeProcessing ? "processing job" : "result"}</button>{!activeProcessing && <button className="ghost" onClick={() => setActiveJob(null)}>Dismiss status</button>}</div>}
        {error && <p className="notice error" role="alert">{error} {selectionFailure && <button className="ghost" onClick={() => selectionFailure === "sample" ? openSample() : openRun(selectionFailure)}>Retry clip selection</button>}</p>}
        {job?.error && <div className="notice error" role="alert">{job.error} {!REQUIRE_ACCESS_KEY && <button className="ghost" onClick={retry}>Retry analysis</button>}</div>}
        {!DEMO_MODE && (!job?.result || job.result.engine === "yolo_bytetrack") && <RealWorkspace result={job?.result ?? null} runs={libraryRuns} onOpen={openRun} onSample={openSample} onUpload={() => fileRef.current?.click()} uploadDisabled={busy || activeProcessing || libraryLoading} openRequest={workspaceOpen} api={API} authHeaders={authHeaders} selectionLoading={selectionLoading} libraryLoading={libraryLoading} libraryError={libraryError} onRefresh={() => refreshRuns(!job)} onRefreshResult={refreshSelectedResult} />}
        {job?.result && job.result.engine !== "yolo_bytetrack" && <p className="notice">DEMO DATA — synthetic analytics for interface exploration.</p>}

        {job?.result && job.result.engine !== "yolo_bytetrack" && <><OverviewDashboard runs={runs} activeResult={job?.result} onDemo={startDemo} busy={busy} />
        <ModuleDock activeModule={activeModule} hasAnalysis={Boolean(job?.result)} onSelect={selectModule} busy={busy} />

        </>}
        {job?.result && job.result.engine !== "yolo_bytetrack" && <Library runs={runs} activeRunId={job?.result?.run_id} onOpen={openRun} />}

        {job?.result && job.result.engine !== "yolo_bytetrack" && activeModule !== "Overview" ? <Analysis job={job} activeModule={activeModule} /> : null}
      </section>
    </main>
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  for (const [name, value] of Object.entries(authHeaders())) headers.set(name, value);
  const response = await fetch(`${API}${path}`, { ...init, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `API request failed (${response.status})`);
  }
  return response.json();
}

function RealAnalysis({ result }: { result: AnalysisResult }) {
  const video = useRef<HTMLVideoElement>(null);
  const summary = result.detection_summary!;
  const [filter, setFilter] = useState("");
  return <section className="real-analysis" id="analysis">
    <div className="panel"><p className="eyebrow">Real model output · unreviewed</p><h2>Detection and tracking</h2>
      <p>{result.report.summary}</p>
      <div className="detection-stats">
        <div><strong>{summary.unique_tracks}</strong><span>Temporary tracks</span></div>
        <div><strong>{summary.analyzed_frames.toLocaleString()}</strong><span>Frames analyzed at {summary.sample_fps.toFixed(1)} fps</span></div>
        <div><strong>{summary.ball_detected_frames}</strong><span>Frames with ball candidates</span></div>
        <div><strong>{summary.processing_seconds.toFixed(1)}s</strong><span>Processing · {"gpu" in summary ? String(summary.gpu) : summary.device}</span></div>
      </div>
      <p>Model: {summary.model} · Tracker: {summary.tracker ?? "Not recorded in this run"} · Detector input: {summary.image_size ?? "unknown"} px · Source: {result.metadata.width} × {result.metadata.height} · {result.metadata.duration_seconds.toFixed(1)} seconds</p>
    </div>
    <div className="video-pane"><div className="pane-label"><Video size={16} /> Annotated video · detections are unreviewed</div>
      <video key={result.run_id} ref={video} controls preload="metadata" src={`${API}${result.files.annotated_video}`} />
    </div>
    <ReviewPanel key={result.run_id} runId={result.run_id} duration={result.metadata.duration_seconds} api={API} authHeaders={authHeaders} />
    <div className="panel"><h2>What the model found</h2><p>Counts are observations across frames, not counts of different people. Ball candidate coverage is not an accuracy score; candidates can include false positives.</p>
      <div className="detection-stats">{Object.entries(summary.detections_by_class).map(([label, count]) => <div key={label}><strong>{count.toLocaleString()}</strong><span>{label} detections</span></div>)}</div>
      <label>Find a track <input className="track-filter" value={filter} onChange={e => setFilter(e.target.value)} placeholder="Track ID or object class" /></label>
      <div className="track-table"><table><thead><tr><th>Track</th><th>Class</th><th>Observations</th><th>First seen</th><th>Last seen</th><th>Review</th></tr></thead><tbody>
        {summary.tracks.filter(t => `${t.track_id} ${t.object_type}`.toLowerCase().includes(filter.toLowerCase())).slice(0, 100).map(track => <tr key={track.track_id}>
          <td>{track.track_id}</td><td>{track.object_type}</td><td>{track.observations}</td><td>{(track.first_seen_ms / 1000).toFixed(1)}s</td><td>{(track.last_seen_ms / 1000).toFixed(1)}s</td>
          <td><button className="ghost" onClick={() => { if (video.current) { video.current.currentTime = track.first_seen_ms / 1000; video.current.scrollIntoView({behavior: "smooth", block: "center"}); } }}>Show in video</button></td>
        </tr>)}
      </tbody></table></div><p>Showing up to 100 matching tracks. Full detections are in the JSON download.</p>
    </div>
    <div className="panel"><h2>Analysis limits</h2><ul>{result.warnings?.map(warning => <li key={warning}>{warning}</li>)}</ul></div>
    <div className="panel"><h2>Download results</h2><div className="actions">
      <a className="ghost" href={`${API}${result.files.tracking_json}`} target="_blank" rel="noreferrer">Tracking JSON</a>
      <a className="ghost" href={`${API}${result.files.annotated_video}`} target="_blank" rel="noreferrer">Annotated MP4</a>
      <a className="ghost" href={result.files.manifest_json ? `${API}${result.files.manifest_json}` : `${API}/runs/${result.run_id}`} target="_blank" rel="noreferrer">Full analysis JSON</a>
    </div></div>
  </section>;
}

function OverviewDashboard({ runs, activeResult, onDemo, busy }: { runs: RunSummary[]; activeResult?: AnalysisResult; onDemo: () => void; busy: boolean }) {
  const result = activeResult ?? (runs[0] as unknown as AnalysisResult | undefined);
  const possession = result?.metrics?.possession;
  const players = result?.metrics?.players ?? [];
  const topPlayers = players.slice(0, 3);
  const home = result?.metrics?.teams?.Home;
  const away = result?.metrics?.teams?.Away;
  const avgRating = players.length ? (6.65 + Math.min(0.9, players[0].influence / 180)).toFixed(2) : "7.42";
  const reviewEvent = result?.metrics?.events?.[0];
  const phase = result?.phases?.[0]?.phase ?? "Progression";
  const demo = demoDashboardData();

  return (
    <section className="overview-board" id="overview">
      <header className="overview-nav">
        <button title="Menu"><Menu size={18} /></button>
        <button title="Create report"><Sparkles size={18} /></button>
        <strong>MatchMind</strong>
        <button title="Search"><Search size={18} /></button>
        <button title="Reports"><Database size={18} /></button>
      </header>

      <div className="overview-grid">
        <article className="overview-tile stat-tile">
          <TileLabel label="Matches Analysed" />
          <strong>{Math.max(runs.length, 1).toString().padStart(2, "0")}</strong>
          <span>{runs.length ? "runs saved" : "demo ready"}</span>
        </article>
        <article className="overview-tile stat-tile">
          <TileLabel label="Average Rating" />
          <strong>{avgRating}</strong>
          <span>match index</span>
        </article>
        <article className="overview-tile playstyle-tile">
          <TileLabel label="Play Styles" />
          <div className="radar-card">
            {demo.playStyles.map((style) => <span key={style}>{style}</span>)}
            <i />
          </div>
        </article>
        <article className="overview-tile scorers-tile">
          <TileLabel label="Top Influencers" />
          {(topPlayers.length ? topPlayers : demoInfluencers()).map((player, index) => (
            <div className="scorer" key={player.track_id}>
              <div>
                <strong>{player.track_id}</strong>
                <span>{player.team}</span>
              </div>
              <b>{Math.round(player.influence)}</b>
              <i style={{ width: `${Math.min(100, player.influence)}%` }} />
            </div>
          ))}
        </article>
        <article className="overview-tile stat-tile">
          <TileLabel label="Home Possession" />
          <strong>{possession?.home_pct ?? 54}%</strong>
          <span>{possession?.away_pct ?? 46}% opposition</span>
        </article>
        <article className="overview-tile stat-tile">
          <TileLabel label="Team Width" />
          <strong>{home?.avg_width ?? 32}m</strong>
          <span>{away?.compactness ?? 28} away compactness</span>
        </article>
        <article className="overview-tile performers-tile">
          <TileLabel label="Top Performers" />
          {(topPlayers.length ? topPlayers : demoInfluencers()).map((player) => (
            <div className="performer" key={player.track_id}>
              <span>{Number(player.influence / 10).toFixed(2)}</span>
              <div>
                <strong>{player.track_id}</strong>
                <small>{player.team}</small>
              </div>
            </div>
          ))}
          <small>influence rating</small>
        </article>
        <article className="overview-tile tips-tile">
          <TileLabel label="Tips" />
          <strong>Prediction Report</strong>
          <p>{reviewEvent?.description ?? "Unlock data-driven insights by reviewing phase changes, player influence, and possession momentum."}</p>
          <div className="tip-dots"><span /><span /><span /></div>
        </article>
        <article className="overview-tile trend-tile">
          <TileLabel label="Momentum" />
          <div className="mini-chart">
            {demo.momentum.map((value, index) => <i key={index} style={{ height: `${value}%` }} />)}
          </div>
          <span>last 8 phases</span>
        </article>
        <article className="overview-tile split-tile">
          <TileLabel label="Home vs Away Control" />
          <div className="split-row">
            <b>{possession?.home_pct ?? 57}%</b>
            <b>{possession?.away_pct ?? 43}%</b>
          </div>
          <div className="split-bar"><i style={{ width: `${possession?.home_pct ?? 57}%` }} /></div>
          <div className="split-row muted">
            <span>home</span>
            <span>away</span>
          </div>
        </article>
        <article className="overview-tile next-tile">
          <TileLabel label="Next Review" />
          <p>{phase} phase, {reviewEvent ? formatDuration(reviewEvent.timestamp_ms / 1000) : "0:00"}</p>
          <div className="versus">
            <span>HOME</span>
            <b>vs</b>
            <span>AWAY</span>
          </div>
        </article>
        <button className="overview-cta" onClick={onDemo} disabled={busy}>
          Get Full Report
        </button>
      </div>
    </section>
  );
}

function TileLabel({ label }: { label: string }) {
  return (
    <div className="tile-label">
      <span>{label}</span>
      <b>..</b>
    </div>
  );
}

function demoInfluencers() {
  return [
    { track_id: "H10", team: "Home", influence: 76 },
    { track_id: "H8", team: "Home", influence: 73 },
    { track_id: "A14", team: "Away", influence: 71 },
  ];
}

function demoDashboardData() {
  return {
    playStyles: ["Vertical Overload", "Counter Press", "Wide Rotations", "Rest Defence"],
    momentum: [34, 48, 42, 76, 68, 82, 61, 74],
  };
}

function createDemoAnalysis(): AnalysisResult {
  const fps = 25;
  const tracking: Track[] = [];
  for (let frame = 0; frame <= 300; frame += 10) {
    const t = frame / fps;
    tracking.push({
      frame_idx: frame,
      timestamp_ms: Math.round(t * 1000),
      track_id: "ball",
      object_type: "ball",
      team: null,
      x: Number((53 + 31 * Math.sin(t / 2.8)).toFixed(2)),
      y: Number((34 + 18 * Math.cos(t / 2.1)).toFixed(2)),
      confidence: 0.86,
    });
    for (let idx = 0; idx < 11; idx += 1) {
      const line = Math.floor(idx / 3);
      const slot = idx % 3;
      tracking.push({
        frame_idx: frame,
        timestamp_ms: Math.round(t * 1000),
        track_id: `H${idx + 1}`,
        object_type: "player",
        team: "Home",
        x: Number((18 + line * 16 + 7 * Math.sin(t / 2 + idx)).toFixed(2)),
        y: Number((12 + slot * 18 + (idx % 2) * 6 + 4 * Math.cos(t / 2.6 + idx)).toFixed(2)),
        confidence: 0.82,
      });
      tracking.push({
        frame_idx: frame,
        timestamp_ms: Math.round(t * 1000),
        track_id: `A${idx + 1}`,
        object_type: "player",
        team: "Away",
        x: Number((88 - line * 15 + 7 * Math.sin(t / 2.2 + idx + 1)).toFixed(2)),
        y: Number((12 + slot * 18 + (idx % 2) * 6 + 4 * Math.cos(t / 2.4 + idx)).toFixed(2)),
        confidence: 0.8,
      });
    }
  }

  const phases = [
    { phase: "Build-Up", start_ms: 0, end_ms: 2600, confidence: 0.78 },
    { phase: "Progression", start_ms: 2600, end_ms: 6200, confidence: 0.82 },
    { phase: "Wide Final Attack", start_ms: 6200, end_ms: 9400, confidence: 0.75 },
    { phase: "Counter Press", start_ms: 9400, end_ms: 12000, confidence: 0.71 },
  ];
  const players = [
    { track_id: "H8", team: "Home", distance_m: 238.5, max_speed_kmh: 28.4, avg_x: 53.2, avg_y: 31.6, influence: 88 },
    { track_id: "H10", team: "Home", distance_m: 226.2, max_speed_kmh: 27.9, avg_x: 61.8, avg_y: 38.1, influence: 82 },
    { track_id: "A6", team: "Away", distance_m: 218.8, max_speed_kmh: 26.7, avg_x: 48.4, avg_y: 29.4, influence: 76 },
    { track_id: "H3", team: "Home", distance_m: 202.1, max_speed_kmh: 25.3, avg_x: 41.9, avg_y: 54.2, influence: 69 },
  ];

  return {
    run_id: "vercel-demo",
    engine: "static_demo",
    metadata: { filename: "vercel-demo-match.mp4", fps, frame_count: 300, duration_seconds: 12, width: 1280, height: 720 },
    tracking,
    metrics: {
      possession: {
        home_pct: 58.4,
        away_pct: 41.6,
        timeline: phases.map((phase, index) => ({ timestamp_ms: phase.start_ms, team: index === 3 ? "Away" : "Home" })),
      },
      teams: {
        Home: { avg_line_height: 48.6, avg_width: 36.4, compactness: 24.8, distance_estimate_km: 3.1, sprint_count_estimate: 14 },
        Away: { avg_line_height: 55.2, avg_width: 31.8, compactness: 29.4, distance_estimate_km: 2.7, sprint_count_estimate: 9 },
      },
      players,
      events: [
        { timestamp_ms: 2600, type: "Phase Entry", team: "Home", description: "Home progressed through the central lane after drawing the first press.", confidence: 0.82 },
        { timestamp_ms: 6200, type: "Wide Overload", team: "Home", description: "Right-side rotation created a temporary 3v2 overload near the final third.", confidence: 0.77 },
        { timestamp_ms: 9400, type: "Possession Swing", team: "Away", description: "Away recovered after a loose touch and forced a counter-press moment.", confidence: 0.71 },
      ],
      data_quality: { tracking_confidence: 0.84, homography_quality: 0.76, ball_missing_ratio: 0.03, mode: "vercel_static_demo" },
    },
    phases,
    report: {
      title: "Static Demo Tactical Intelligence Report",
      summary: "Home controlled the strongest spells through compact central support and right-side rotations. The most valuable review moment is the progression-to-wide-attack sequence, where H8 and H10 create the platform for a final-third entry.",
      recommendations: [
        { label: "Protect the progression lane", detail: "Keep the near-side fullback deeper when H8 steps into the half-space so rest defence is not exposed.", confidence: 0.79 },
        { label: "Review H8 influence clips", detail: "H8 leads the demo influence model through repeated support angles and fast second actions after phase entries.", confidence: 0.84 },
        { label: "Pressure after loss", detail: "The counter-press moment at 9.4s shows the first recovery wave arriving late, leaving the central lane open.", confidence: 0.73 },
      ],
    },
    files: { annotated_video: "", tracking_json: "", metrics_json: "", report_json: "" },
  };
}

function demoJob(result: AnalysisResult, stage: string): Job {
  return {
    id: "demo-vercel-job",
    status: "complete",
    stage,
    progress: 1,
    created_at: new Date().toISOString(),
    result,
  };
}

function demoRun(result: AnalysisResult): RunSummary {
  return {
    run_id: result.run_id,
    created_at: new Date().toISOString(),
    metadata: result.metadata,
    metrics: result.metrics,
    phases: result.phases,
    report: result.report,
    files: result.files,
  };
}

function demoCapabilities(): Capabilities {
  return {
    engines: {
      fallback: { available: true, description: "Static frontend demo data" },
      yolo: { available: false, model_exists: false, ultralytics_installed: false, opencv_available: false, model_path: "backend not attached" },
    },
  };
}

function Library({ runs, activeRunId, onOpen }: { runs: RunSummary[]; activeRunId?: string; onOpen: (runId: string) => void }) {
  if (!runs.length) return null;
  return (
    <section className="library">
      <div>
        <p className="eyebrow">Match Library</p>
        <h2>Completed analyses</h2>
      </div>
      <div className="run-list">
        {runs.slice(0, 6).map((run) => (
          <button className={run.run_id === activeRunId ? "run active" : "run"} key={run.run_id} onClick={() => onOpen(run.run_id)}>
            <span className="run-kicker">{new Date(run.created_at).toLocaleDateString()}</span>
            <strong>{run.metadata?.filename ?? run.run_id}</strong>
            <span>{run.metrics?.data_quality?.mode === "yolo_bytetrack" ? "Real object tracking" : "Synthetic demo"}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function Progress({ job, capabilities }: { job: Job | null; capabilities: Capabilities | null }) {
  const pct = Math.round((job?.progress ?? 0) * 100);
  const yolo = capabilities?.engines.yolo;
  return (
    <div className="progress-wrap">
      <div>
        <span>{job ? job.stage : "Ready"}</span>
        <strong>{job ? `${pct}%` : yolo?.available ? "Detection ready" : "Analysis unavailable"}</strong>
      </div>
      <div className="progress-track"><div style={{ width: `${pct}%` }} /></div>
      {yolo?.blockers?.length ? <p role="status">{yolo.blockers.join(" ")}</p> : null}
      {yolo && (
        <div className="capability-line">
          <span>{yolo.compute_provider === "modal" ? (yolo.available ? "Modal GPU configured" : "Modal connection required") : (yolo.available ? "YOLO engine available" : "YOLO engine pending dependencies")}</span>
          <span>{yolo.compute_provider === "modal" ? "Clips up to 2 minutes / 250 MiB" : `model ${yolo.model_exists ? "found" : "missing"} / ultralytics ${yolo.ultralytics_installed ? "installed" : "missing"}`}</span>
        </div>
      )}
    </div>
  );
}

function EmptyState({ onDemo, busy }: { onDemo: () => void; busy: boolean }) {
  return (
    <section className="empty-grid">
      <div className="pitch-hero">
        <Pitch result={null} />
      </div>
      <div className="intro-panel">
        <Sparkles size={22} />
        <h2>Upload a clip and turn it into tactical evidence.</h2>
        <p>
          The first build produces tracking data, possession timelines, phase tags, annotated video, and a grounded AI report.
        </p>
        <button className="primary wide" onClick={onDemo} disabled={busy}>
          <Wand2 size={18} /> Run demo flow
        </button>
      </div>
    </section>
  );
}

function Analysis({ job, activeModule }: { job: Job; activeModule: string }) {
  const result = job.result!;
  return (
    <section className={`analysis-grid module-${slug(activeModule)}`} id="analysis">
      <MatchHeader result={result} />
      <InsightDeck result={result} />
      <ModuleContent result={result} activeModule={activeModule} />
    </section>
  );
}

function ModuleContent({ result, activeModule }: { result: AnalysisResult; activeModule: string }) {
  if (activeModule === "Timeline") return <TimelineModule result={result} activeModule={activeModule} />;
  if (activeModule === "Team Shape") return <TeamShapeModule result={result} activeModule={activeModule} />;
  if (activeModule === "Possession") return <PossessionModule result={result} activeModule={activeModule} />;
  if (activeModule === "Pressure Lab") return <PressureModule result={result} activeModule={activeModule} />;
  return <MatchReviewModule result={result} activeModule={activeModule} />;
}

function MatchReviewModule({ result, activeModule }: { result: AnalysisResult; activeModule: string }) {
  return (
    <>
      <VideoPane result={result} />
      <CanvasPane result={result} activeModule={activeModule} />
      <MetricStrip result={result} />
      <QualityPanel result={result} />
      <ReportPanel result={result} />
      <ArtifactPanel result={result} />
    </>
  );
}

function TimelineModule({ result, activeModule }: { result: AnalysisResult; activeModule: string }) {
  return (
    <>
      <VideoPane result={result} />
      <section className="panel timeline-focus">
        <div className="panel-title">
          <h2>Event Timeline</h2>
          <span>{result.metrics.events?.length ?? 0} key events</span>
        </div>
        <TimelineRail result={result} />
      </section>
      <CanvasPane result={result} activeModule={activeModule} />
      <EventsPanel result={result} />
      <PhasePanel result={result} />
    </>
  );
}

function TeamShapeModule({ result, activeModule }: { result: AnalysisResult; activeModule: string }) {
  return (
    <>
      <CanvasPane result={result} activeModule={activeModule} />
      <ShapeMetrics result={result} />
      <PhasePanel result={result} />
      <PlayerPanel result={result} />
      <ReportPanel result={result} />
    </>
  );
}

function PossessionModule({ result, activeModule }: { result: AnalysisResult; activeModule: string }) {
  return (
    <>
      <CanvasPane result={result} activeModule={activeModule} />
      <PossessionPanel result={result} />
      <PossessionFlow result={result} />
      <EventsPanel result={result} />
      <MetricStrip result={result} />
    </>
  );
}

function PressureModule({ result, activeModule }: { result: AnalysisResult; activeModule: string }) {
  return (
    <>
      <CanvasPane result={result} activeModule={activeModule} />
      <PressureVerdict result={result} />
      <EventsPanel result={result} />
      <PlayerPanel result={result} />
      <QualityPanel result={result} />
    </>
  );
}

function VideoPane({ result }: { result: AnalysisResult }) {
  if (!result.files.annotated_video) {
    return (
      <div className="video-pane demo-video-pane">
        <div className="pane-label">
          <Video size={16} />
          Demo Match Feed
        </div>
        <div className="demo-video-frame">
          <Pitch result={result} />
          <div>
            <strong>Static Vercel Demo</strong>
            <span>Connect the Railway API to enable real uploads and annotated video playback.</span>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="video-pane">
      <div className="pane-label">
        <Video size={16} />
        Annotated Match Feed
      </div>
      <video controls src={`${API}${result.files.annotated_video}`} />
    </div>
  );
}

function CanvasPane({ result, activeModule }: { result: AnalysisResult; activeModule: string }) {
  return (
    <div className="pitch-pane">
      <div className="pane-label">
        <Radar size={16} />
        Canvas Analysis
      </div>
      <CanvasAnalysis result={result} mode={activeModule} />
    </div>
  );
}

function ModuleDock({ activeModule, hasAnalysis, onSelect, busy }: { activeModule: string; hasAnalysis: boolean; onSelect: (moduleName: string) => void; busy: boolean }) {
  const modules = [
    ["Overview", "Command center", "Live"],
    ["Match Review", "Video + pitch", hasAnalysis ? "Ready" : "Run demo"],
    ["Timeline", "Events + clips", hasAnalysis ? "Ready" : "Pending"],
    ["Team Shape", "Lines + spaces", hasAnalysis ? "Ready" : "Pending"],
    ["Possession", "Control flow", hasAnalysis ? "Ready" : "Pending"],
    ["Pressure Lab", "Defensive actions", "Soon"],
  ];
  return (
    <nav className="module-dock" aria-label="Application modules">
      {modules.map(([name, detail, status], index) => (
        <button
          className={activeModule === name ? "active" : ""}
          key={name}
          onClick={() => onSelect(name)}
          disabled={busy}
        >
          <i>{String(index + 1).padStart(2, "0")}</i>
          <strong>{name}</strong>
          <span>{detail}</span>
          <b>{status}</b>
        </button>
      ))}
    </nav>
  );
}

function InsightDeck({ result }: { result: AnalysisResult }) {
  const firstEvent = result.metrics.events?.[0];
  const topPlayer = result.metrics.players?.[0];
  const longestPhase = [...result.phases].sort((a, b) => (b.end_ms - b.start_ms) - (a.end_ms - a.start_ms))[0];
  return (
    <section className="insight-deck">
      <article className="insight-card primary-insight">
        <span><Sparkles size={16} /> Review Priority</span>
        <strong>{firstEvent?.type ?? "Shape Review"}</strong>
        <p>{firstEvent?.description ?? "Start with possession and phase context, then validate clips."}</p>
      </article>
      <article className="insight-card">
        <span><Users size={16} /> Player Signal</span>
        <strong>{topPlayer?.track_id ?? "Pending"}</strong>
        <p>{topPlayer ? `${topPlayer.distance_m}m covered, peak ${topPlayer.max_speed_kmh} km/h.` : "Run a fresh analysis to populate player summaries."}</p>
      </article>
      <article className="insight-card">
        <span><Layers3 size={16} /> Dominant Phase</span>
        <strong>{longestPhase?.phase ?? "Pending"}</strong>
        <p>{longestPhase ? `${Math.round((longestPhase.end_ms - longestPhase.start_ms) / 1000)} seconds of sustained behavior.` : "Phase segmentation will appear after processing."}</p>
      </article>
    </section>
  );
}

function PlayerPanel({ result }: { result: AnalysisResult }) {
  const players = result.metrics.players ?? [];
  return (
    <section className="panel player-panel">
      <div className="panel-title">
        <h2>Player Influence</h2>
        <span>Top {Math.min(players.length, 6)}</span>
      </div>
      <div className="player-table">
        {players.slice(0, 6).map((player) => (
          <div className="player-row" key={player.track_id}>
            <span className={player.team === "Home" ? "team-dot home" : "team-dot away"} />
            <strong>{player.track_id}</strong>
            <span>{player.distance_m}m</span>
            <span>{player.max_speed_kmh} km/h</span>
            <i style={{ width: `${Math.min(100, player.influence)}%` }} />
          </div>
        ))}
      </div>
    </section>
  );
}

function EventsPanel({ result }: { result: AnalysisResult }) {
  const events = result.metrics.events ?? [];
  return (
    <section className="panel events-panel">
      <div className="panel-title">
        <h2>Key Moments</h2>
        <span>{events.length} detected</span>
      </div>
      <div className="event-list">
        {events.slice(0, 7).map((event, index) => (
          <article className="event" key={`${event.timestamp_ms}-${index}`}>
            <time>{formatDuration(event.timestamp_ms / 1000)}</time>
            <div>
              <strong>{event.type}</strong>
              <span>{event.description}</span>
            </div>
          </article>
        ))}
      </div>
      <ClipQueue result={result} />
    </section>
  );
}

function TimelineRail({ result }: { result: AnalysisResult }) {
  const events = result.metrics.events ?? [];
  const duration = Math.max(result.metadata.duration_seconds, 1);
  return (
    <div className="timeline-rail">
      <div className="timeline-track">
        {events.map((event, index) => (
          <span
            key={`${event.timestamp_ms}-${index}`}
            style={{ left: `${Math.min(96, (event.timestamp_ms / 1000 / duration) * 100)}%` }}
            title={event.description}
          />
        ))}
      </div>
      <div className="timeline-cards">
        {events.slice(0, 5).map((event, index) => (
          <article key={`${event.type}-${index}`}>
            <time>{formatDuration(event.timestamp_ms / 1000)}</time>
            <strong>{event.type}</strong>
            <p>{event.description}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

function ShapeMetrics({ result }: { result: AnalysisResult }) {
  const home = result.metrics.teams.Home;
  const away = result.metrics.teams.Away;
  const rows = [
    ["Width", `${home.avg_width}m`, `${away.avg_width ?? 0}m`],
    ["Line height", `${home.avg_line_height}m`, `${away.avg_line_height}m`],
    ["Compactness", `${home.compactness}`, `${away.compactness}`],
    ["Sprint exposure", `${home.sprint_count_estimate}`, `${away.sprint_count_estimate}`],
  ];
  return (
    <section className="panel shape-metrics">
      <div className="panel-title">
        <h2>Shape Comparison</h2>
        <span>Home / Away</span>
      </div>
      <div className="shape-table">
        {rows.map(([label, left, right]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{left}</strong>
            <b>{right}</b>
          </div>
        ))}
      </div>
    </section>
  );
}

function PossessionFlow({ result }: { result: AnalysisResult }) {
  const timeline = result.metrics.possession.timeline.slice(0, 16);
  return (
    <section className="panel possession-flow">
      <div className="panel-title">
        <h2>Control Flow</h2>
        <span>{timeline.length} samples</span>
      </div>
      <div className="flow-lanes">
        {timeline.map((item, index) => (
          <article className={item.team === "Home" ? "home" : "away"} key={`${item.timestamp_ms}-${index}`}>
            <time>{formatDuration(item.timestamp_ms / 1000)}</time>
            <strong>{item.team}</strong>
          </article>
        ))}
      </div>
    </section>
  );
}

function PressureVerdict({ result }: { result: AnalysisResult }) {
  const events = result.metrics.events ?? [];
  const pressureScore = Math.min(99, Math.round((events.length * 12) + Number(result.metrics.data_quality.tracking_confidence ?? 0) * 40));
  const failureRisk = Math.max(1, Math.round(Number(result.metrics.data_quality.ball_missing_ratio ?? 0.04) * 100));
  return (
    <section className="panel pressure-verdict">
      <div className="panel-title">
        <h2>Pressure Verdict</h2>
        <span>{pressureScore > 65 ? "aggressive" : "moderate"}</span>
      </div>
      <div className="verdict-grid">
        <article>
          <span>Hunter Score</span>
          <strong>{pressureScore}</strong>
          <small>pressure actions inferred</small>
        </article>
        <article>
          <span>Escape Risk</span>
          <strong>{failureRisk}%</strong>
          <small>ball visibility risk</small>
        </article>
        <article>
          <span>Trigger Count</span>
          <strong>{events.length}</strong>
          <small>reviewable moments</small>
        </article>
      </div>
    </section>
  );
}

function ClipQueue({ result }: { result: AnalysisResult }) {
  const clips = [
    ...(result.metrics.events ?? []).slice(0, 3).map((event) => ({
      time: event.timestamp_ms / 1000,
      label: event.type,
    })),
    ...result.phases.slice(0, 2).map((phase) => ({
      time: phase.start_ms / 1000,
      label: phase.phase,
    })),
  ];
  return (
    <div className="clip-queue">
      <div className="panel-title">
        <h2>Clip Queue</h2>
        <span>{clips.length} ready</span>
      </div>
      <div className="clip-list">
        {clips.map((clip, index) => (
          <button key={`${clip.label}-${index}`}>
            <Play size={14} />
            <strong>{formatDuration(clip.time)}</strong>
            <span>{clip.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function MatchHeader({ result }: { result: AnalysisResult }) {
  return (
    <section className="match-header">
      <div>
        <p className="eyebrow">Current Analysis</p>
        <h2>{result.metadata.filename}</h2>
      </div>
      <div className="match-meta">
        <span><Clock3 size={15} /> {formatDuration(result.metadata.duration_seconds)}</span>
        <span><Video size={15} /> {result.metadata.width}x{result.metadata.height}</span>
        <span><CalendarDays size={15} /> {Math.round(result.metadata.fps)} FPS</span>
        <span><Bot size={15} /> {result.engine ?? "fallback"}</span>
      </div>
    </section>
  );
}

function QualityPanel({ result }: { result: AnalysisResult }) {
  const quality = result.metrics.data_quality;
  return (
    <section className="panel quality-panel">
      <h2>Engine Quality</h2>
      <div className="quality-grid">
        <Metric icon={<Bot />} label="Engine" value={String(result.engine ?? quality.mode ?? "unknown")} />
        <Metric icon={<Gauge />} label="Tracking" value={`${Math.round(Number(quality.tracking_confidence ?? 0) * 100)}%`} />
        <Metric icon={<Gauge />} label="Ball missing" value={`${Math.round(Number(quality.ball_missing_ratio ?? 0) * 100)}%`} />
      </div>
    </section>
  );
}

function PossessionPanel({ result }: { result: AnalysisResult }) {
  const home = result.metrics.possession.home_pct;
  const away = result.metrics.possession.away_pct;
  return (
    <section className="panel possession-panel">
      <div className="panel-title">
        <h2>Possession Control</h2>
        <span>{home}% / {away}%</span>
      </div>
      <div className="possession-bar">
        <div className="home" style={{ width: `${home}%` }} />
        <div className="away" style={{ width: `${away}%` }} />
      </div>
      <div className="timeline">
        {result.metrics.possession.timeline.slice(0, 42).map((item, index) => (
          <span
            key={`${item.timestamp_ms}-${index}`}
            className={item.team === "Home" ? "home" : "away"}
            title={`${item.team} at ${Math.round(item.timestamp_ms / 1000)}s`}
          />
        ))}
      </div>
    </section>
  );
}

function Pitch({ result }: { result: AnalysisResult | null }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setTick((value) => value + 1), 520);
    return () => window.clearInterval(timer);
  }, []);

  const frame = useMemo(() => {
    if (!result) return [];
    const frames = Array.from(new Set(result.tracking.map((row) => row.frame_idx))).sort((a, b) => a - b);
    const nearest = frames[tick % frames.length] ?? 0;
    return result.tracking.filter((row) => row.frame_idx === nearest);
  }, [result, tick]);

  return (
    <div className="pitch">
      <div className="zone z1">Build</div>
      <div className="zone z2">Progress</div>
      <div className="zone z3">Attack</div>
      <div className="box left" />
      <div className="box right" />
      <div className="midline" />
      <div className="circle" />
      {result && <ShapeHull frame={frame} team="Home" />}
      {result && <ShapeHull frame={frame} team="Away" />}
      {(result ? frame : demoDots()).map((row, index) => (
        <span
          key={`${row.track_id}-${index}`}
          className={`dot ${row.object_type === "ball" ? "ball" : row.team === "Away" ? "away" : "home"}`}
          style={{ left: `${(row.x / 105) * 100}%`, top: `${(row.y / 68) * 100}%` }}
        />
      ))}
    </div>
  );
}

function CanvasAnalysis({ result, mode }: { result: AnalysisResult; mode: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [tick, setTick] = useState(0);
  const [metric, setMetric] = useState(canvasMetricForModule(mode));
  const [layers, setLayers] = useState({
    shape: true,
    trails: true,
    zones: true,
    heat: true,
    labels: true,
  });

  useEffect(() => {
    setMetric(canvasMetricForModule(mode));
  }, [mode]);

  useEffect(() => {
    const timer = window.setInterval(() => setTick((value) => value + 1), 520);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * scale));
    canvas.height = Math.max(1, Math.floor(rect.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(scale, scale);
    drawCanvasAnalysis(ctx, rect.width, rect.height, result, mode, tick, metric, layers);
  }, [result, mode, tick, metric, layers]);

  return (
    <div className="canvas-analysis">
      <div className="canvas-toolbar">
        <div>
          {["Shape", "Pressure", "Possession", "Momentum", "Player Load"].map((item) => (
            <button className={metric === item ? "active" : ""} key={item} onClick={() => setMetric(item)}>
              {item}
            </button>
          ))}
        </div>
        <div>
          {Object.entries(layers).map(([key, value]) => (
            <button className={value ? "active subtle" : "subtle"} key={key} onClick={() => setLayers((next) => ({ ...next, [key]: !value }))}>
              {key}
            </button>
          ))}
        </div>
      </div>
      <canvas ref={canvasRef} />
      <div className="canvas-legend">
        <span><i className="home" /> Home</span>
        <span><i className="away" /> Away</span>
        <span><i className="ball" /> Ball</span>
        <span>{metric}</span>
      </div>
      <div className="canvas-caption">
        <strong>{mode}</strong>
        <span>Tracking, shape, control and event context rendered on one tactical surface.</span>
      </div>
    </div>
  );
}

function drawCanvasAnalysis(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  result: AnalysisResult,
  mode: string,
  tick: number,
  metric: string,
  layers: { shape: boolean; trails: boolean; zones: boolean; heat: boolean; labels: boolean },
) {
  ctx.clearRect(0, 0, width, height);
  const pad = 34;
  const pitchW = width - pad * 2;
  const pitchH = height - pad * 2;
  const x = (v: number) => pad + (v / 105) * pitchW;
  const y = (v: number) => pad + (v / 68) * pitchH;
  const frames = Array.from(new Set(result.tracking.map((row) => row.frame_idx))).sort((a, b) => a - b);
  const frameId = frames[tick % Math.max(frames.length, 1)] ?? 0;
  const current = result.tracking.filter((row) => row.frame_idx === frameId);
  const players = current.filter((row) => row.object_type === "player");
  const ball = current.find((row) => row.object_type === "ball");

  const gradient = ctx.createLinearGradient(pad, pad, width - pad, height - pad);
  gradient.addColorStop(0, "#135a33");
  gradient.addColorStop(1, "#0c3c28");
  ctx.fillStyle = gradient;
  roundRect(ctx, pad, pad, pitchW, pitchH, 16);
  ctx.fill();

  for (let i = 0; i < 6; i++) {
    ctx.fillStyle = i % 2 ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0.055)";
    ctx.fillRect(pad + (pitchW / 6) * i, pad, pitchW / 6, pitchH);
  }

  ctx.strokeStyle = "rgba(255,255,255,0.62)";
  ctx.lineWidth = 1.4;
  ctx.strokeRect(pad, pad, pitchW, pitchH);
  line(ctx, x(52.5), pad, x(52.5), pad + pitchH);
  circle(ctx, x(52.5), y(34), Math.min(pitchW, pitchH) * 0.12, false);
  ctx.strokeRect(pad, y(17), pitchW * 0.16, y(51) - y(17));
  ctx.strokeRect(width - pad - pitchW * 0.16, y(17), pitchW * 0.16, y(51) - y(17));

  if (layers.zones) {
    drawZones(ctx, pad, pad, pitchW, pitchH);
  }

  if (layers.heat) {
    drawHeatLayer(ctx, result.tracking, x, y, metric);
  }

  if (layers.trails) {
    drawTrails(ctx, result.tracking, frameId, x, y);
  }

  if (layers.shape) {
    drawTeamShape(ctx, players.filter((p) => p.team === "Home"), x, y, "rgba(121,111,255,0.2)", "#b9b2ff");
    drawTeamShape(ctx, players.filter((p) => p.team === "Away"), x, y, "rgba(255,113,166,0.16)", "#ff8dbb");
  }

  if (mode === "Timeline" || mode === "Possession" || metric === "Possession" || metric === "Momentum") {
    drawPossessionRibbon(ctx, result, pad, pad + pitchH + 12, pitchW);
  }

  if (mode === "Pressure Lab" || metric === "Pressure") {
    drawPressureField(ctx, players, x, y);
  }

  players.forEach((player) => {
    ctx.fillStyle = player.team === "Home" ? "#f7f4ff" : "#ffcf5b";
    ctx.strokeStyle = "#080816";
    ctx.lineWidth = 2;
    circle(ctx, x(player.x), y(player.y), 7, true);
    ctx.stroke();
    if (layers.labels) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "700 10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(player.track_id.replace(/[A-Z]/g, ""), x(player.x), y(player.y) - 12);
    }
  });

  if (ball) {
    ctx.fillStyle = "#7b6dff";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    circle(ctx, x(ball.x), y(ball.y), 5, true);
    ctx.stroke();
  }
}

function drawZones(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number) {
  ["DEFENSIVE THIRD", "PROGRESSION THIRD", "FINAL THIRD"].forEach((label, index) => {
    ctx.fillStyle = index === 1 ? "rgba(123,109,255,0.08)" : "rgba(255,255,255,0.025)";
    ctx.fillRect(x + (width / 3) * index, y, width / 3, height);
    ctx.fillStyle = "rgba(255,255,255,0.72)";
    ctx.font = "800 10px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, x + width * (index + 0.5) / 3, y + 20);
  });
}

function drawHeatLayer(ctx: CanvasRenderingContext2D, tracking: Track[], x: (v: number) => number, y: (v: number) => number, metric: string) {
  const sample = tracking.filter((row) => row.object_type === "player").filter((_, index) => index % 18 === 0).slice(-180);
  sample.forEach((point) => {
    const radius = metric === "Player Load" ? 28 : 22;
    const grd = ctx.createRadialGradient(x(point.x), y(point.y), 2, x(point.x), y(point.y), radius);
    const color = point.team === "Home" ? "123,109,255" : "244,113,166";
    grd.addColorStop(0, `rgba(${color},0.16)`);
    grd.addColorStop(1, `rgba(${color},0)`);
    ctx.fillStyle = grd;
    circle(ctx, x(point.x), y(point.y), radius, true);
  });
}

function drawTrails(ctx: CanvasRenderingContext2D, tracking: Track[], frameId: number, x: (v: number) => number, y: (v: number) => number) {
  const recent = tracking.filter((row) => row.frame_idx <= frameId && row.frame_idx >= frameId - 70 && row.object_type === "player");
  const byPlayer: Record<string, Track[]> = {};
  recent.forEach((row) => {
    byPlayer[row.track_id] = [...(byPlayer[row.track_id] ?? []), row];
  });
  Object.values(byPlayer).forEach((rows) => {
    if (rows.length < 2) return;
    ctx.beginPath();
    rows.forEach((row, index) => {
      if (index === 0) ctx.moveTo(x(row.x), y(row.y));
      else ctx.lineTo(x(row.x), y(row.y));
    });
    ctx.strokeStyle = rows[0].team === "Home" ? "rgba(185,178,255,0.42)" : "rgba(255,141,187,0.34)";
    ctx.lineWidth = 1.4;
    ctx.stroke();
  });
}

function drawTeamShape(ctx: CanvasRenderingContext2D, players: Track[], x: (v: number) => number, y: (v: number) => number, fill: string, stroke: string) {
  if (players.length < 3) return;
  const minX = Math.min(...players.map((p) => p.x));
  const maxX = Math.max(...players.map((p) => p.x));
  const minY = Math.min(...players.map((p) => p.y));
  const maxY = Math.max(...players.map((p) => p.y));
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  roundRect(ctx, x(minX), y(minY), Math.max(20, x(maxX) - x(minX)), Math.max(20, y(maxY) - y(minY)), 18);
  ctx.fill();
  ctx.stroke();
}

function drawPressureField(ctx: CanvasRenderingContext2D, players: Track[], x: (v: number) => number, y: (v: number) => number) {
  players.slice(0, 12).forEach((player) => {
    const grd = ctx.createRadialGradient(x(player.x), y(player.y), 4, x(player.x), y(player.y), 42);
    grd.addColorStop(0, player.team === "Home" ? "rgba(121,111,255,0.22)" : "rgba(255,113,166,0.2)");
    grd.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = grd;
    circle(ctx, x(player.x), y(player.y), 42, true);
  });
}

function drawPossessionRibbon(ctx: CanvasRenderingContext2D, result: AnalysisResult, x: number, y: number, width: number) {
  const items = result.metrics.possession.timeline.slice(0, 42);
  items.forEach((item, index) => {
    ctx.fillStyle = item.team === "Home" ? "#7b6dff" : "#f471a6";
    ctx.fillRect(x + (width / Math.max(items.length, 1)) * index, y, Math.max(3, width / Math.max(items.length, 1) - 2), 8);
  });
}

function line(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function circle(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number, fill: boolean) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  fill ? ctx.fill() : ctx.stroke();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function ShapeHull({ frame, team }: { frame: Track[]; team: "Home" | "Away" }) {
  const players = frame.filter((row) => row.object_type === "player" && row.team === team);
  if (players.length < 3) return null;
  const minX = Math.min(...players.map((row) => row.x));
  const maxX = Math.max(...players.map((row) => row.x));
  const minY = Math.min(...players.map((row) => row.y));
  const maxY = Math.max(...players.map((row) => row.y));
  return (
    <span
      className={`shape-hull ${team === "Home" ? "home" : "away"}`}
      style={{
        left: `${(minX / 105) * 100}%`,
        top: `${(minY / 68) * 100}%`,
        width: `${((maxX - minX) / 105) * 100}%`,
        height: `${((maxY - minY) / 68) * 100}%`,
      }}
    />
  );
}

function MetricStrip({ result }: { result: AnalysisResult }) {
  const home = result.metrics.teams.Home;
  const away = result.metrics.teams.Away;
  return (
    <div className="metric-strip">
      <Metric icon={<Shield />} label="Home possession" value={`${result.metrics.possession.home_pct}%`} />
      <Metric icon={<Target />} label="Away possession" value={`${result.metrics.possession.away_pct}%`} />
      <Metric icon={<Activity />} label="Home width" value={`${home.avg_width}m`} />
      <Metric icon={<Activity />} label="Away compactness" value={`${away.compactness}`} />
    </div>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      <span>{icon}</span>
      <small>{label}</small>
      <strong>{value}</strong>
    </div>
  );
}

function PhasePanel({ result }: { result: AnalysisResult }) {
  const maxDuration = Math.max(...result.phases.map((phase) => phase.end_ms - phase.start_ms), 1);
  return (
    <section className="panel phase-panel">
      <div className="panel-title">
        <h2>Tactical Phases</h2>
        <span>{result.phases.length} segments</span>
      </div>
      <div className="phase-list">
        {result.phases.slice(0, 8).map((phase, index) => (
          <div className="phase" key={`${phase.phase}-${index}`}>
            <div>
              <strong>{phase.phase}</strong>
              <span>{Math.round((phase.end_ms - phase.start_ms) / 1000)}s</span>
            </div>
            <i style={{ width: `${Math.max(12, ((phase.end_ms - phase.start_ms) / maxDuration) * 100)}%` }} />
          </div>
        ))}
      </div>
    </section>
  );
}

function ReportPanel({ result }: { result: AnalysisResult }) {
  return (
    <section className="panel report-panel">
      <div className="report-heading">
        <Bot size={20} />
        <h2>{result.report.title}</h2>
      </div>
      <p>{result.report.summary}</p>
      {result.report.recommendations.map((item) => (
        <article className="recommendation" key={item.label}>
          <strong>{item.label}</strong>
          <span>{item.detail}</span>
        </article>
      ))}
    </section>
  );
}

function ArtifactPanel({ result }: { result: AnalysisResult }) {
  const links = [
    ["Annotated video", result.files.annotated_video],
    ["Tracking JSON", result.files.tracking_json],
    ["Metrics JSON", result.files.metrics_json],
    ["Report JSON", result.files.report_json],
  ].filter(([, href]) => Boolean(href));

  return (
    <section className="panel artifact-panel">
      <h2>Artifacts</h2>
      {!links.length ? <p className="artifact-note">Static demo mode. Deploy the backend and set `VITE_API_URL` to unlock downloadable artifacts.</p> : null}
      <div className="artifact-list">
        {links.map(([label, href]) => (
          <a key={label} href={`${API}${href}`} target="_blank" rel="noreferrer">
            <Database size={16} />
            <span>{label}</span>
          </a>
        ))}
      </div>
    </section>
  );
}

function formatDuration(seconds: number) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function canvasMetricForModule(moduleName: string) {
  if (moduleName === "Pressure Lab") return "Pressure";
  if (moduleName === "Possession") return "Possession";
  if (moduleName === "Timeline") return "Momentum";
  if (moduleName === "Team Shape") return "Shape";
  return "Shape";
}

function demoDots(): Track[] {
  return Array.from({ length: 23 }, (_, index) => ({
    frame_idx: 0,
    timestamp_ms: 0,
    track_id: `${index}`,
    object_type: index === 22 ? "ball" : "player",
    team: index === 22 ? null : index < 11 ? "Home" : "Away",
    x: index === 22 ? 52 : index < 11 ? 18 + (index % 4) * 12 : 87 - (index % 4) * 12,
    y: index === 22 ? 34 : 12 + (index % 5) * 11,
    confidence: 1,
  }));
}

function PrivateWorkspace() {
  const [unlocked, setUnlocked] = useState(!REQUIRE_ACCESS_KEY || Boolean(sessionStorage.getItem("laas-access-key")));
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);
  if (unlocked) return <App />;
  return <main style={{ maxWidth: 580, margin: "12vh auto", padding: 24 }}><section className="panel">
    <p className="eyebrow">Private workspace</p><h1>Match Intelligence Studio</h1>
    <p>Enter your workspace access key to upload footage and review analyses.</p>
    <form onSubmit={async event => {
      event.preventDefault(); setChecking(true); setError("");
      try {
        const response = await fetch(`${API}/capabilities`, { headers: { Authorization: `Bearer ${key.trim()}` } });
        if (!response.ok) throw new Error("Access key was not accepted.");
        sessionStorage.setItem("laas-access-key", key.trim()); setUnlocked(true);
      } catch (err) { setError(err instanceof TypeError ? "Cannot reach the analysis service. Try again shortly." : String(err)); }
      finally { setChecking(false); }
    }}>
      <label>Workspace access key <input type="password" autoComplete="off" value={key} onChange={e => setKey(e.target.value)} required style={{ display: "block", width: "100%", margin: "12px 0", padding: 12 }} /></label>
      <button className="primary" disabled={checking}>{checking ? "Checking…" : "Open workspace"}</button>
      {error && <p role="alert">{error}</p>}
    </form>
  </section></main>;
}

createRoot(document.getElementById("root")!).render(<PrivateWorkspace />);
