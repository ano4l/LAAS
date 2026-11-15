import { Suspense, lazy, useEffect, useRef, useState, type FormEvent } from "react";
import { createRoot } from "react-dom/client";
import { AlertTriangle, Info } from "lucide-react";
import "@fontsource-variable/big-shoulders-display";
import "@fontsource-variable/hanken-grotesk";
import "@fontsource-variable/jetbrains-mono";
import "./styles.css";
import { AppBar } from "./AppBar";
import { Library } from "./Library";
import type { VideoInsights } from "./VideoInsights";
import type { VideoMeasurements } from "./VideoMeasurements";
import { Workspace } from "./Workspace";

/* ---------- Configuration ---------- */

const configuredApiUrl = import.meta.env.VITE_API_URL?.trim();
const API = configuredApiUrl || (window.location.port === "5173" ? "http://127.0.0.1:8000" : window.location.origin);
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";
const REQUIRE_ACCESS_KEY = import.meta.env.VITE_REQUIRE_ACCESS_KEY === "true";
const authHeaders = (): Record<string, string> => {
  const key = sessionStorage.getItem("laas-access-key");
  return key ? { Authorization: `Bearer ${key}` } : {};
};

// The synthetic-demo dashboard and its data factories load on demand; real clips never need them.
const loadLegacyDemo = () => import("./LegacyDemo");
const LegacyDemo = lazy(() => loadLegacyDemo().then(module => ({ default: module.LegacyDemo })));

/* ---------- Shared types ---------- */

export type Job = {
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

export type Track = {
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
  detection_summary?: {
    tracker?: string;
    analyzed_frames: number;
    decoded_frames: number;
    sample_fps: number;
    image_size?: number;
    ball_processing?: string;
    unique_tracks: number;
    ball_detected_frames: number;
    processing_seconds: number;
    device: string;
    model: string;
    detections_by_class: Record<string, number>;
    tracks: { track_id: string; object_type: string; observations: number; first_seen_ms: number; last_seen_ms: number }[];
    track_linking?: { tracks_before: number; identities_after: number; links_accepted: number };
    linked_tracks?: { linked_track_id: string; track_ids: string[]; object_type: string; team: string | null; observations: number; first_seen_ms: number; last_seen_ms: number }[];
  };
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
  files: { annotated_video: string; source_video?: string; tracking_json?: string; metrics_json?: string; report_json?: string; manifest_json?: string };
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

export type LibraryRun = {
  run_id: string;
  engine?: string;
  created_at: string;
  metadata: AnalysisResult["metadata"];
  metrics: { data_quality?: Record<string, string | number> };
  summary?: { analyzed_frames?: number | null; temporary_tracks?: number | null; mean_visible_players?: number | null; ball_candidate_coverage_pct?: number | null; selected_ball_candidate_coverage_pct?: number | null };
};

export type Capabilities = {
  engines: {
    fallback: { available: boolean; description: string };
    yolo: { blockers?: string[]; available: boolean; compute_provider?: "local" | "modal"; model_exists?: boolean; ultralytics_installed?: boolean; opencv_available?: boolean; model_path?: string };
  };
};

/* ---------- API ---------- */

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

/* ---------- App: state and data flow ---------- */

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
  // Presentation only: whether the library is in front of the open clip, and whether demos are listed.
  const [showLibrary, setShowLibrary] = useState(false);
  const [showSynthetic, setShowSynthetic] = useState(false);
  const activeProcessing = activeJob?.status === "queued" || activeJob?.status === "running";

  useEffect(() => {
    if (DEMO_MODE) {
      loadLegacyDemo().then(({ createDemoAnalysis, demoJob, demoRun, demoCapabilities }) => {
        const result = createDemoAnalysis();
        setJob(demoJob(result, "Static Vercel demo"));
        setRuns([demoRun(result)]);
        setCapabilities(demoCapabilities());
      });
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
        if (!cancelled) {
          setActiveJob({ ...next, filename: activeJob.filename });
          setActiveJobError("");
          setJob(current => current?.id === next.id ? next : current);
        }
      } catch (err) {
        if (!cancelled) setActiveJobError(String(err));
      } finally {
        polling = false;
      }
    }, 900);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [activeJob?.id, activeJob?.status]);

  useEffect(() => { if (activeJob?.status === "complete") refreshRuns(); }, [activeJob?.status]);

  async function refreshRuns(selectInitial = false) {
    if (DEMO_MODE) {
      const { createDemoAnalysis, demoRun } = await loadLegacyDemo();
      setRuns([demoRun(createDemoAnalysis())]);
      setLibraryLoading(false);
      return;
    }
    const currentLibraryRequest = ++libraryEpoch.current;
    setLibraryLoading(true);
    setLibraryError("");
    try {
      const library = await request<{ runs: LibraryRun[]; jobs: LibraryJob[] }>("/library");
      if (currentLibraryRequest !== libraryEpoch.current) return;
      setLibraryRuns(library.runs);
      if (selectInitial && selectionEpoch.current === 0) {
        const next = library.jobs.find(item => item.status === "running" || item.status === "queued") ?? library.jobs.find(item => item.engine === "yolo_bytetrack" && item.status === "complete");
        if (next) {
          const epoch = ++selectionEpoch.current;
          setSelectionLoading("Loading selected clip…");
          try {
            const detail = await request<Job>(`/analysis-jobs/${next.id}`);
            if (next.status === "running" || next.status === "queued") setActiveJob({ ...detail, filename: next.filename });
            if (epoch === selectionEpoch.current) setJob(detail);
          } catch (cause) {
            if (epoch === selectionEpoch.current) { setError(String(cause)); setSelectionFailure(next.run_id ?? "sample"); }
          } finally {
            if (epoch === selectionEpoch.current) setSelectionLoading("");
          }
        } else {
          const firstReal = library.runs.find(item => item.engine === "yolo_bytetrack");
          if (firstReal) await openRun(firstReal.run_id);
        }
      }
    } catch (cause) {
      if (currentLibraryRequest === libraryEpoch.current) setLibraryError(String(cause));
    } finally {
      if (currentLibraryRequest === libraryEpoch.current) setLibraryLoading(false);
    }
  }

  async function startDemo() {
    selectionEpoch.current++;
    if (DEMO_MODE) {
      const { createDemoAnalysis, demoJob } = await loadLegacyDemo();
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
      const { createDemoAnalysis, demoJob } = await loadLegacyDemo();
      setJob(demoJob(createDemoAnalysis(), "Loaded static demo"));
      return;
    }
    const epoch = ++selectionEpoch.current;
    setSelectionLoading("Loading clip…");
    setSelectionFailure("");
    setError("");
    const result = await request<AnalysisResult>(`/runs/${runId}`).catch(err => {
      if (epoch === selectionEpoch.current) { setError(String(err)); setSelectionFailure(runId); }
      return null;
    });
    if (epoch !== selectionEpoch.current) return;
    setSelectionLoading("");
    if (!result) return;
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
    const uploadSelection = ++selectionEpoch.current;
    setSelectionLoading("");
    setBusy(true);
    setError("");
    setUploadPct(0);
    try {
      const body = new FormData();
      body.append("file", file);
      const next = await new Promise<Job>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API}/videos/upload`);
        for (const [name, value] of Object.entries(authHeaders())) xhr.setRequestHeader(name, value);
        xhr.upload.onprogress = event => { if (event.lengthComputable) setUploadPct(Math.round(event.loaded / event.total * 100)); };
        xhr.onerror = () => reject(new Error("Cannot reach the analysis API. Check that the backend is running."));
        xhr.onload = () => {
          try {
            const payload = JSON.parse(xhr.responseText);
            if (xhr.status >= 400) reject(new Error(payload.detail || "Upload failed")); else resolve(payload);
          } catch { reject(new Error("The API returned an invalid response.")); }
        };
        xhr.send(body);
      });
      setActiveJob({ ...next, filename: file.name });
      if (uploadSelection === selectionEpoch.current) { setJob(next); setShowLibrary(false); }
      setActiveModule("Match Review");
    } catch (err) { setError(String(err)); }
    finally {
      setBusy(false);
      setUploadPct(null);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function retry() {
    if (!job) return;
    if (activeProcessing || busy) return;
    const expectedSelection = selectionEpoch.current;
    try {
      setError("");
      const next = await request<Job>(`/analysis-jobs/${job.id}/retry`, { method: "POST" });
      setActiveJob(next);
      if (expectedSelection === selectionEpoch.current) setJob(next);
    } catch (err) { setError(String(err)); }
  }

  async function openSample() {
    const epoch = ++selectionEpoch.current;
    setSelectionLoading("Opening full sample…");
    setSelectionFailure("");
    setError("");
    try {
      const sample = await request<Job>("/sample");
      if (epoch === selectionEpoch.current) setJob(sample);
    } catch (err) {
      if (epoch === selectionEpoch.current) { setError(String(err)); setSelectionFailure("sample"); }
    } finally {
      if (epoch === selectionEpoch.current) setSelectionLoading("");
    }
  }

  /* ---------- Presentation ---------- */

  // Opening something from the interface also brings the clip in front of the library.
  const openFromUi = (runId: string) => { openRun(runId).then(() => setShowLibrary(false)); };
  const sampleFromUi = () => { openSample().then(() => setShowLibrary(false)); };
  const demoFromUi = () => { startDemo().then(() => setShowLibrary(false)); };
  const openActiveJob = () => {
    if (!activeJob) return;
    selectionEpoch.current++;
    setSelectionLoading("");
    setJob(activeJob);
    setWorkspaceOpen(value => value + 1);
    setShowLibrary(false);
  };

  const result = job?.result ?? null;
  const modelResult = result?.engine === "yolo_bytetrack";
  const libraryVisible = !DEMO_MODE && (showLibrary || !job);
  const uploadDisabled = busy || activeProcessing || libraryLoading;
  const uploadTitle = activeProcessing ? "One video is processing. Browse saved results while it finishes." : libraryLoading ? "Checking existing analyses before accepting another upload." : "Upload football footage";
  const jobStatus = uploadPct !== null ? `Uploading ${uploadPct}%` : activeJob && activeProcessing ? `${activeJob.stage} · ${Math.round(activeJob.progress * 100)}%` : null;

  return (
    <div className="app">
      <AppBar
        clipName={result?.metadata.filename ?? job?.filename ?? null}
        currentRunId={result?.run_id}
        synthetic={Boolean(result) && !modelResult && !libraryVisible}
        runs={libraryRuns}
        libraryLoading={libraryLoading}
        demoMode={DEMO_MODE}
        showSynthetic={showSynthetic}
        onShowSynthetic={setShowSynthetic}
        onOpen={openFromUi}
        onLibrary={() => setShowLibrary(true)}
        onSample={sampleFromUi}
        sampleDisabled={busy}
        fileRef={fileRef}
        onFile={upload}
        uploadDisabled={uploadDisabled}
        uploadTitle={uploadTitle}
        uploadPct={uploadPct}
        jobStatus={jobStatus}
        onOpenJob={activeJob && activeProcessing ? openActiveJob : undefined}
      />

      <main className="app__main">
        <div className="app__notices">
          {uploadPct !== null && (
            <p className="notice" role="status">
              <span className="spinner" aria-hidden="true" />
              <span>Uploading footage · {uploadPct}%. Another upload is disabled until this job finishes. You can keep browsing saved clips.</span>
            </p>
          )}
          {activeJob && (
            <div className={`jobbanner ${!activeProcessing && activeJob.status !== "complete" ? "jobbanner--attention" : ""}`} role="status">
              <div className="jobbanner__text">
                <strong>{activeProcessing ? "Analysis running in the background" : activeJob.status === "complete" ? "Your analysis is ready" : "Analysis needs attention"}</strong>
                <span>
                  {activeJob.filename ?? activeJob.result?.metadata.filename ?? "Uploaded video"} · {activeProcessing
                    ? `${activeJob.stage} · ${Math.round(activeJob.progress * 100)}% · One analysis at a time; saved clips remain available.`
                    : activeJob.status === "complete" ? "Open the full result to review it." : activeJob.error}
                </span>
                {activeJobError && <span>Could not refresh status. Retrying automatically. {activeJobError}</span>}
              </div>
              <button type="button" className="btn btn--secondary" onClick={openActiveJob}>Open {activeProcessing ? "processing job" : "result"}</button>
              {!activeProcessing && <button type="button" className="btn btn--quiet" onClick={() => setActiveJob(null)}>Dismiss status</button>}
              {activeProcessing && <div className="jobbanner__progress" aria-hidden="true"><span style={{ width: `${Math.round(activeJob.progress * 100)}%` }} /></div>}
            </div>
          )}
          {selectionLoading && (
            <p className="notice" role="status">
              <span className="spinner" aria-hidden="true" />
              <span>{selectionLoading} {result ? "Your current clip stays available until the selection is ready." : ""}</span>
            </p>
          )}
          {error && (
            <p className="notice notice--error" role="alert">
              <AlertTriangle size={16} aria-hidden="true" />
              <span>{error}</span>
              {selectionFailure && <button type="button" className="btn btn--secondary btn--sm" onClick={() => selectionFailure === "sample" ? sampleFromUi() : openFromUi(selectionFailure)}>Retry clip selection</button>}
            </p>
          )}
          {job?.error && (
            <div className="notice notice--error" role="alert">
              <AlertTriangle size={16} aria-hidden="true" />
              <span>{job.error}</span>
              {!REQUIRE_ACCESS_KEY && <button type="button" className="btn btn--secondary btn--sm" onClick={retry}>Retry analysis</button>}
            </div>
          )}
          {libraryError && !libraryVisible && (
            <p className="notice notice--error" role="alert">
              <AlertTriangle size={16} aria-hidden="true" />
              <span>Could not load saved clips. {libraryError}</span>
              <button type="button" className="btn btn--secondary btn--sm" onClick={() => refreshRuns(!job)}>Retry library</button>
            </p>
          )}
        </div>

        {libraryVisible && (
          <Library
            runs={libraryRuns}
            currentRunId={result?.run_id}
            loading={libraryLoading}
            error={libraryError}
            showSynthetic={showSynthetic}
            onShowSynthetic={setShowSynthetic}
            onRefresh={() => refreshRuns(!job)}
            onOpen={openFromUi}
            onSample={sampleFromUi}
            onUpload={() => fileRef.current?.click()}
            uploadDisabled={uploadDisabled}
            uploadTitle={uploadTitle}
            onDemo={REQUIRE_ACCESS_KEY ? undefined : demoFromUi}
            busy={busy}
            blockers={capabilities?.engines.yolo.blockers}
          />
        )}

        {/* The clip stays mounted behind the library so the player position, filters and drafts survive. */}
        {!DEMO_MODE && result && modelResult && (
          <div hidden={libraryVisible}>
            <Workspace result={result} active={!libraryVisible} openRequest={workspaceOpen} api={API} authHeaders={authHeaders} onRefreshResult={refreshSelectedResult} />
          </div>
        )}

        {job && !result && !libraryVisible && <PendingJob job={job} onLibrary={() => setShowLibrary(true)} />}

        {job && result && !modelResult && !libraryVisible && (
          <Suspense fallback={<div className="page"><p className="notice" role="status"><span className="spinner" aria-hidden="true" /><span>Loading demo…</span></p></div>}>
            <LegacyDemo job={job} runs={runs} busy={busy} api={API} activeModule={activeModule} onSelectModule={selectModule} onDemo={startDemo} onOpen={openRun} />
          </Suspense>
        )}
      </main>

      <footer className="app__footer">
        <span>LAAS · Footage to evidence</span>
        <span>Hosted test: 2 min / 250 MiB · Match-half target: 45 min</span>
      </footer>
    </div>
  );
}

/** A selected job that has no result yet: still processing, or failed. Figures come from the job itself. */
function PendingJob({ job, onLibrary }: { job: Job; onLibrary: () => void }) {
  const processing = job.status === "queued" || job.status === "running";
  return (
    <div className="page">
      <section className="pending" role="status">
        <h1>{processing ? "This clip is being analysed" : job.status === "failed" ? "This analysis did not finish" : "No result for this clip yet"}</h1>
        {processing && <p className="pending__value">{Math.round(job.progress * 100)}%</p>}
        <p>
          {processing
            ? `${job.stage}. The model is finding players and the ball; results open here when it finishes. One analysis runs at a time, and saved clips stay available.`
            : job.status === "failed" ? "The error is shown above. Nothing is estimated for a clip that did not finish." : job.stage}
        </p>
        <div className="row">
          <button type="button" className="btn btn--secondary" onClick={onLibrary}><Info size={15} aria-hidden="true" /> Browse saved clips</button>
        </div>
      </section>
    </div>
  );
}

/* ---------- Access gate ---------- */

function PrivateWorkspace() {
  const [unlocked, setUnlocked] = useState(!REQUIRE_ACCESS_KEY || Boolean(sessionStorage.getItem("laas-access-key")));
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setChecking(true);
    setError("");
    try {
      const response = await fetch(`${API}/capabilities`, { headers: { Authorization: `Bearer ${key.trim()}` } });
      if (!response.ok) throw new Error("Access key was not accepted.");
      sessionStorage.setItem("laas-access-key", key.trim());
      setUnlocked(true);
    } catch (err) {
      setError(err instanceof TypeError ? "Cannot reach the analysis service. Try again shortly." : String(err));
    } finally {
      setChecking(false);
    }
  }

  if (unlocked) return <App />;
  return (
    <main className="gate">
      <section className="gate__card" aria-labelledby="gate-title">
        <p className="gate__brand" aria-hidden="true">LAAS</p>
        <div className="stack stack--sm">
          <h1 id="gate-title">Private workspace</h1>
          <p>Enter your workspace access key to upload footage and review analyses.</p>
        </div>
        <form onSubmit={submit}>
          <label className="field">
            Workspace access key
            <input type="password" autoComplete="off" value={key} onChange={event => setKey(event.target.value)} required />
          </label>
          <button type="submit" className="btn btn--primary btn--block" disabled={checking}>{checking ? "Checking…" : "Open workspace"}</button>
          {error && <p className="notice notice--error" role="alert"><AlertTriangle size={16} aria-hidden="true" /><span>{error}</span></p>}
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<PrivateWorkspace />);
