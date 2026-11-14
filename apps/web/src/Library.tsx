import { useMemo, useState } from "react";
import { ArrowRight, Play, RefreshCw, Search, Upload } from "lucide-react";
import type { LibraryRun } from "./main";
import { isModelRun, matchesClip, shortDate } from "./ClipSwitcher";
import { clock, num, pct } from "./format";

type Props = {
  runs: LibraryRun[];
  currentRunId?: string;
  loading: boolean;
  error: string;
  showSynthetic: boolean;
  onShowSynthetic: (show: boolean) => void;
  onRefresh: () => void;
  onOpen: (runId: string) => void;
  onSample: () => void;
  onUpload: () => void;
  uploadDisabled: boolean;
  uploadTitle: string;
  /** Starts a new synthetic interface demo. Absent when the workspace requires an access key. */
  onDemo?: () => void;
  busy: boolean;
  /** Reasons the analysis engine reports it cannot run, from /capabilities. */
  blockers?: string[];
};

/** Up to three numbers from the library summary. Missing values are left out, never shown as zero. */
function cardStats(run: LibraryRun) {
  const summary = run.summary;
  if (!summary) return [];
  const ball = summary.selected_ball_candidate_coverage_pct ?? summary.ball_candidate_coverage_pct;
  return [
    summary.temporary_tracks != null && { label: "Players tracked", value: num(summary.temporary_tracks, 0), title: "Temporary tracks. Track IDs are not verified players." },
    summary.mean_visible_players != null && { label: "Players in view", value: num(summary.mean_visible_players), title: "Mean player detections per analysed frame." },
    ball != null && { label: "Ball detected", value: pct(ball), title: "Share of analysed frames with a ball candidate. Not an accuracy score." },
  ].filter(Boolean) as { label: string; value: string; title: string }[];
}

export function Library({ runs, currentRunId, loading, error, showSynthetic, onShowSynthetic, onRefresh, onOpen, onSample, onUpload, uploadDisabled, uploadTitle, onDemo, busy, blockers }: Props) {
  const [query, setQuery] = useState("");
  const real = useMemo(() => runs.filter(isModelRun), [runs]);
  const synthetic = useMemo(() => runs.filter(run => !isModelRun(run)), [runs]);
  // Real clips first; synthetic demos only on request.
  const visible = [...real, ...(showSynthetic ? synthetic : [])].filter(run => matchesClip(run, query));
  const firstLoad = loading && !runs.length;

  return (
    <div className="page library">
      <header className="page__head">
        <div>
          <h1 className="page__title">Library {!firstLoad && !error && <span>{real.length} {real.length === 1 ? "clip" : "clips"}</span>}</h1>
          <p className="page__lead">Opening a saved clip uses its existing analysis. It does not reprocess the video.</p>
        </div>
      </header>

      <div className="library__tools">
        <label className="searchbox">
          <span className="sr-only">Find a saved clip</span>
          <Search size={15} aria-hidden="true" />
          <input className="input" type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search clip names" />
        </label>
        <label className="check">
          <input type="checkbox" checked={showSynthetic} onChange={event => onShowSynthetic(event.target.checked)} />
          Show synthetic demos{synthetic.length ? ` (${synthetic.length})` : ""}
        </label>
        <button type="button" className="btn btn--secondary" onClick={onRefresh} disabled={loading}>
          <RefreshCw size={15} aria-hidden="true" /> {loading ? "Refreshing…" : "Refresh"}
        </button>
        {showSynthetic && onDemo && (
          <button type="button" className="btn btn--quiet" onClick={onDemo} disabled={busy} title="Creates a new synthetic demo run for exploring the interface. It is not an analysis of real footage.">
            <Play size={15} aria-hidden="true" /> Run a synthetic demo
          </button>
        )}
      </div>

      {error && (
        <div className="notice notice--error" role="alert">
          <span>Could not load saved clips. {error}</span>
          <button type="button" className="btn btn--secondary btn--sm" onClick={onRefresh}>Retry library</button>
        </div>
      )}

      <ul className="library__grid">
        <li>
          <div className="uploadtile">
            <div>
              <h2>Add a match clip</h2>
              <p>Hosted test clips: up to 2 minutes / 250 MiB. Target: a 45-minute match half.</p>
              {blockers?.length ? <p role="status">{blockers.join(" ")}</p> : null}
            </div>
            <div className="uploadtile__actions">
              <button type="button" className="btn btn--primary" onClick={onUpload} disabled={uploadDisabled} title={uploadTitle}><Upload size={16} aria-hidden="true" /> Upload video</button>
              <button type="button" className="btn btn--secondary" onClick={onSample} disabled={busy}>Open sample · 30 s</button>
            </div>
          </div>
        </li>

        {firstLoad && [0, 1, 2].map(index => <li key={index} aria-hidden="true"><div className="skeleton" /></li>)}

        {visible.map(run => {
          const stats = cardStats(run);
          const model = isModelRun(run);
          const current = run.run_id === currentRunId;
          return (
            <li key={run.run_id}>
              <button type="button" className="clipcard" aria-current={current ? "true" : undefined} onClick={() => onOpen(run.run_id)}>
                <span className="clipcard__top">
                  <span className={`tag ${model ? "" : "tag--synthetic"}`}>{model ? "Model analysis" : "Synthetic demo"}</span>
                  <span className="clipcard__date">{shortDate(run.created_at)}</span>
                </span>
                <span className="clipcard__name">{run.metadata.filename}</span>
                <span className="clipcard__meta">{clock(run.metadata.duration_seconds)} · {run.metadata.width} × {run.metadata.height}</span>
                {stats.length > 0 && (
                  <span className="clipcard__stats">
                    {stats.map(stat => <span key={stat.label} title={stat.title}><b>{stat.value}</b><span>{stat.label}</span></span>)}
                  </span>
                )}
                <span className="clipcard__foot"><span>{current ? "Open now" : "Open clip"}</span><ArrowRight size={16} aria-hidden="true" /></span>
              </button>
            </li>
          );
        })}
      </ul>

      {firstLoad && <p className="sr-only" role="status">Loading your clip library…</p>}

      {!loading && !error && !visible.length && (
        <div className="library__state" role="status">
          <h2>{runs.length ? (query.trim() ? "No matching clips" : "Only synthetic demos here") : "Your library is ready for its first clip"}</h2>
          <p>
            {runs.length
              ? query.trim()
                ? `Nothing matches “${query.trim()}”. Try a different file name${!showSynthetic && synthetic.length ? ", or show synthetic demos" : ""}.`
                : "No analysed footage yet. Turn on “Show synthetic demos” to see the interface examples, or upload a clip."
              : "Upload footage or open the sample to explore a completed analysis."}
          </p>
          {query.trim() && <button type="button" className="btn btn--secondary" onClick={() => setQuery("")}>Clear search</button>}
        </div>
      )}
    </div>
  );
}
