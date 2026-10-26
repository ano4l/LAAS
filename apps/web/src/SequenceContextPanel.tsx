import { useEffect, useId, useRef, useState, type FormEvent } from "react";

type Context = { schema_version: string; run_id: string; source_sha256: string | null; revision: number; kit_mappings: Record<string, string>; dead_ball_intervals: [number, number][]; confirmed: boolean };
type Draft = { context: Context; saved: Context; start: string; end: string };
const drafts = new Map<string, Draft>();
const comparable = (state: Context) => JSON.stringify({ revision: state.revision, kit_mappings: state.kit_mappings, dead_ball_intervals: state.dead_ball_intervals });
const dirtyDraft = (draft: Draft) => comparable(draft.context) !== comparable(draft.saved) || Boolean(draft.start || draft.end);
const warn = (event: BeforeUnloadEvent) => { if ([...drafts.values()].some(dirtyDraft)) { event.preventDefault(); event.returnValue = ""; } };
window.addEventListener("beforeunload", warn);
if (import.meta.hot) import.meta.hot.dispose(() => window.removeEventListener("beforeunload", warn));

export function SequenceContextPanel({ runId, sourceSha, duration, kits, api, authHeaders, onRefresh, onSeek }: { runId: string; sourceSha?: string; duration: number; kits: string[]; api: string; authHeaders: () => Record<string, string>; onRefresh: (runId: string) => Promise<void>; onSeek: (seconds: number) => void }) {
  const id = useId(); const key = `${api}/runs/${runId}`; const epoch = useRef(0);
  const [context, setContext] = useState<Context | null>(null); const [saved, setSaved] = useState<Context | null>(null);
  const [pending, setPending] = useState(false); const [refreshing, setRefreshing] = useState(false); const [refreshError, setRefreshError] = useState(""); const [error, setError] = useState(""); const [conflict, setConflict] = useState(false); const [restored, setRestored] = useState(false); const [reload, setReload] = useState(0);
  const [confirmed, setConfirmed] = useState(false); const [start, setStart] = useState(""); const [end, setEnd] = useState("");
  const dirty = Boolean(context && saved && comparable(context) !== comparable(saved));
  useEffect(() => { if (context && saved) drafts.set(key, { context, saved, start, end }); }, [key, context, saved, start, end]);
  useEffect(() => {
    const request = ++epoch.current; const abort = new AbortController(); setPending(true); setError(""); setConflict(false); setConfirmed(false);
    fetch(`${api}/runs/${encodeURIComponent(runId)}/sequence-context`, { headers: authHeaders(), signal: abort.signal })
      .then(async response => { const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Context load failed (${response.status}).`); if (body.run_id !== runId || (sourceSha && body.source_sha256 !== sourceSha)) throw new Error("Context source does not match this clip. Nothing was loaded."); return body as Context; })
      .then(body => { if (request !== epoch.current) return; const draft = reload === 0 ? drafts.get(key) : undefined; const restore = draft && draft.context.source_sha256 === body.source_sha256 && dirtyDraft(draft); setSaved(body); if (restore && draft) { const dataChanged = comparable(draft.context) !== comparable(draft.saved); setContext(dataChanged ? draft.context : body); setStart(draft.start); setEnd(draft.end); setRestored(true); if (dataChanged && draft.context.revision !== body.revision) { setConflict(true); setError("A newer context revision exists. Your per-clip draft is retained; reload only to discard it."); } } else { setContext(body); setStart(""); setEnd(""); setRestored(false); } })
      .catch(cause => { if (!abort.signal.aborted && request === epoch.current) setError(String(cause)); })
      .finally(() => { if (request === epoch.current) setPending(false); });
    return () => { abort.abort(); epoch.current++; };
  }, [runId, sourceSha, api, reload]);
  function update(change: Partial<Context>) { setContext(current => current ? { ...current, ...change } : current); setConfirmed(false); setRefreshError(""); }
  async function refreshResult() {
    const request = epoch.current; setRefreshing(true); setRefreshError("");
    try { await onRefresh(runId); }
    catch (cause) { if (request === epoch.current) setRefreshError(`Context is saved, but updated estimates could not be loaded. ${String(cause)}`); }
    finally { if (request === epoch.current) setRefreshing(false); }
  }
  async function save() {
    if (!context || !confirmed || !dirty || pending || conflict || start || end) return;
    const request = epoch.current; setPending(true); setError(""); setRefreshError("");
    const mappings = Object.fromEntries(Object.entries(context.kit_mappings).filter(([, name]) => name.trim()).map(([kit, name]) => [kit, name.trim()]));
    try {
      const response = await fetch(`${api}/runs/${encodeURIComponent(runId)}/sequence-context`, { method: "POST", headers: { ...authHeaders(), "Content-Type": "application/json" }, body: JSON.stringify({ revision: context.revision, kit_mappings: mappings, dead_ball_intervals: context.dead_ball_intervals, confirmed: true }) });
      const body = await response.json().catch(() => ({})); if (request !== epoch.current) return;
      if (response.status === 409) { setConflict(true); throw new Error("Another session changed this context. Your edits are retained. Reload the saved context only when ready to discard them."); }
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Context save failed (${response.status}). Edits are retained.`);
      if (body.run_id !== runId || (sourceSha && body.source_sha256 !== sourceSha)) throw new Error("Saved context response has a different source; it was not applied locally.");
      setContext(body); setSaved(body); setConfirmed(false); setRestored(false);
      await refreshResult();
    } catch (cause) { if (request === epoch.current) setError(String(cause)); }
    finally { if (request === epoch.current) setPending(false); }
  }
  function addStoppage(event: FormEvent) {
    event.preventDefault(); if (!context) return; const a = Number(start), b = Number(end);
    if (!start.trim() || !end.trim() || !Number.isFinite(a) || !Number.isFinite(b) || a < 0 || b <= a || b > duration) { setError(`Stoppage must be inside 0–${duration}s, with end after start.`); return; }
    if (context.dead_ball_intervals.some(([x, y]) => a < y && b > x)) { setError("This stoppage overlaps an existing interval. Adjust or remove that interval first."); return; }
    update({ dead_ball_intervals: [...context.dead_ball_intervals, [a, b] as [number, number]].sort((x, y) => x[0] - y[0]) }); setStart(""); setEnd(""); setError("");
  }
  return <details className="sequence-context" aria-labelledby={`${id}-title`}><summary id={`${id}-title`}>Team & stoppage context <span>{pending ? "Working…" : dirty || start || end ? "Unsaved context" : saved ? `Saved revision ${saved.revision}` : "Not loaded"}</span></summary><div className="sequence-context-body" aria-busy={pending || refreshing}>
    <p>Confirm real team names and stoppages against the footage. This adjusts automatic sequence estimates; it never approves football event records. Missing ball detections are not stoppages.</p>
    {error && <p className="review-panel__error" role="alert">{error}</p>}{restored && <p role="status">Your unsaved context draft for this source was restored. Confirmation must be checked again before saving.</p>}
    {(conflict || (!context && !pending)) && <button className="ghost" onClick={() => setReload(value => value + 1)} disabled={pending}>{conflict ? "Reload saved context · discard edits" : "Retry context load"}</button>}
    {context && <fieldset disabled={pending || refreshing}><legend>Context for this {duration.toLocaleString(undefined, { maximumFractionDigits: 3 })}s source</legend><div className="sequence-context-mappings">{kits.map(kit => <label key={kit}>{kit}<input maxLength={80} value={context.kit_mappings[kit] ?? ""} onChange={event => { const next = { ...context.kit_mappings }; if (event.target.value) next[kit] = event.target.value; else delete next[kit]; update({ kit_mappings: next }); }} placeholder="Unmapped · enter a verified team name" /></label>)}</div>{!kits.length && <p>No established kit groups in this source. Team mappings are unavailable.</p>}
      <h5>Confirmed dead-ball intervals</h5><p>Only intervals explicitly confirmed here are excluded from the non-dead denominator. Leave blank when unverified.</p><form onSubmit={addStoppage} className="sequence-context-range"><label>Stoppage start (seconds)<input type="number" min="0" max={duration} step="any" value={start} onChange={event => { setStart(event.target.value); setConfirmed(false); }} /></label><label>Stoppage end (seconds)<input type="number" min="0" max={duration} step="any" value={end} onChange={event => { setEnd(event.target.value); setConfirmed(false); }} /></label><button className="ghost" type="submit">Add stoppage</button><button className="ghost" type="button" onClick={() => { setStart(""); setEnd(""); setConfirmed(false); }}>Cancel interval form</button></form>
      <ul className="sequence-context-intervals">{context.dead_ball_intervals.map(([a, b], index) => <li key={`${a}-${b}`}><span>{a}s–{b}s</span><button type="button" className="ghost" onClick={() => onSeek(a)}>Watch stoppage</button><button type="button" className="ghost" onClick={() => update({ dead_ball_intervals: context.dead_ball_intervals.filter((_, row) => row !== index) })}>Remove stoppage</button></li>)}</ul>
      {(start || end) && <p>Add or cancel the interval form before saving context.</p>}<label className="sequence-context-confirm"><input type="checkbox" checked={confirmed} onChange={event => setConfirmed(event.target.checked)} />I checked these team mappings and stoppages against this source video.</label><button className="primary" disabled={!dirty || !confirmed || conflict || Boolean(start || end)} onClick={save}>Save context & refresh estimates</button>
    </fieldset>}
    {saved && !dirty && !refreshError && <p className="sequence-context-status" role="status">Server-saved context revision {saved.revision}. {refreshing ? "Refreshing selected-source estimates…" : "Only confirmed context changes affect the next estimate refresh."}</p>}
    {refreshError && <div role="alert"><p>{refreshError}</p><button className="ghost" disabled={refreshing || pending} onClick={refreshResult}>Retry estimate refresh · context already saved</button></div>}
    <p className="insights-caption">Per-source drafts and unfinished intervals stay in this tab’s memory when you switch clips. Closing or refreshing warns for any unsaved context. Saved mappings remain separate from saved human event reviews.</p>
  </div></details>;
}
