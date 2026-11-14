import { useEffect, useRef, useState, type FormEvent } from "react";
import { AlertTriangle } from "lucide-react";
import { useKits } from "./kits";
import { Disclosure, Field, WatchButton } from "./ui";

type Context = { schema_version: string; run_id: string; source_sha256: string | null; revision: number; kit_mappings: Record<string, string>; dead_ball_intervals: [number, number][]; confirmed: boolean };
type Draft = { context: Context; saved: Context; start: string; end: string };
const drafts = new Map<string, Draft>();
const comparable = (state: Context) => JSON.stringify({ revision: state.revision, kit_mappings: state.kit_mappings, dead_ball_intervals: state.dead_ball_intervals });
const dirtyDraft = (draft: Draft) => comparable(draft.context) !== comparable(draft.saved) || Boolean(draft.start || draft.end);
const warn = (event: BeforeUnloadEvent) => { if ([...drafts.values()].some(dirtyDraft)) { event.preventDefault(); event.returnValue = ""; } };
window.addEventListener("beforeunload", warn);
if (import.meta.hot) import.meta.hot.dispose(() => window.removeEventListener("beforeunload", warn));

type Props = { runId: string; sourceSha?: string; duration: number; kits: string[]; api: string; authHeaders: () => Record<string, string>; onRefresh: (runId: string) => Promise<void> };

/**
 * Where a person confirms real team names and stoppages. This is the only place a kit group gains a
 * team name, and the only way time is removed from playing time.
 */
export function SequenceContextPanel({ runId, sourceSha, duration, kits, api, authHeaders, onRefresh }: Props) {
  const kitNames = useKits();
  const key = `${api}/runs/${runId}`; const epoch = useRef(0);
  const [context, setContext] = useState<Context | null>(null); const [saved, setSaved] = useState<Context | null>(null);
  const [pending, setPending] = useState(false); const [refreshing, setRefreshing] = useState(false); const [refreshError, setRefreshError] = useState(""); const [error, setError] = useState(""); const [conflict, setConflict] = useState(false); const [restored, setRestored] = useState(false); const [reload, setReload] = useState(0);
  const [confirmed, setConfirmed] = useState(false); const [start, setStart] = useState(""); const [end, setEnd] = useState("");
  const dirty = Boolean(context && saved && comparable(context) !== comparable(saved));
  useEffect(() => { if (context && saved) drafts.set(key, { context, saved, start, end }); }, [key, context, saved, start, end]);
  useEffect(() => {
    const request = ++epoch.current; const abort = new AbortController(); setPending(true); setError(""); setConflict(false); setConfirmed(false);
    fetch(`${api}/runs/${encodeURIComponent(runId)}/sequence-context`, { headers: authHeaders(), signal: abort.signal })
      .then(async response => { const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Context load failed (${response.status}).`); if (body.run_id !== runId || (sourceSha && body.source_sha256 !== sourceSha)) throw new Error("Context source does not match this clip. Nothing was loaded."); return body as Context; })
      .then(body => { if (request !== epoch.current) return; const draft = reload === 0 ? drafts.get(key) : undefined; const restore = draft && draft.context.source_sha256 === body.source_sha256 && dirtyDraft(draft); setSaved(body); if (restore && draft) { const dataChanged = comparable(draft.context) !== comparable(draft.saved); setContext(dataChanged ? draft.context : body); setStart(draft.start); setEnd(draft.end); setRestored(true); if (dataChanged && draft.context.revision !== body.revision) { setConflict(true); setError("A newer saved version of this context exists. Your per-clip draft is retained; reload only to discard it."); } } else { setContext(body); setStart(""); setEnd(""); setRestored(false); } })
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
    if (!start.trim() || !end.trim() || !Number.isFinite(a) || !Number.isFinite(b) || a < 0 || b <= a || b > duration) { setError(`Stoppage must be inside 0–${duration} s, with end after start.`); return; }
    if (context.dead_ball_intervals.some(([x, y]) => a < y && b > x)) { setError("This stoppage overlaps an existing interval. Adjust or remove that interval first."); return; }
    update({ dead_ball_intervals: [...context.dead_ball_intervals, [a, b] as [number, number]].sort((x, y) => x[0] - y[0]) }); setStart(""); setEnd(""); setError("");
  }

  const status = pending ? "Working…" : dirty || start || end ? "Unsaved context" : saved ? `Saved revision ${saved.revision}` : "Not loaded";

  return (
    <Disclosure variant="panel" summary="Team & stoppage context" meta={status}>
      <div className="stack" aria-busy={pending || refreshing}>
        <p className="muted">Confirm real team names and stoppages against the footage. This adjusts the automatic possession estimates; it never approves football event records. Missing ball detections are not stoppages.</p>
        {error && <p className="notice notice--error" role="alert"><AlertTriangle size={16} aria-hidden="true" /><span>{error}</span></p>}
        {restored && <p className="note note--boxed note--flood" role="status">Your unsaved context draft for this source was restored. Confirmation must be checked again before saving.</p>}
        {(conflict || (!context && !pending)) && <div><button type="button" className="btn btn--secondary" onClick={() => setReload(value => value + 1)} disabled={pending}>{conflict ? "Reload saved context · discard edits" : "Retry context load"}</button></div>}
        {context && (
          <fieldset disabled={pending || refreshing} className="review__work">
            <legend className="sr-only">Context for this {duration.toLocaleString(undefined, { maximumFractionDigits: 3 })} s source</legend>
            <div className="stack stack--sm">
              <h4>Team names</h4>
              <div className="formgrid formgrid--two">
                {kits.map(kit => (
                  <Field key={kit} label={<span title={`Group id in the data: ${kit}`}>{kitNames.anonymous(kit)} is…</span>}>
                    <input maxLength={80} value={context.kit_mappings[kit] ?? ""} onChange={event => { const next = { ...context.kit_mappings }; if (event.target.value) next[kit] = event.target.value; else delete next[kit]; update({ kit_mappings: next }); }} placeholder="Unmapped · enter a verified team name" />
                  </Field>
                ))}
              </div>
              {!kits.length && <p className="note">No established kit groups in this source. Team mappings are unavailable.</p>}
            </div>

            <div className="stack stack--sm">
              <h4>Stoppages (confirmed)</h4>
              <p className="note">Only intervals explicitly confirmed here are taken out of playing time (the non-dead denominator). Leave blank when unverified.</p>
              <form onSubmit={addStoppage} className="formgrid formgrid--action">
                <Field label="Stoppage start (seconds)"><input type="number" min="0" max={duration} step="any" value={start} onChange={event => { setStart(event.target.value); setConfirmed(false); }} /></Field>
                <Field label="Stoppage end (seconds)"><input type="number" min="0" max={duration} step="any" value={end} onChange={event => { setEnd(event.target.value); setConfirmed(false); }} /></Field>
                <span className="row">
                  <button className="btn btn--secondary" type="submit">Add stoppage</button>
                  <button className="btn btn--quiet" type="button" onClick={() => { setStart(""); setEnd(""); setConfirmed(false); }}>Cancel interval form</button>
                </span>
              </form>
              <ul className="itemlist">
                {context.dead_ball_intervals.map(([a, b], index) => (
                  <li key={`${a}-${b}`}>
                    <span className="mono">{a}–{b} s</span>
                    <span className="row">
                      <WatchButton at={a} label="Watch stoppage" />
                      <button type="button" className="btn btn--secondary btn--sm btn--danger" onClick={() => update({ dead_ball_intervals: context.dead_ball_intervals.filter((_, row) => row !== index) })}>Remove stoppage</button>
                    </span>
                  </li>
                ))}
              </ul>
              {(start || end) && <p className="note note--boxed note--flood">Add or cancel the interval form before saving context.</p>}
            </div>

            <label className="check"><input type="checkbox" checked={confirmed} onChange={event => setConfirmed(event.target.checked)} />I checked these team mappings and stoppages against this source video.</label>
            <div><button type="button" className="btn btn--primary" disabled={!dirty || !confirmed || conflict || Boolean(start || end)} onClick={save}>Save context & refresh estimates</button></div>
          </fieldset>
        )}
        {saved && !dirty && !refreshError && <p className="note" role="status">Saved on the server as version {saved.revision} of this context. {refreshing ? "Refreshing selected-source estimates…" : "Only confirmed context changes affect the next estimate refresh."}</p>}
        {refreshError && (
          <div className="notice notice--error" role="alert">
            <AlertTriangle size={16} aria-hidden="true" />
            <span>{refreshError}</span>
            <button type="button" className="btn btn--secondary btn--sm" disabled={refreshing || pending} onClick={refreshResult}>Retry estimate refresh · context already saved</button>
          </div>
        )}
        <p className="note">Per-source drafts and unfinished intervals stay in this tab’s memory when you switch clips. Closing or refreshing warns for any unsaved context. Saved mappings remain separate from saved human event reviews.</p>
      </div>
    </Disclosure>
  );
}
