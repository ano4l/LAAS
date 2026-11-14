import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { AlertTriangle, ArrowDownToLine, Clock3, Save } from "lucide-react";
import { num, secs, sentence, words } from "./format";
import { usePlayback } from "./playback";
import { DataTable, Disclosure, EmptyState, EvidenceChip, Field, Pager, Panel, WatchButton } from "./ui";

const outcomes = {
  pass: ["completed", "incomplete", "unknown"], shot: ["goal", "on_target", "off_target", "blocked", "unknown"],
  tackle: ["won", "lost", "unknown"], interception: ["successful", "unknown"],
  dribble: ["successful", "unsuccessful", "unknown"], duel: ["won", "lost", "unknown"],
} as const;
type EventType = keyof typeof outcomes;
type Team = "home" | "away";
type Approval = "draft" | "approved";
type Annotation = { id: string; timestamp_seconds: number; type: EventType; team: Team; player: string | null; outcome: string; review_status: Approval; notes: string };
type Possession = { id: string; start_seconds: number; end_seconds: number; team: Team | "unknown"; review_status: Approval };
type TeamSummary = { name: string; event_count: number; passes: { recorded: number; completed: number; known_outcomes: number; outcome_coverage_pct: number | null; completion_pct: number | null };
  shots: { recorded: number; on_target: number; goals: number }; tackles: { recorded: number; won: number }; interceptions: number;
  dribbles: { recorded: number; successful: number }; duels: { recorded: number; won: number } };
type ReviewSummary = { scope: string; approved_events: number; draft_events: number; reviewed_seconds: number; reviewed_video_pct: number | null;
  teams: { home: TeamSummary | null; away: TeamSummary | null }; players: { team: Team; player: string; event_count: number; counts: Record<string, number> }[];
  possession: { home_seconds: number; away_seconds: number; unknown_seconds: number; home_share_pct: number | null; away_share_pct: number | null; known_coverage_pct: number | null } };
export type Review = { schema_version: string; run_id: string; revision: number; teams: { home: string; away: string }; coverage: [number, number][];
  events: Annotation[]; possession_intervals: Possession[]; summary: ReviewSummary };
type Props = { runId: string; duration: number; api: string; authHeaders: () => Record<string, string>; onSaved?: (saved: Review | null) => void; onUnsaved?: (unsaved: boolean) => void };
type CoverageForm = { start: string; end: string };
type PossessionForm = CoverageForm & { team: Possession["team"]; approved: boolean };
type ReviewDraft = { review: Review; saved: Review; event: Annotation; editing: boolean; eventDirty: boolean; coverage: CoverageForm; interval: PossessionForm };
// Kept in this tab's memory, scoped to API and run. Run navigation must not discard annotation work.
const reviewDrafts = new Map<string, ReviewDraft>();
const freshEvent = (): Annotation => ({ id: crypto.randomUUID(), timestamp_seconds: 0, type: "pass", team: "home", player: null, outcome: "unknown", review_status: "draft", notes: "" });
const count = (value: number | null) => value === null ? "Unknown" : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
const payload = (review: Review) => ({ revision: review.revision, teams: review.teams, coverage: review.coverage, events: review.events, possession_intervals: review.possession_intervals });
const hasUnsavedDraft = (draft: ReviewDraft) => JSON.stringify(payload(draft.review)) !== JSON.stringify(payload(draft.saved)) || draft.eventDirty || Boolean(draft.coverage.start || draft.coverage.end || draft.interval.start || draft.interval.end || draft.interval.approved || draft.interval.team !== "unknown");
// The warning belongs to the tab, not the currently mounted run: other runs may hold drafts.
const warnForReviewDrafts = (event: BeforeUnloadEvent) => {
  if (![...reviewDrafts.values()].some(hasUnsavedDraft)) return;
  event.preventDefault(); event.returnValue = "";
};
window.addEventListener("beforeunload", warnForReviewDrafts);
if (import.meta.hot) import.meta.hot.dispose(() => window.removeEventListener("beforeunload", warnForReviewDrafts));

export function ReviewPanel({ runId, duration, api, authHeaders, onSaved, onUnsaved }: Props) {
  const playback = usePlayback();
  const [review, setReview] = useState<Review | null>(null);
  const [saved, setSaved] = useState<Review | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState(false);
  const [reload, setReload] = useState(0);
  const [restored, setRestored] = useState(false);
  const [event, setEvent] = useState<Annotation>(freshEvent);
  const [editing, setEditing] = useState(false);
  const [eventDirty, setEventDirty] = useState(false);
  const [coverage, setCoverage] = useState({ start: "", end: "" });
  const [interval, setInterval] = useState({ start: "", end: "", team: "unknown" as Possession["team"], approved: false });
  const [eventSearch, setEventSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [eventPage, setEventPage] = useState(0);
  const epoch = useRef(0);
  const dirty = Boolean(review && saved && JSON.stringify(payload(review)) !== JSON.stringify(payload(saved)));
  const formDirty = eventDirty || Boolean(coverage.start || coverage.end || interval.start || interval.end || interval.approved || interval.team !== "unknown");
  const cacheKey = `${api}/runs/${runId}`;
  const eventMatches = review?.events.filter(item => (statusFilter === "all" || item.review_status === statusFilter) && `${item.type} ${item.outcome} ${item.player ?? ""} ${review.teams[item.team]} ${item.notes}`.toLowerCase().includes(eventSearch.trim().toLowerCase())) ?? [];
  const lastEventPage = Math.max(0, Math.ceil(eventMatches.length / 8) - 1);
  const currentEventPage = Math.min(eventPage, lastEventPage);
  useEffect(() => { onSaved?.(saved); }, [saved, onSaved]);
  useEffect(() => { onUnsaved?.(dirty || formDirty); }, [dirty, formDirty, onUnsaved]);

  useEffect(() => {
    if (review && saved) reviewDrafts.set(cacheKey, { review, saved, event, editing, eventDirty, coverage, interval });
  }, [cacheKey, review, saved, event, editing, eventDirty, coverage, interval]);

  useEffect(() => {
    const requestEpoch = ++epoch.current;
    const controller = new AbortController();
    setPending(true); setError(""); setConflict(false);
    setReview(current => current?.run_id === runId ? current : null);
    setSaved(current => current?.run_id === runId ? current : null);
    // A reload is explicitly chosen by the reviewer; successful fetch replaces the local edit state.
    fetch(`${api}/runs/${encodeURIComponent(runId)}/review`, { headers: authHeaders(), signal: controller.signal })
      .then(async response => { const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Could not load review (${response.status}).`); return body as Review; })
      .then(body => {
        if (epoch.current !== requestEpoch) return;
        const cached = reload === 0 ? reviewDrafts.get(cacheKey) : undefined;
        const cachedChanges = Boolean(cached && JSON.stringify(payload(cached.review)) !== JSON.stringify(payload(cached.saved)));
        const restore = cached && (cachedChanges || cached.eventDirty || cached.coverage.start || cached.coverage.end || cached.interval.start || cached.interval.end || cached.interval.approved || cached.interval.team !== "unknown");
        setSaved(body);
        if (restore && cached) {
          setReview(cachedChanges ? cached.review : body); setEvent(cached.event); setEditing(cached.editing); setEventDirty(cached.eventDirty); setCoverage(cached.coverage); setInterval(cached.interval); setRestored(true);
          if (cachedChanges && cached.review.revision !== body.revision) { setConflict(true); setError("Your per-run draft was restored, but the saved revision has changed. Unsaved edits are retained. Reload the saved review only to discard your draft."); }
        } else {
          setReview(body); setEvent(freshEvent()); setEditing(false); setEventDirty(false); setCoverage({ start: "", end: "" }); setInterval({ start: "", end: "", team: "unknown", approved: false }); setRestored(false);
        }
      })
      .catch(cause => { if (!controller.signal.aborted && epoch.current === requestEpoch) setError(String(cause)); })
      .finally(() => { if (epoch.current === requestEpoch) setPending(false); });
    return () => { controller.abort(); epoch.current++; };
  }, [runId, api, reload]);

  async function save() {
    if (!review || pending || eventDirty || conflict) return;
    const requestEpoch = epoch.current;
    setPending(true); setError("");
    try {
      const response = await fetch(`${api}/runs/${encodeURIComponent(runId)}/review`, { method: "POST", headers: { ...authHeaders(), "Content-Type": "application/json" }, body: JSON.stringify(payload(review)) });
      const body = await response.json().catch(() => ({}));
      if (epoch.current !== requestEpoch) return;
      if (response.status === 409) { setConflict(true); throw new Error("Another reviewer saved a newer revision. Your unsaved edits are still here. Reload the saved review only when you are ready to discard these edits."); }
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Save failed (${response.status}). Your edits are retained.`);
      setReview(body); setSaved(body); setRestored(false);
    } catch (cause) { if (epoch.current === requestEpoch) setError(String(cause)); }
    finally { if (epoch.current === requestEpoch) setPending(false); }
  }
  function edit(update: Partial<Review>) { setReview(current => current ? { ...current, ...update } : current); }
  function downloadSaved() {
    if (!saved) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(saved, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `${runId}-saved-review-revision-${saved.revision}.json`; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function resetEvent() { setEvent(freshEvent()); setEditing(false); setEventDirty(false); }
  function changeEvent(update: Partial<Annotation>) { setEvent(current => ({ ...current, ...update, review_status: update.review_status ?? "draft" })); setEventDirty(true); }
  function applyEvent(e: FormEvent) {
    e.preventDefault(); if (!review) return;
    if (!Number.isFinite(event.timestamp_seconds) || event.timestamp_seconds < 0 || event.timestamp_seconds >= duration) { setError("Event time must be inside the source video, before its end."); return; }
    const next = { ...event, player: event.player?.trim() || null, notes: event.notes.trim() };
    edit({ events: [...review.events.filter(item => item.id !== event.id), next].sort((a, b) => a.timestamp_seconds - b.timestamp_seconds) });
    setEventSearch(""); setStatusFilter("all"); setEventPage(0);
    resetEvent(); setError("");
  }
  function validateRange(start: string, end: string) {
    if (!start.trim() || !end.trim()) { setError("Enter both interval times in seconds."); return false; }
    const a = Number(start), b = Number(end);
    if (!Number.isFinite(a) || !Number.isFinite(b) || a < 0 || b <= a || b > duration) { setError(`Intervals must start at or after 0 and end within ${count(duration)} seconds, after their start.`); return false; }
    return true;
  }

  const status = pending ? "Working…" : conflict ? "Conflict · edits retained" : dirty || formDirty ? "Unsaved changes" : saved ? `Saved revision ${saved.revision}` : "Review not loaded";

  return (
    <Panel
      level={2}
      className="review"
      title="Review events"
      subtitle="Record what you can verify against the footage. Approval is always your choice."
      aside={<>
        <span className={`chip ${dirty || formDirty || conflict ? "chip--draft" : "chip--plain"}`} role="status">{status}</span>
        <button type="button" className="btn btn--primary btn--sm" onClick={save} disabled={pending || !dirty || eventDirty || conflict}><Save size={14} aria-hidden="true" /> Save review</button>
        <button type="button" className="btn btn--secondary btn--sm" onClick={downloadSaved} disabled={!saved || pending} aria-label="Download saved review JSON"><ArrowDownToLine size={14} aria-hidden="true" /> Saved review JSON</button>
      </>}
    >
      <div aria-busy={pending} className="stack">
        {error && <p className="notice notice--error" role="alert"><AlertTriangle size={16} aria-hidden="true" /><span>{error}</span></p>}
        {(conflict || (!review && !pending)) && <div><button type="button" className="btn btn--secondary" disabled={pending} onClick={() => setReload(value => value + 1)}>{conflict ? "Reload saved review · discard my edits" : "Retry loading review"}</button></div>}
        {eventDirty && <p className="note note--boxed note--flood">Apply or cancel the event form before saving the review.</p>}
        {restored && <p className="note note--boxed note--flood" role="status">Restored your unsaved work for this run. Nothing was automatically saved or approved.</p>}
        {!review && pending && <p className="notice" role="status"><span className="spinner" aria-hidden="true" /><span>Loading the saved review…</span></p>}

        {review && (
          <fieldset disabled={pending} className="review__work">
            <legend className="sr-only">Manual review · {count(duration)}-second source clip</legend>

            <form onSubmit={applyEvent} className="formcard">
              <h3 className="formcard__title">{editing ? "Edit event annotation" : "Add event annotation"}</h3>
              <div className="formgrid">
                <div className="formgrid__time">
                  <Field label="Video time (seconds)">
                    <input type="number" min="0" max={Math.max(0, duration - .000001)} step="any" required value={event.timestamp_seconds} onChange={e => changeEvent({ timestamp_seconds: Number(e.target.value) })} />
                  </Field>
                  <button type="button" className="btn btn--secondary" onClick={() => changeEvent({ timestamp_seconds: Math.max(0, Math.min(duration - .001, playback.getTime())) })}><Clock3 size={15} aria-hidden="true" /> Use video time</button>
                </div>
                <Field label="Event type">
                  <select value={event.type} onChange={e => changeEvent({ type: e.target.value as EventType, outcome: "unknown", review_status: "draft" })}>{Object.keys(outcomes).map(type => <option key={type} value={type}>{sentence(type)}</option>)}</select>
                </Field>
                <Field label="Team">
                  <select value={event.team} onChange={e => changeEvent({ team: e.target.value as Team })}>{(["home", "away"] as const).map(team => <option key={team} value={team}>{review.teams[team] || team}</option>)}</select>
                </Field>
                <Field label="Outcome">
                  <select value={event.outcome} onChange={e => changeEvent({ outcome: e.target.value })}>{outcomes[event.type].map(outcome => <option key={outcome} value={outcome}>{sentence(outcome)}</option>)}</select>
                </Field>
                <Field label="Player (optional)">
                  <input maxLength={100} value={event.player ?? ""} onChange={e => changeEvent({ player: e.target.value })} placeholder="Verified name or manual label" />
                </Field>
              </div>
              <p className="note">Team uses your own labels (“{review.teams.home || "Home"}” and “{review.teams.away || "Away"}”), which you can rename under Team labels below. They are separate from the model’s Kit A / Kit B groups and do not map onto them.</p>
              <Disclosure summary={`Notes (optional)${event.notes ? " · added" : ""}`}>
                <Field label="Annotation notes"><textarea maxLength={1000} rows={2} value={event.notes} onChange={e => changeEvent({ notes: e.target.value })} /></Field>
              </Disclosure>
              <label className="check"><input type="checkbox" checked={event.review_status === "approved"} onChange={e => changeEvent({ review_status: e.target.checked ? "approved" : "draft" })} />I reviewed this event against the footage and approve this annotation.</label>
              <div className="row">
                <button className="btn btn--primary" type="submit">{editing ? "Apply event edit" : "Add annotation"}</button>
                <button className="btn btn--quiet" type="button" onClick={resetEvent}>Cancel event form</button>
              </div>
            </form>

            <div className="stack">
              <h3 className="formcard__title">Annotations</h3>
              <div className="toolbar">
                <Field label="Find an annotation"><input type="search" value={eventSearch} placeholder="Event, player or notes" onChange={e => { setEventSearch(e.target.value); setEventPage(0); }} /></Field>
                <Field label="Review status" className="field--fixed">
                  <select value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setEventPage(0); }}><option value="all">All annotations</option><option value="draft">Draft</option><option value="approved">Approved</option></select>
                </Field>
              </div>
              {eventMatches.length > 0 && (
                <DataTable label="Manual event annotations; scroll to view all columns" stack>
                  <thead><tr><th scope="col">Video time</th><th scope="col">Event / outcome</th><th scope="col">Team / player</th><th scope="col">Status</th><th scope="col">Actions</th></tr></thead>
                  <tbody>
                    {eventMatches.slice(currentEventPage * 8, (currentEventPage + 1) * 8).map(item => (
                      <tr key={item.id}>
                        <td><WatchButton at={item.timestamp_seconds} describe="View annotation" /></td>
                        <td data-label="Event"><div>{sentence(item.type)} · {words(item.outcome)}{item.notes && <small>{item.notes}</small>}</div></td>
                        <td data-label="Team / player"><div>{review.teams[item.team]}<small>{item.player || "Player not identified"}</small></div></td>
                        <td data-label="Status">{item.review_status === "approved" ? <EvidenceChip level="reviewed" title="Approved by a reviewer" /> : <span className="chip chip--draft">Draft</span>}</td>
                        <td data-label="Actions">
                          <div className="row">
                            <button type="button" className="btn btn--secondary btn--sm" disabled={eventDirty} onClick={() => { setEvent({ ...item }); setEditing(true); }}>Edit</button>
                            <button type="button" className="btn btn--secondary btn--sm" onClick={() => edit({ events: review.events.map(other => other.id === item.id ? { ...other, review_status: other.review_status === "approved" ? "draft" : "approved" } : other) })}>{item.review_status === "approved" ? "Return to draft" : "Approve event"}</button>
                            <button type="button" className="btn btn--secondary btn--sm btn--danger" onClick={() => { edit({ events: review.events.filter(other => other.id !== item.id) }); if (event.id === item.id) resetEvent(); }}>Delete event</button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </DataTable>
              )}
              {review.events.length > 0 && <Pager page={currentEventPage} pages={lastEventPage + 1} count={eventMatches.length} noun="annotations" onPage={setEventPage} />}
              {review.events.length > 0 && !eventMatches.length && <p className="note" role="status">No annotations match these filters.</p>}
              {!review.events.length && <EmptyState title="No manual event annotations yet">Add a draft while watching the footage; approval is a separate review choice.</EmptyState>}
            </div>

            <Disclosure summary="Team labels" meta={`${review.teams.home || "Home"} / ${review.teams.away || "Away"}`}>
              <div className="formgrid formgrid--two">
                {(["home", "away"] as const).map(team => <Field key={team} label={`${team === "home" ? "Home" : "Away"} team name`}><input maxLength={80} required value={review.teams[team]} onChange={e => edit({ teams: { ...review.teams, [team]: e.target.value } })} /></Field>)}
              </div>
              <p>Enter real team names only when verified. These labels are independent of the model's anonymous kit groups.</p>
            </Disclosure>

            <Disclosure summary="Reviewed footage coverage" meta="Optional">
              <p>Record only intervals you have watched and reviewed. The full clip is not presumed reviewed. Overlapping intervals are handled by the server.</p>
              <form className="formgrid formgrid--action" onSubmit={e => { e.preventDefault(); if (!validateRange(coverage.start, coverage.end)) return; edit({ coverage: [...review.coverage, [Number(coverage.start), Number(coverage.end)]] }); setCoverage({ start: "", end: "" }); setError(""); }}>
                <Field label="Reviewed start (seconds)"><input type="number" min="0" max={duration} step="any" required value={coverage.start} onChange={e => setCoverage({ ...coverage, start: e.target.value })} /></Field>
                <Field label="Reviewed end (seconds)"><input type="number" min="0" max={duration} step="any" required value={coverage.end} onChange={e => setCoverage({ ...coverage, end: e.target.value })} /></Field>
                <button className="btn btn--secondary" type="submit">Add reviewed interval</button>
              </form>
              <ul className="itemlist">
                {review.coverage.map(([start, end], index) => (
                  <li key={index}><span className="mono">{count(start)}–{count(end)} s reviewed</span><button type="button" className="btn btn--secondary btn--sm" onClick={() => edit({ coverage: review.coverage.filter((_, i) => i !== index) })}>Remove coverage interval</button></li>
                ))}
              </ul>
              {!review.coverage.length && <p>No reviewed coverage recorded.</p>}
            </Disclosure>

            <Disclosure summary="Manually reviewed possession intervals" meta="Optional">
              <p>Annotate possession from footage, independently of jersey groups and ball proximity. Unknown intervals remain unknown. Draft intervals are excluded from statistics.</p>
              <form className="stack" onSubmit={e => { e.preventDefault(); if (!validateRange(interval.start, interval.end)) return; edit({ possession_intervals: [...review.possession_intervals, { id: crypto.randomUUID(), start_seconds: Number(interval.start), end_seconds: Number(interval.end), team: interval.team, review_status: interval.approved ? "approved" : "draft" }] }); setInterval({ start: "", end: "", team: "unknown", approved: false }); setError(""); }}>
                <div className="formgrid">
                  <Field label="Possession start (seconds)"><input type="number" min="0" max={duration} step="any" required value={interval.start} onChange={e => setInterval({ ...interval, start: e.target.value })} /></Field>
                  <Field label="Possession end (seconds)"><input type="number" min="0" max={duration} step="any" required value={interval.end} onChange={e => setInterval({ ...interval, end: e.target.value })} /></Field>
                  <Field label="Possession team"><select value={interval.team} onChange={e => setInterval({ ...interval, team: e.target.value as Possession["team"] })}><option value="unknown">Unknown</option><option value="home">{review.teams.home}</option><option value="away">{review.teams.away}</option></select></Field>
                </div>
                <label className="check"><input type="checkbox" checked={interval.approved} onChange={e => setInterval({ ...interval, approved: e.target.checked })} />I reviewed and approve this possession interval.</label>
                <div><button className="btn btn--secondary" type="submit">Add possession interval</button></div>
              </form>
              <ul className="itemlist">
                {review.possession_intervals.map(item => (
                  <li key={item.id}>
                    <span><span className="mono">{count(item.start_seconds)}–{count(item.end_seconds)} s</span> · {item.team === "unknown" ? "Unknown" : review.teams[item.team]} · {sentence(item.review_status)}</span>
                    <span className="row">
                      <WatchButton at={item.start_seconds} label="View in video" />
                      <button type="button" className="btn btn--secondary btn--sm" onClick={() => edit({ possession_intervals: review.possession_intervals.map(other => other.id === item.id ? { ...other, review_status: item.review_status === "approved" ? "draft" : "approved" } : other) })}>{item.review_status === "approved" ? "Return interval to draft" : "Approve interval"}</button>
                      <button type="button" className="btn btn--secondary btn--sm btn--danger" onClick={() => edit({ possession_intervals: review.possession_intervals.filter(other => other.id !== item.id) })}>Delete interval</button>
                    </span>
                  </li>
                ))}
              </ul>
              {!review.possession_intervals.length && <p>No possession intervals recorded.</p>}
            </Disclosure>
          </fieldset>
        )}

        <Disclosure summary="Where drafts are kept">
          <p>Drafts stay in this tab when you switch clips or views. Only Save review persists applied annotations.</p>
          <p>Per-run edits and unfinished forms are held in memory until this tab is closed or refreshed. Closing warns if any run has unsaved work. The JSON download contains the last server-saved review, excluding unsaved edits and unfinished forms.</p>
        </Disclosure>
      </div>
    </Panel>
  );
}

/** Totals from saved, reviewer-approved records only. Never mixed with model estimates. */
export function ApprovedSummary({ saved, unsaved }: { saved: Review; unsaved: boolean }) {
  const instanceId = useId();
  const s = saved.summary;
  const hasPossession = saved.possession_intervals.some(interval => interval.review_status === "approved");
  const rows: [string, (team: TeamSummary) => string][] = [
    ["Approved events", t => count(t.event_count)],
    ["Passes recorded", t => count(t.passes.recorded)],
    ["Pass outcomes · completed / known", t => `${count(t.passes.completed)} / ${count(t.passes.known_outcomes)}`],
    ["Pass completion · known outcomes only", t => t.passes.completion_pct === null ? "Unknown" : `${count(t.passes.completion_pct)}%`],
    ["Passes with unknown outcomes", t => count(t.passes.recorded - t.passes.known_outcomes)],
    ["Pass outcome coverage · known / recorded", t => t.passes.outcome_coverage_pct === null ? "Unknown" : `${count(t.passes.outcome_coverage_pct)}% (${count(t.passes.known_outcomes)} / ${count(t.passes.recorded)})`],
    ["Shots · on target / recorded", t => `${count(t.shots.on_target)} / ${count(t.shots.recorded)}`],
    ["Goals recorded", t => count(t.shots.goals)],
    ["Tackles · won / recorded", t => `${count(t.tackles.won)} / ${count(t.tackles.recorded)}`],
    ["Interceptions recorded", t => count(t.interceptions)],
    ["Dribbles · successful / recorded", t => `${count(t.dribbles.successful)} / ${count(t.dribbles.recorded)}`],
    ["Duels · won / recorded", t => `${count(t.duels.won)} / ${count(t.duels.recorded)}`],
  ];

  return (
    <Panel
      title="Reviewed-record summary"
      subtitle="Saved, reviewer-approved records only. This is not full-match statistics."
      aside={<EvidenceChip level="reviewed" />}
    >
      <p className="facts">
        <span>Approved events<b>{count(s.approved_events)}</b></span>
        <span>Saved drafts<b>{count(s.draft_events)}</b></span>
        <span>Reviewed footage<b>{secs(s.reviewed_seconds, 2)}</b></span>
        <span>Share of the clip reviewed<b>{s.reviewed_video_pct === null ? "Unknown" : `${num(s.reviewed_video_pct, 2)}%`}</b></span>
      </p>
      <p className="note">Coverage is recorded separately from individual event approval.</p>
      {unsaved && <p className="note note--boxed note--flood">Unsaved edits and unfinished forms are excluded. Save the review to update this summary.</p>}
      {!s.approved_events ? (
        <EmptyState title="No approved event records yet">Football event statistics remain unavailable until a reviewer approves and saves annotations. Passes, shots and possession are not inferred from detection counts.</EmptyState>
      ) : <>
        <DataTable label="Saved approved team event summary; scroll for all columns">
          <thead><tr><th scope="col">Recorded metric</th><th scope="col" className="num">{saved.teams.home}</th><th scope="col" className="num">{saved.teams.away}</th></tr></thead>
          <tbody>
            {rows.map(([label, format]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                {(["home", "away"] as const).map(team => <td key={team} className="num">{s.teams[team] ? format(s.teams[team]!) : "No approved records"}</td>)}
              </tr>
            ))}
          </tbody>
        </DataTable>
        {s.players.length > 0 && (
          <div className="stack stack--sm">
            <h4 id={instanceId}>Manually labelled player records</h4>
            <DataTable label="Approved annotations by manually labelled player">
              <thead><tr><th scope="col">Player / team</th><th scope="col" className="num">Approved events</th><th scope="col">Recorded event types</th></tr></thead>
              <tbody>
                {s.players.map(player => (
                  <tr key={`${player.team}-${player.player}`}>
                    <th scope="row">{player.player}<small>{saved.teams[player.team]}</small></th>
                    <td className="num">{player.event_count}</td>
                    <td>{Object.entries(player.counts).map(([type, total]) => `${words(type)}: ${total}`).join(" · ")}</td>
                  </tr>
                ))}
              </tbody>
            </DataTable>
          </div>
        )}
      </>}
      {hasPossession ? (
        <div className="stack stack--sm">
          <h4>Approved possession annotations</h4>
          <p>{saved.teams.home}: {secs(s.possession.home_seconds, 2)} · {saved.teams.away}: {secs(s.possession.away_seconds, 2)} · Unknown: {secs(s.possession.unknown_seconds, 2)}.</p>
          <p className="note">
            Share of known approved possession: {saved.teams.home} {s.possession.home_share_pct === null ? "unknown" : `${count(s.possession.home_share_pct)}%`}, {saved.teams.away} {s.possession.away_share_pct === null ? "unknown" : `${count(s.possession.away_share_pct)}%`}.
            Known possession coverage: {s.possession.known_coverage_pct === null ? "unknown" : `${count(s.possession.known_coverage_pct)}%`} of source. Unannotated footage is not assigned a team.
          </p>
        </div>
      ) : <p className="note">No saved approved possession intervals. Reviewed possession statistics remain unavailable.</p>}
    </Panel>
  );
}
