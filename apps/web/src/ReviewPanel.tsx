import { useEffect, useId, useRef, useState, type FormEvent } from "react";

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
type Props = { runId: string; duration: number; api: string; authHeaders: () => Record<string, string>; onSaved?: (saved: Review | null) => void; onUnsaved?: (unsaved: boolean) => void; onSeek?: (seconds: number) => void; getVideoTime?: () => number; hideSummary?: boolean };
type CoverageForm = { start: string; end: string };
type PossessionForm = CoverageForm & { team: Possession["team"]; approved: boolean };
type ReviewDraft = { review: Review; saved: Review; event: Annotation; editing: boolean; eventDirty: boolean; coverage: CoverageForm; interval: PossessionForm };
// Kept in this tab's memory, scoped to API and run. Run navigation must not discard annotation work.
const reviewDrafts = new Map<string, ReviewDraft>();
const freshEvent = (): Annotation => ({ id: crypto.randomUUID(), timestamp_seconds: 0, type: "pass", team: "home", player: null, outcome: "unknown", review_status: "draft", notes: "" });
const words = (text: string) => text.replaceAll("_", " ");
const num = (value: number | null) => value === null ? "Unknown" : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
const payload = (review: Review) => ({ revision: review.revision, teams: review.teams, coverage: review.coverage, events: review.events, possession_intervals: review.possession_intervals });
const hasUnsavedDraft = (draft: ReviewDraft) => JSON.stringify(payload(draft.review)) !== JSON.stringify(payload(draft.saved)) || draft.eventDirty || Boolean(draft.coverage.start || draft.coverage.end || draft.interval.start || draft.interval.end || draft.interval.approved || draft.interval.team !== "unknown");
// The warning belongs to the tab, not the currently mounted run: other runs may hold drafts.
const warnForReviewDrafts = (event: BeforeUnloadEvent) => {
  if (![...reviewDrafts.values()].some(hasUnsavedDraft)) return;
  event.preventDefault(); event.returnValue = "";
};
window.addEventListener("beforeunload", warnForReviewDrafts);
if (import.meta.hot) import.meta.hot.dispose(() => window.removeEventListener("beforeunload", warnForReviewDrafts));

export function ReviewPanel({ runId, duration, api, authHeaders, onSaved, onUnsaved, onSeek, getVideoTime, hideSummary }: Props) {
  const instanceId = useId();
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
  function seek(seconds: number) { if (onSeek) { onSeek(seconds); return; } const video = document.querySelector<HTMLVideoElement>(".real-analysis video"); if (video) { video.currentTime = seconds; video.scrollIntoView({ behavior: "smooth", block: "center" }); } }
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
    if (!Number.isFinite(a) || !Number.isFinite(b) || a < 0 || b <= a || b > duration) { setError(`Intervals must start at or after 0 and end within ${num(duration)} seconds, after their start.`); return false; }
    return true;
  }

  return <section className="review-panel panel" aria-labelledby={instanceId + "-event-review-title"} aria-busy={pending}>
    <header className="review-panel__header"><div><p className="eyebrow">Human event review</p><h2 id={instanceId + "-event-review-title"}>Review events</h2><p>Record what you can verify. Approval is always your choice.</p></div>
      <div className="review-panel__save"><span role="status">{pending ? "Working…" : conflict ? "Conflict · edits retained" : dirty || formDirty ? "Unsaved changes" : saved ? `Saved revision ${saved.revision}` : "Review not loaded"}</span><button className="ghost" onClick={save} disabled={pending || !dirty || eventDirty || conflict}>Save review</button><button className="ghost" onClick={downloadSaved} disabled={!saved || pending}>Download saved review JSON</button></div>
    </header>
    {error && <p className="review-panel__error" role="alert">{error}</p>}
    {(conflict || (!review && !pending)) && <button className="ghost" disabled={pending} onClick={() => setReload(value => value + 1)}>{conflict ? "Reload saved review · discard my edits" : "Retry loading review"}</button>}
    {eventDirty && <p className="review-panel__notice">Apply or cancel the event form before saving the review.</p>}
    {restored && <p className="review-panel__notice" role="status">Restored your unsaved work for this run. Nothing was automatically saved or approved.</p>}
    <p className="review-panel__draft-note">Drafts stay in this tab when you switch clips or views. Only Save review persists applied annotations.</p><details className="review-panel__storage"><summary>Draft storage & saved exports</summary><p>Per-run edits and unfinished forms are held in memory until this tab is closed or refreshed. Closing warns if any run has unsaved work. The JSON download contains the last server-saved review, excluding unsaved edits and unfinished forms.</p></details>
    {review && <>
      <fieldset disabled={pending} className="review-panel__workspace"><legend>Manual review · {num(duration)}-second source clip</legend>
        <details className="review-panel__intervals"><summary>Team labels · {review.teams.home || "Home"} / {review.teams.away || "Away"}</summary><div className="review-panel__team-names">{(["home", "away"] as const).map(team => <label key={team}>{team === "home" ? "Home" : "Away"} team name<input maxLength={80} required value={review.teams[team]} onChange={e => edit({ teams: { ...review.teams, [team]: e.target.value } })} /></label>)}</div><p>Enter real team names only when verified. These labels are independent of anonymous model kit groups.</p></details>
        <form onSubmit={applyEvent} className="review-panel__event-form"><h3>{editing ? "Edit event annotation" : "Add event annotation"}</h3>
          <div className="review-panel__fields"><label>Video time (seconds)<input type="number" min="0" max={Math.max(0, duration - .000001)} step="any" required value={event.timestamp_seconds} onChange={e => changeEvent({ timestamp_seconds: Number(e.target.value) })} /></label>
            <button type="button" className="ghost" onClick={() => { const time = getVideoTime ? getVideoTime() : document.querySelector<HTMLVideoElement>(".real-analysis video")?.currentTime; if (time !== undefined) changeEvent({ timestamp_seconds: Math.max(0, Math.min(duration - .001, time)) }); }}>Use video time</button>
            <label>Event type<select value={event.type} onChange={e => changeEvent({ type: e.target.value as EventType, outcome: "unknown", review_status: "draft" })}>{Object.keys(outcomes).map(type => <option key={type} value={type}>{words(type)}</option>)}</select></label>
            <label>Team<select value={event.team} onChange={e => changeEvent({ team: e.target.value as Team })}>{(["home", "away"] as const).map(team => <option key={team} value={team}>{review.teams[team] || team}</option>)}</select></label>
            <label>Outcome<select value={event.outcome} onChange={e => changeEvent({ outcome: e.target.value })}>{outcomes[event.type].map(outcome => <option key={outcome} value={outcome}>{words(outcome)}</option>)}</select></label>
            <label>Player (optional)<input maxLength={100} value={event.player ?? ""} onChange={e => changeEvent({ player: e.target.value })} placeholder="Verified name or manual label" /></label>
          </div>
          <details className="review-panel__notes"><summary>Notes (optional){event.notes && " · added"}</summary><label>Annotation notes<textarea maxLength={1000} rows={2} value={event.notes} onChange={e => changeEvent({ notes: e.target.value })} /></label></details>
          <label className="review-panel__checkbox"><input type="checkbox" checked={event.review_status === "approved"} onChange={e => changeEvent({ review_status: e.target.checked ? "approved" : "draft" })} />I reviewed this event against the footage and approve this annotation.</label>
          <div className="review-panel__actions"><button className="ghost" type="submit">{editing ? "Apply event edit" : "Add annotation"}</button><button className="ghost" type="button" onClick={resetEvent}>Cancel event form</button></div>
        </form>

        <div className="review-panel__event-tools"><label>Find an annotation<input type="search" value={eventSearch} placeholder="Event, player or notes" onChange={e => { setEventSearch(e.target.value); setEventPage(0); }} /></label><label>Review status<select value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setEventPage(0); }}><option value="all">All annotations</option><option value="draft">Draft</option><option value="approved">Approved</option></select></label></div>
{eventMatches.length > 0 &&         <div className="review-panel__scroll" tabIndex={0} role="region" aria-label="Manual event annotations; scroll to view all columns"><table><thead><tr><th scope="col">Video time</th><th scope="col">Event / outcome</th><th scope="col">Team / player</th><th scope="col">Status</th><th scope="col">Actions</th></tr></thead><tbody>{eventMatches.slice(currentEventPage * 8, (currentEventPage + 1) * 8).map(item => <tr key={item.id}><td><button type="button" className="review-panel__seek" onClick={() => seek(item.timestamp_seconds)}>{num(item.timestamp_seconds)}s · view</button></td><td>{words(item.type)} · {words(item.outcome)}{item.notes && <small>{item.notes}</small>}</td><td>{review.teams[item.team]}<small>{item.player || "Player not identified"}</small></td><td><span className={`studio-status ${item.review_status}`}>{item.review_status === "approved" ? "Approved" : "Draft"}</span></td><td><div className="review-panel__actions"><button type="button" className="ghost" disabled={eventDirty} onClick={() => { setEvent({ ...item }); setEditing(true); }}>Edit</button><button type="button" className="ghost" onClick={() => edit({ events: review.events.map(other => other.id === item.id ? { ...other, review_status: other.review_status === "approved" ? "draft" : "approved" } : other) })}>{item.review_status === "approved" ? "Return to draft" : "Approve event"}</button><button type="button" className="ghost" onClick={() => { edit({ events: review.events.filter(other => other.id !== item.id) }); if (event.id === item.id) resetEvent(); }}>Delete event</button></div></td></tr>)}</tbody></table></div>}
        {review.events.length > 0 && <div className="studio-pagination"><span>{eventMatches.length} matching annotations · page {currentEventPage + 1} of {lastEventPage + 1}</span><button type="button" className="ghost" disabled={currentEventPage === 0} onClick={() => setEventPage(currentEventPage - 1)}>Previous annotations</button><button type="button" className="ghost" disabled={currentEventPage >= lastEventPage} onClick={() => setEventPage(currentEventPage + 1)}>Next annotations</button></div>}
        {review.events.length > 0 && !eventMatches.length && <p role="status">No annotations match these filters.</p>}
        {!review.events.length && <p className="review-panel__notice">No manual event annotations. Add a draft while watching the footage; approval is a separate review choice.</p>}

        <details className="review-panel__intervals"><summary>Reviewed footage coverage · optional</summary><p>Record only intervals you have watched and reviewed. The full clip is not presumed reviewed. Overlapping intervals are handled by the server.</p>
          <form className="review-panel__fields" onSubmit={e => { e.preventDefault(); if (!validateRange(coverage.start, coverage.end)) return; edit({ coverage: [...review.coverage, [Number(coverage.start), Number(coverage.end)]] }); setCoverage({ start: "", end: "" }); setError(""); }}><label>Reviewed start (seconds)<input type="number" min="0" max={duration} step="any" required value={coverage.start} onChange={e => setCoverage({ ...coverage, start: e.target.value })} /></label><label>Reviewed end (seconds)<input type="number" min="0" max={duration} step="any" required value={coverage.end} onChange={e => setCoverage({ ...coverage, end: e.target.value })} /></label><button className="ghost" type="submit">Add reviewed interval</button></form>
          <ul>{review.coverage.map(([start, end], index) => <li key={index}><span>{num(start)}–{num(end)}s reviewed</span><button type="button" className="ghost" onClick={() => edit({ coverage: review.coverage.filter((_, i) => i !== index) })}>Remove coverage interval</button></li>)}</ul>{!review.coverage.length && <p>No reviewed coverage recorded.</p>}
        </details>
        <details className="review-panel__intervals"><summary>Manually reviewed possession intervals · optional</summary><p>Annotate possession from footage, independently of jersey groups and ball proximity. Unknown intervals remain unknown. Draft intervals are excluded from statistics.</p>
          <form onSubmit={e => { e.preventDefault(); if (!validateRange(interval.start, interval.end)) return; edit({ possession_intervals: [...review.possession_intervals, { id: crypto.randomUUID(), start_seconds: Number(interval.start), end_seconds: Number(interval.end), team: interval.team, review_status: interval.approved ? "approved" : "draft" }] }); setInterval({ start: "", end: "", team: "unknown", approved: false }); setError(""); }}>
            <div className="review-panel__fields"><label>Possession start (seconds)<input type="number" min="0" max={duration} step="any" required value={interval.start} onChange={e => setInterval({ ...interval, start: e.target.value })} /></label><label>Possession end (seconds)<input type="number" min="0" max={duration} step="any" required value={interval.end} onChange={e => setInterval({ ...interval, end: e.target.value })} /></label><label>Possession team<select value={interval.team} onChange={e => setInterval({ ...interval, team: e.target.value as Possession["team"] })}><option value="unknown">Unknown</option><option value="home">{review.teams.home}</option><option value="away">{review.teams.away}</option></select></label></div>
            <label className="review-panel__checkbox"><input type="checkbox" checked={interval.approved} onChange={e => setInterval({ ...interval, approved: e.target.checked })} />I reviewed and approve this possession interval.</label><button className="ghost" type="submit">Add possession interval</button>
          </form>
          <ul>{review.possession_intervals.map(item => <li key={item.id}><span>{num(item.start_seconds)}–{num(item.end_seconds)}s · {item.team === "unknown" ? "Unknown" : review.teams[item.team]} · {item.review_status}</span><div className="review-panel__actions"><button type="button" className="ghost" onClick={() => seek(item.start_seconds)}>View in video</button><button type="button" className="ghost" onClick={() => edit({ possession_intervals: review.possession_intervals.map(other => other.id === item.id ? { ...other, review_status: item.review_status === "approved" ? "draft" : "approved" } : other) })}>{item.review_status === "approved" ? "Return interval to draft" : "Approve interval"}</button><button type="button" className="ghost" onClick={() => edit({ possession_intervals: review.possession_intervals.filter(other => other.id !== item.id) })}>Delete interval</button></div></li>)}</ul>{!review.possession_intervals.length && <p>No possession intervals recorded.</p>}
        </details>
      </fieldset>
      {saved && !hideSummary && <ApprovedSummary saved={saved} unsaved={dirty || formDirty} />}
    </>}
  </section>;
}

export function ApprovedSummary({ saved, unsaved }: { saved: Review; unsaved: boolean }) {
  const instanceId = useId();
  const s = saved.summary;
  const hasPossession = saved.possession_intervals.some(interval => interval.review_status === "approved");
  return <section className="review-panel__summary" aria-labelledby={instanceId + "-approved-summary-title"}><p className="eyebrow">Saved reviewer-approved records only</p><h3 id={instanceId + "-approved-summary-title"}>Reviewed-record summary</h3>
    <p>This is a summary of saved approved annotations, not full-match statistics. Reviewed coverage: {num(s.reviewed_seconds)}s ({num(s.reviewed_video_pct)}% of the source). Coverage is recorded separately from individual event approval.</p>
    {unsaved && <p className="review-panel__notice">Unsaved edits are excluded. Save the review to update this summary.</p>}
    {!s.approved_events ? <p className="review-panel__notice">No saved approved events. Football event statistics remain unavailable until a reviewer approves and saves annotations.</p> : <>
      <div className="review-panel__scroll" tabIndex={0} role="region" aria-label="Saved approved team event summary; scroll for all columns"><table><thead><tr><th scope="col">Recorded metric</th><th scope="col">{saved.teams.home}</th><th scope="col">{saved.teams.away}</th></tr></thead><tbody>
        {[
          ["Approved events", (t: TeamSummary) => num(t.event_count)],
          ["Passes recorded", (t: TeamSummary) => num(t.passes.recorded)],
          ["Pass outcomes · completed / known", (t: TeamSummary) => `${num(t.passes.completed)} / ${num(t.passes.known_outcomes)}`],
          ["Pass completion · known outcomes only", (t: TeamSummary) => t.passes.completion_pct === null ? "Unknown" : `${num(t.passes.completion_pct)}%`],
          ["Passes with unknown outcomes", (t: TeamSummary) => num(t.passes.recorded - t.passes.known_outcomes)],
          ["Pass outcome coverage · known / recorded", (t: TeamSummary) => t.passes.outcome_coverage_pct === null ? "Unknown" : `${num(t.passes.outcome_coverage_pct)}% (${num(t.passes.known_outcomes)} / ${num(t.passes.recorded)})`],
          ["Shots · on target / recorded", (t: TeamSummary) => `${num(t.shots.on_target)} / ${num(t.shots.recorded)}`],
          ["Goals recorded", (t: TeamSummary) => num(t.shots.goals)],
          ["Tackles · won / recorded", (t: TeamSummary) => `${num(t.tackles.won)} / ${num(t.tackles.recorded)}`],
          ["Interceptions recorded", (t: TeamSummary) => num(t.interceptions)],
          ["Dribbles · successful / recorded", (t: TeamSummary) => `${num(t.dribbles.successful)} / ${num(t.dribbles.recorded)}`],
          ["Duels · won / recorded", (t: TeamSummary) => `${num(t.duels.won)} / ${num(t.duels.recorded)}`],
        ].map(([label, format]) => <tr key={label as string}><th scope="row">{label as string}</th>{(["home", "away"] as const).map(team => <td key={team}>{s.teams[team] ? (format as (t: TeamSummary) => string)(s.teams[team]!) : "No approved records"}</td>)}</tr>)}
      </tbody></table></div>
      {s.players.length > 0 && <><h4>Manually labelled player records</h4><div className="review-panel__scroll" tabIndex={0} role="region" aria-label="Approved annotations by manually labelled player"><table><thead><tr><th scope="col">Player / team</th><th scope="col">Approved events</th><th scope="col">Recorded event types</th></tr></thead><tbody>{s.players.map(player => <tr key={`${player.team}-${player.player}`}><th scope="row">{player.player}<small>{saved.teams[player.team]}</small></th><td>{player.event_count}</td><td>{Object.entries(player.counts).map(([type, count]) => `${words(type)}: ${count}`).join(" · ")}</td></tr>)}</tbody></table></div></>}
    </>}
    {hasPossession ? <div className="review-panel__possession-summary"><h4>Approved possession annotations</h4><p>{saved.teams.home}: {num(s.possession.home_seconds)}s · {saved.teams.away}: {num(s.possession.away_seconds)}s · Unknown / unassigned: {num(s.possession.unknown_seconds)}s.</p><p>Share of known approved possession: {saved.teams.home} {s.possession.home_share_pct === null ? "unknown" : `${num(s.possession.home_share_pct)}%`}, {saved.teams.away} {s.possession.away_share_pct === null ? "unknown" : `${num(s.possession.away_share_pct)}%`}. Known possession coverage: {num(s.possession.known_coverage_pct)}% of source. Unannotated footage is not assigned a team.</p></div> : <p>No saved approved possession intervals. Possession statistics remain unavailable.</p>}
  </section>;
}
