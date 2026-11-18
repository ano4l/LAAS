import { useId, useMemo, useState } from "react";
import { analyticsAreas, requiredFilters } from "./analytics-requirements";
import { approvedMetric, type MetricState, type MetricValue, type Side } from "./approved-analytics";
import { num, secs } from "./format";
import type { Review } from "./ReviewPanel";
import { DataTable, Disclosure, EmptyState, Field } from "./ui";

type Props = { saved: Review | null; unsaved?: boolean; onReview?: () => void };

const STATE: Record<MetricState, { label: string; chip: string; meaning: string }> = {
  calculated: { label: "Calculated", chip: "chip--reviewed", meaning: "Calculated from saved, approved records for this clip." },
  partial: { label: "Partly calculated", chip: "chip--draft", meaning: "Part of this is calculated from approved records. The note says what is missing." },
  unavailable: { label: "Not available yet", chip: "chip--unknown", meaning: "Nothing is shown because the data it needs does not exist yet." },
};

/**
 * Every analytic the product was asked for. An entry shows a figure only when saved, approved
 * records can supply it; every other entry says what it still needs. Nothing is estimated here.
 */
export function AnalyticsDashboard({ saved, unsaved, onReview }: Props) {
  const instanceId = useId();
  const [activeId, setActiveId] = useState(analyticsAreas[0].id);
  const [query, setQuery] = useState("");
  const [ownSide, setOwnSide] = useState<Side>("home");
  const active = analyticsAreas.find(area => area.id === activeId)!;

  const values = useMemo(() => {
    const map = new Map<string, MetricValue>();
    analyticsAreas.forEach(area => area.metrics.forEach(metric => {
      // An entry approved events can supply says so itself when none are saved. The rest keep the data they need.
      map.set(metric.id, (saved ? approvedMetric(metric.id, saved, ownSide) : null) ?? { state: "unavailable", lines: [], note: metric.needs });
    }));
    return map;
  }, [saved, ownSide]);
  const shown = (areaId: string) => analyticsAreas.find(area => area.id === areaId)!.metrics.filter(metric => values.get(metric.id)!.state !== "unavailable").length;
  const total = analyticsAreas.reduce((sum, area) => sum + area.metrics.length, 0);
  const totalShown = analyticsAreas.reduce((sum, area) => sum + shown(area.id), 0);
  const matches = active.metrics.filter(metric => metric.label.toLowerCase().includes(query.trim().toLowerCase()));
  const s = saved?.summary;

  return (
    <section className="stack" aria-labelledby={instanceId + "-title"}>
      <div className="row row--between">
        <div>
          <h4 id={instanceId + "-title"}>{totalShown} of {total} requested analytics have figures for this clip</h4>
          <p className="note" role="status">
            {!saved ? "Loading saved review records…"
              : !s!.approved_events ? "No approved events are saved for this clip, so no event figures can be shown. Nothing is filled in from detections."
              : `From ${num(s!.approved_events, 0)} saved, approved ${s!.approved_events === 1 ? "event" : "events"}${s!.draft_events ? ` (${num(s!.draft_events, 0)} saved ${s!.draft_events === 1 ? "draft" : "drafts"} left out)` : ""}. ${s!.reviewed_video_pct ? `${num(s!.reviewed_video_pct)}% of the clip (${secs(s!.reviewed_seconds)}) is marked as reviewed.` : "No part of the clip is marked as reviewed, so events may be missing anywhere."}`}
          </p>
        </div>
        {onReview && <button type="button" className="btn btn--secondary btn--sm" onClick={onReview}>Record and approve events</button>}
      </div>
      <p className="note note--boxed">
        <strong>These are totals for this clip, not for a match. </strong>
        They count what a person recorded and approved, so anything not recorded is missing from them. Figures that need a season of matches, a pitch location on each event, GPS files or other teams’ data stay unavailable and say why.
      </p>
      {unsaved && <p className="note note--boxed note--flood">Unsaved edits and unfinished forms under Review are not counted. Save the review to update these figures.</p>}

      <Disclosure summary="Match & historical data filters" meta="Awaiting verified data">
        <p id={instanceId + "-filter-help"}>These filters need verified match and historical datasets. No competition, team or player data is connected yet.</p>
        <fieldset disabled aria-describedby={instanceId + "-filter-help"} className="review__work">
          <legend className="sr-only">Dataset filters · not available yet</legend>
          <div className="formgrid">
            {requiredFilters.map(filter => <Field key={filter} label={filter}><select defaultValue=""><option value="">Not available yet</option></select></Field>)}
          </div>
        </fieldset>
      </Disclosure>

      <nav className="pillnav" aria-label="Football analytics modules">
        {analyticsAreas.map(area => (
          <button key={area.id} type="button" aria-pressed={activeId === area.id} aria-label={`${area.title}: ${shown(area.id)} of ${area.metrics.length} with figures`} onClick={() => { setActiveId(area.id); setQuery(""); }}>
            {area.title} <span className="mono">{shown(area.id)}/{area.metrics.length}</span>
          </button>
        ))}
      </nav>

      <div className="row row--between">
        <div className="grow">
          <h4>{active.title}</h4>
          <p className="note">{active.description}</p>
        </div>
        <div className="row row--end">
          {active.id === "opponents" && saved && (
            <Field label="Your team">
              <select value={ownSide} onChange={event => setOwnSide(event.target.value as Side)}>
                <option value="home">{saved.teams.home}</option>
                <option value="away">{saved.teams.away}</option>
              </select>
            </Field>
          )}
          <Field label="Find a metric in this module">
            <input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search metric names" />
          </Field>
          {query && <button type="button" className="btn btn--quiet btn--sm" onClick={() => setQuery("")}>Clear</button>}
        </div>
      </div>

      {matches.length > 0 ? (
        <DataTable label={`${active.title}: requested metrics, their figures and what they need`} stack>
          <thead><tr><th scope="col">Requested metric</th><th scope="col">Status</th><th scope="col">Figures from approved records</th><th scope="col">What it still needs</th></tr></thead>
          <tbody>
            {matches.map(metric => {
              const value = values.get(metric.id)!, state = STATE[value.state];
              return (
                <tr key={metric.id}>
                  <th scope="row">{metric.label}</th>
                  <td data-label="Status"><span className={`chip ${state.chip}`} title={state.meaning}>{value.state === "unavailable" && <i className="chip__swatch" aria-hidden="true" />}{state.label}</span></td>
                  <td data-label="Figures from approved records" className="cell--prose">
                    {value.lines.length
                      ? <div className="stack stack--sm">{value.lines.map(line => <span key={line}>{line}</span>)}</div>
                      : <span className="mono faint" aria-label="No calculated value">—</span>}
                  </td>
                  <td data-label="What it still needs" className="cell--prose">{value.note ?? (value.state === "calculated" ? "Nothing more at clip level. A match total needs the whole match reviewed." : metric.needs)}</td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      ) : (
        <EmptyState title={`No matching metrics in ${active.title}`} action={<button type="button" className="btn btn--secondary btn--sm" onClick={() => setQuery("")}>Clear search</button>}>
          Try another metric name or select a different module.
        </EmptyState>
      )}

      {s?.definitions && (
        <Disclosure summary="How these figures are counted">
          {Object.entries(s.definitions).map(([key, text]) => <p key={key}>{text}</p>)}
        </Disclosure>
      )}
      <p className="note">
        Team names are the labels typed under Review. They are separate from the model’s kit groups, and nothing here comes from detections or possible events that have not been approved.
        {active.id === "opponents" && " “Your team” only changes which side is treated as the opponent."}
        {active.id === "workload" && " Workload and injury indicators support qualified staff review; they are not diagnoses."}
      </p>
      <p className="note faint">Requirements: {active.source}</p>
    </section>
  );
}
