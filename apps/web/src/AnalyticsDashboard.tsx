import { useId, useState } from "react";
import { analyticsAreas, requiredFilters } from "./analytics-requirements";
import { DataTable, Disclosure, EmptyState, Field } from "./ui";

type Props = { filename?: string; status?: "queued" | "running" | "complete" | "failed"; hasDetections: boolean; onReview?: () => void };

/**
 * The requirements catalogue: every analytics area the product is meant to cover and what data each
 * metric still needs. Nothing here is a calculated value.
 */
export function AnalyticsDashboard({ filename, status, hasDetections, onReview }: Props) {
  const instanceId = useId();
  const [activeId, setActiveId] = useState(analyticsAreas[0].id);
  const [query, setQuery] = useState("");
  const active = analyticsAreas.find(area => area.id === activeId)!;
  const matches = active.metrics.filter(metric => metric.label.toLowerCase().includes(query.trim().toLowerCase()));
  const state = status === "failed" ? "Video analysis failed. See the error above and retry when available."
    : status === "queued" || status === "running" ? "Video processing is in progress. Football analytics are not available yet."
    : hasDetections ? "Detections ready. Review annotations to build scoped football records."
    : "Awaiting footage and verified match data.";

  return (
    <section className="stack" aria-labelledby={instanceId + "-title"}>
      <div className="row row--between">
        <div>
          <h4 id={instanceId + "-title"}>Football analytics still to come</h4>
          <p className="note" role="status">{state}{filename ? ` Selected footage: ${filename}.` : ""}</p>
        </div>
        {hasDetections && onReview && <button type="button" className="btn btn--secondary btn--sm" onClick={onReview}>Review video & detections</button>}
      </div>
      <p className="note"><strong>Match-half target: 45 minutes.</strong> Current hosted test limit: 2 minutes / 250 MiB per clip.</p>

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
          <button key={area.id} type="button" aria-pressed={activeId === area.id} onClick={() => { setActiveId(area.id); setQuery(""); }}>
            {area.title} <span className="mono">{area.metrics.length}</span>
          </button>
        ))}
      </nav>

      <div className="row row--between">
        <div className="grow">
          <h4>{active.title}</h4>
          <p className="note">{active.description}</p>
        </div>
        <div className="row row--end">
          <Field label="Find a metric in this module">
            <input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search metric names" />
          </Field>
          {query && <button type="button" className="btn btn--quiet btn--sm" onClick={() => setQuery("")}>Clear</button>}
        </div>
      </div>

      {matches.length > 0 ? (
        <DataTable label={`${active.title}: required metrics and data needed`} stack>
          <thead><tr><th scope="col">Required metric</th><th scope="col">Value / status</th><th scope="col">Data needed</th></tr></thead>
          <tbody>
            {matches.map(metric => (
              <tr key={metric.id}>
                <th scope="row">{metric.label}</th>
                <td data-label="Value / status"><div><span className="mono faint" aria-label="No calculated value">—</span> <span className="muted">Not available yet</span></div></td>
                <td data-label="Data needed">{metric.needs}</td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      ) : (
        <EmptyState title={`No matching metrics in ${active.title}`} action={<button type="button" className="btn btn--secondary btn--sm" onClick={() => setQuery("")}>Clear search</button>}>
          Try another metric name or select a different module.
        </EmptyState>
      )}

      <p className="note">
        This catalogue describes the full analytics requirements. Saved approved annotations have a separate reviewed-record summary under Review. Wider analytics still need verified identities and the relevant source data. Detection counts do not establish match statistics.
        {active.id === "workload" && " Workload and injury indicators support qualified staff review; they are not diagnoses."}
      </p>
      <p className="note faint">Requirements: {active.source}</p>
    </section>
  );
}
