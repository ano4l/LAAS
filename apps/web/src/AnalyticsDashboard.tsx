import { useId, useState } from "react";
import { analyticsAreas, requiredFilters } from "./analytics-requirements";
import { VideoMeasurementsPanel, type VideoMeasurements } from "./VideoMeasurements";

type Props = { filename?: string; status?: "queued" | "running" | "complete" | "failed"; hasDetections: boolean; measurements?: VideoMeasurements; onReview?: () => void };

export function AnalyticsDashboard({ filename, status, hasDetections, measurements, onReview }: Props) {
  const instanceId = useId();
  const [activeId, setActiveId] = useState(analyticsAreas[0].id);
  const [query, setQuery] = useState("");
  const active = analyticsAreas.find(area => area.id === activeId)!;
  const matches = active.metrics.filter(metric => metric.label.toLowerCase().includes(query.trim().toLowerCase()));
  const state = status === "failed" ? "Video analysis failed. See the error above and retry when available."
    : status === "queued" || status === "running" ? "Video processing is in progress. Football analytics are not available yet."
    : hasDetections ? "Detections ready. Review annotations below to build scoped football records."
    : "Awaiting footage and verified match data.";

  return <section className="analytics-dashboard" id={instanceId + "-football-analytics"} aria-labelledby={instanceId + "-football-analytics-title"}>
    <header className="analytics-dashboard__header">
      <div><p className="eyebrow">Analyst workspace</p><h2 id={instanceId + "-football-analytics-title"}>Football analytics</h2>
        <p className="analytics-dashboard__state" role="status">{state}</p>
        {filename && <p className="analytics-dashboard__filename">Selected footage: {filename}</p>}
      </div>
      {hasDetections && <a className="analytics-dashboard__review" href="#analysis" onClick={event => { if (onReview) { event.preventDefault(); onReview(); } }}>Review video & detections <span aria-hidden="true">↓</span></a>}
    </header>
    <p className="analytics-dashboard__scope"><strong>Match-half target: 45 minutes.</strong> Current hosted test limit: 2 minutes / 250 MiB per clip.</p>
    {measurements && <VideoMeasurementsPanel measurements={measurements} />}

    <details className="analytics-dashboard__filters">
      <summary>Match & historical data filters <span>Awaiting verified data</span></summary>
      <p id={instanceId + "-analytics-filter-help"}>These filters need verified match and historical datasets. No competition, team or player data is connected yet.</p>
      <fieldset disabled aria-describedby={instanceId + "-analytics-filter-help"}>
        <legend>Dataset filters · not available yet</legend>
        <div>{requiredFilters.map(filter => <label key={filter}>{filter}<select defaultValue=""><option value="">Not available yet</option></select></label>)}</div>
      </fieldset>
    </details>

    <nav className="analytics-dashboard__modules" aria-label="Football analytics modules">
      {analyticsAreas.map(area => <button key={area.id} type="button" aria-pressed={activeId === area.id}
        onClick={() => { setActiveId(area.id); setQuery(""); }}>{area.title}</button>)}
    </nav>

    <div className="analytics-dashboard__module-heading">
      <div><h3>{active.title}</h3><p>{active.description}</p></div>
      <div className="analytics-dashboard__search">
        <label htmlFor={instanceId + "-analytics-metric-search"}>Find a metric in this module</label>
        <div><input id={instanceId + "-analytics-metric-search"} type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search metric names" />
          {query && <button type="button" onClick={() => setQuery("")}>Clear</button>}</div>
      </div>
    </div>
    <div className="analytics-dashboard__table-wrap">
      <table><caption className="analytics-dashboard__sr-only">{active.title}: required metrics and data needed</caption>
        <thead><tr><th scope="col">Required metric</th><th scope="col">Value / status</th><th scope="col">Data needed</th></tr></thead>
        <tbody>{matches.map(metric => <tr key={metric.id}>
          <th scope="row">{metric.label}</th><td><span className="analytics-dashboard__value" aria-label="No calculated value">—</span><span className="analytics-dashboard__unavailable">Not available yet</span></td><td>{metric.needs}</td>
        </tr>)}</tbody>
      </table>
      {!matches.length && <div className="analytics-dashboard__empty" role="status"><strong>No matching metrics in {active.title}.</strong><p>Try another metric name or select a different module.</p><button type="button" onClick={() => setQuery("")}>Clear search</button></div>}
    </div>
    <footer className="analytics-dashboard__footer"><p>This catalogue describes the full analytics requirements. Saved approved annotations have a separate reviewed-record summary in Human event review below. Wider analytics still need verified identities and the relevant source data. Detection counts do not establish match statistics.</p>
      {active.id === "workload" && <p>Workload and injury indicators support qualified staff review; they are not diagnoses.</p>}
      <small>Requirements: {active.source}</small></footer>
  </section>;
}
