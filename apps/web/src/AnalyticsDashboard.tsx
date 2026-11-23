import { useId, useMemo, useState } from "react";
import { analyticsAreas, requiredFilters } from "./analytics-requirements";
import { approvedMetric, type MetricState, type MetricValue, type Side } from "./approved-analytics";
import { estimatedMetric } from "./estimated-analytics";
import { useKits } from "./kits";
import type { AnalysisResult } from "./main";
import { num, secs, sentence, words } from "./format";
import { PitchPicker } from "./PitchPicker";
import type { PassNetwork, Review } from "./ReviewPanel";
import { Figures } from "./Figures";
import { DataTable, Disclosure, EmptyState, Field } from "./ui";

type Props = { result: AnalysisResult; saved: Review | null; unsaved?: boolean; onReview?: () => void };

const STATE: Record<MetricState, { label: string; chip: string; meaning: string }> = {
  calculated: { label: "Calculated", chip: "chip--reviewed", meaning: "Calculated from saved, approved records for this clip." },
  partial: { label: "Partly calculated", chip: "chip--draft", meaning: "Part of this is calculated from approved records. The note says what is missing." },
  estimated: { label: "~ Estimated from video", chip: "chip--estimated", meaning: "The model’s own estimate from the tracked ball and players. Not counted from approved events." },
  unavailable: { label: "Not available yet", chip: "chip--unknown", meaning: "Nothing is shown because the data it needs does not exist yet." },
};

/** Who passed to whom. Drawn on the pitch only for players with a located pass; the table below always lists every pair. */
function PassNetworkFigure({ team, network }: { team: string; network: PassNetwork }) {
  const placed = network.nodes.filter(node => node.location);
  const at = new Map(placed.map(node => [node.player, node.location!]));
  const drawn = network.links.filter(link => link.completed > 0 && at.has(link.source) && at.has(link.target));
  const most = Math.max(1, ...drawn.map(link => link.completed));
  const busiest = Math.max(1, ...placed.map(node => node.passes_made + node.passes_received));
  return (
    <div className="stack stack--sm">
      <h4>Pass network · {team}</h4>
      <p className="note">
        {network.completed} of {network.passes} approved {network.passes === 1 ? "pass" : "passes"} with both players named were completed.
        {placed.length >= 2
          ? ` ${placed.length} of ${network.nodes.length} players have a position: the average of where their passes started and where they received completed ones, with ${team} attacking to the right. Thicker lines mean more completed passes.`
          : " Too few of these passes are placed on the pitch, with a direction of play, to draw the network; the pairs are listed below."}
      </p>
      {placed.length >= 2 && (
        <PitchPicker marks={[]} leftLabel={`${team} goal`} rightLabel="Attacking this way →"
          label={`Pass network for ${team}: ${placed.length} players placed and ${drawn.length} passing links drawn. Every pair is in the table below.`}
          overlay={<g className="passnet">
            {drawn.map(link => { const a = at.get(link.source)!, b = at.get(link.target)!; return <line key={`${link.source}>${link.target}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} strokeWidth={1 + 5 * link.completed / most} />; })}
            {placed.map(node => (
              <g key={node.player}>
                <title>{node.player}: {node.passes_made} made, {node.passes_received} received</title>
                <circle cx={node.location!.x} cy={node.location!.y} r={1.6 + 1.6 * (node.passes_made + node.passes_received) / busiest} />
                <text x={node.location!.x} y={node.location!.y - 3.6} textAnchor="middle">{node.number ?? node.player.split(" ").map(part => part[0]).join("").slice(0, 3)}</text>
              </g>
            ))}
          </g>} />
      )}
      <DataTable label={`Passing pairs for ${team}`}>
        <thead><tr><th scope="col">From</th><th scope="col">To</th><th scope="col" className="num">Completed</th><th scope="col" className="num">Recorded</th></tr></thead>
        <tbody>
          {network.links.slice(0, 12).map(link => (
            <tr key={`${link.source}>${link.target}`}><th scope="row">{link.source}</th><td>{link.target}</td><td className="num">{link.completed}</td><td className="num">{link.recorded}</td></tr>
          ))}
        </tbody>
      </DataTable>
      {network.links.length > 12 && <p className="note">Showing the 12 most frequent of {network.links.length} pairs.</p>}
    </div>
  );
}

/**
 * Every analytic the product was asked for. An entry shows a figure only when saved, approved
 * records can supply it; every other entry says what it still needs. Nothing is estimated here.
 */
export function AnalyticsDashboard({ result, saved, unsaved, onReview }: Props) {
  const kits = useKits();
  const instanceId = useId();
  const [activeId, setActiveId] = useState(analyticsAreas[0].id);
  const [query, setQuery] = useState("");
  const [ownSide, setOwnSide] = useState<Side>("home");
  const active = analyticsAreas.find(area => area.id === activeId)!;

  const values = useMemo(() => {
    const map = new Map<string, MetricValue>();
    analyticsAreas.forEach(area => area.metrics.forEach(metric => {
      // Figures counted from approved events come first. Where there are none, the model's own estimate from the
      // video is shown as an estimate; where both exist, the estimate sits beneath and is never added in.
      const approved = saved ? approvedMetric(metric.id, saved, ownSide) : null;
      const estimate = estimatedMetric(metric.id, result, kits.label);
      map.set(metric.id, approved && approved.state !== "unavailable" ? { ...approved, estimate: estimate ?? undefined }
        : estimate ? { state: "estimated", lines: estimate.lines, note: estimate.note }
        : approved ?? { state: "unavailable", lines: [], note: metric.needs });
    }));
    return map;
  }, [saved, ownSide, result, kits]);
  const shown = (areaId: string) => analyticsAreas.find(area => area.id === areaId)!.metrics.filter(metric => values.get(metric.id)!.state !== "unavailable").length;
  const total = analyticsAreas.reduce((sum, area) => sum + area.metrics.length, 0);
  const totalShown = analyticsAreas.reduce((sum, area) => sum + shown(area.id), 0);
  const totalEstimated = [...values.values()].filter(value => value.state === "estimated").length;
  const matches = active.metrics.filter(metric => metric.label.toLowerCase().includes(query.trim().toLowerCase()));
  // Entries with figures come first; the ones still waiting for data are listed compactly beneath them.
  const withFigures = matches.filter(metric => values.get(metric.id)!.state !== "unavailable");
  const missing = matches.filter(metric => values.get(metric.id)!.state === "unavailable");
  const s = saved?.summary;
  const approvedShots = saved?.events.filter(event => event.review_status === "approved" && event.type === "shot") ?? [];
  const placedShots = approvedShots.filter(event => event.location);

  return (
    <section className="stack" aria-labelledby={instanceId + "-title"}>
      <div className="row row--between">
        <div>
          <h4 id={instanceId + "-title"}>{totalShown} of {total} requested analytics have figures for this clip</h4>
          {totalShown > 0 && <p className="note">{totalShown - totalEstimated} counted from approved events · {totalEstimated} estimated from the video by the model.</p>}
          <p className="note" role="status">
            {!saved ? "Loading saved review records…"
              : !s!.approved_events ? "No approved events are saved for this clip yet, so nothing here is a counted event. Estimated rows are the model’s own reading of the video."
              : `From ${num(s!.approved_events, 0)} saved, approved ${s!.approved_events === 1 ? "event" : "events"}${s!.draft_events ? ` (${num(s!.draft_events, 0)} saved ${s!.draft_events === 1 ? "draft" : "drafts"} left out)` : ""}. ${s!.reviewed_video_pct ? `${num(s!.reviewed_video_pct)}% of the clip (${secs(s!.reviewed_seconds)}) is marked as reviewed.` : "No part of the clip is marked as reviewed, so events may be missing anywhere."}`}
          </p>
        </div>
        {onReview && <button type="button" className="btn btn--secondary btn--sm" onClick={onReview}>Record and approve events</button>}
      </div>
      <p className="note note--boxed">
        <strong>These are figures for this clip, not for a match. </strong>
        A row marked “Estimated from video” is the model’s reading of the tracked ball and players: it misses whatever it did not see, and it is not a counted event. A row marked “Calculated” counts events a person approved. Figures that need a season of matches, GPS files or other teams’ data stay unavailable and say why.
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
        <>
          {withFigures.length > 0 && (
            <ul className="metrics" aria-label={`${active.title}: metrics with figures for this clip`}>
              {withFigures.map(metric => {
                const value = values.get(metric.id)!, state = STATE[value.state];
                return (
                  <li key={metric.id} className={`metric metric--${value.state}`}>
                    <div className="metric__head">
                      <h5>{metric.label}</h5>
                      <span className={`chip ${state.chip}`} title={state.meaning}>{value.state === "estimated" && <i className="chip__swatch" aria-hidden="true" />}{state.label}</span>
                    </div>
                    <div className="metric__body">
                      {value.lines.length > 0 && <Figures lines={value.lines} label={`${metric.label}: figures`} />}
                      {value.estimate && (
                        <div className="metric__estimate">
                          <p className="fig__label">Model estimate from the video · not added in</p>
                          <Figures lines={value.estimate.lines} label={`${metric.label}: the model’s estimate from the video`} />
                        </div>
                      )}
                      <p className="metric__note">{value.note ?? (value.state === "calculated" ? "Nothing more at clip level. A match total needs the whole match reviewed." : metric.needs)}</p>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
          {missing.length > 0 && (
            <div className="stack stack--sm">
              <h5>Not available yet · {missing.length}</h5>
              <ul className="metrics metrics--missing" aria-label={`${active.title}: metrics with no figures yet, and what each needs`}>
                {missing.map(metric => (
                  <li key={metric.id} className="metric metric--unavailable">
                    <div className="metric__head"><h5>{metric.label}</h5></div>
                    <p className="metric__note">{values.get(metric.id)!.note ?? metric.needs}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      ) : (
        <EmptyState title={`No matching metrics in ${active.title}`} action={<button type="button" className="btn btn--secondary btn--sm" onClick={() => setQuery("")}>Clear search</button>}>
          Try another metric name or select a different module.
        </EmptyState>
      )}

      {active.id === "match" && placedShots.length > 0 && saved && (
        <div className="stack stack--sm">
          <h4>Shot map</h4>
          <p className="note">
            {placedShots.length} of {approvedShots.length} approved {approvedShots.length === 1 ? "shot has" : "shots have"} a location. Each is where the reviewer clicked on a drawn 105 × 68 m pitch, with left and right as the camera shows them. A filled circle is a goal; a line shows where the shot went, when that was placed.
          </p>
          <PitchPicker label={`Shot map: ${placedShots.length} approved shots placed by the reviewer. They are listed below the pitch.`}
            marks={placedShots.map(event => ({ key: event.id, at: event.location!, to: event.end_location, tone: event.team, filled: event.outcome === "goal", title: `${saved.teams[event.team]} · ${words(event.outcome)} · ${secs(event.timestamp_seconds)}` }))} />
          <p className="note pitchpick__legend"><span><i className="pitchpick__key pitchpick__key--home" />{saved.teams.home}</span><span><i className="pitchpick__key pitchpick__key--away" />{saved.teams.away}</span></p>
          <ul className="itemlist">
            {placedShots.map(event => (
              <li key={event.id}><span><span className="mono">{secs(event.timestamp_seconds)}</span> · {saved.teams[event.team]} · {sentence(event.outcome)}{event.player ? ` · ${event.player}` : ""} · {num(event.location!.x)} m from the left goal line, {num(event.location!.y)} m from the top touchline</span></li>
            ))}
          </ul>
        </div>
      )}

      {active.id === "match" && saved && (["home", "away"] as const).map(side => {
        const network = s?.pass_networks?.[side];
        return network ? <PassNetworkFigure key={side} team={saved.teams[side]} network={network} /> : null;
      })}

      {s?.definitions && (
        <Disclosure summary="How these figures are counted">
          {Object.entries(s.definitions).map(([key, text]) => <p key={key}>{text}</p>)}
        </Disclosure>
      )}
      <p className="note">
        In counted rows, team names are the labels typed under Review. In estimated rows, the sides are the model’s kit-colour groups (Kit A and Kit B), which are not confirmed teams. The two are never mixed in one figure.
        {active.id === "opponents" && " “Your team” only changes which side is treated as the opponent."}
        {active.id === "workload" && " Workload and injury indicators support qualified staff review; they are not diagnoses."}
      </p>
      <p className="note faint">Requirements: {active.source}</p>
    </section>
  );
}
