import { useCallback, useMemo, useState } from "react";
import { num, pct, range, secs, sentence, words } from "./format";
import { TeamName, useKits } from "./kits";
import { usePlaybackValue } from "./playback";
import { DataTable, Disclosure, EmptyState, Field, Pager, StackedBar, StatTile, Swatch, WatchButton, type BarSegment } from "./ui";
import { KIND_FILL, STATE_LABEL, STATE_TECHNICAL, segmentKind } from "./vocabulary";

export type SequenceState = "controlled" | "in_transit" | "control_gap" | "team_retained" | "contested" | "unknown" | "dead_ball";
export type SequenceInterval = { start_seconds: number; end_seconds: number; state: SequenceState; team: string | null; kit: string | null; track_id: string | null; support: string };
export type TemporalSequences = {
  schema_version: string; review_status: string; context_revision: number; mapping_status: "anonymous_scene_groups" | "user_confirmed";
  source_duration_seconds: number; classified_non_dead_seconds: number; known_seconds: number; unknown_seconds: number; contested_seconds: number; dead_ball_seconds: number; coverage_pct: number | null;
  by_team: { team: string; estimated_seconds: number; share_of_known_pct: number | null; share_of_non_dead_pct: number | null; observed_control_seconds: number; flight_and_gap_seconds: number; bounded_missing_ball_seconds: number }[];
  intervals: SequenceInterval[];
  anchors: { id: string; scene_id: number; track_id: string; kit: string | null; team: string | null; start_seconds: number; end_seconds: number; observed_seconds: number; observed_frames: number; support_type: string }[];
  candidates: { id: string; [key: string]: unknown }[];
  diagnostics: { anchors: number; motion_supported_touches: number; rejected_transitions: Record<string, number> };
  parameters: Record<string, number | string>; limitations: string;
  // Inferred time between two consecutive touches by the same team. Reported beside the supported totals, never inside them.
  retention?: { max_gap_seconds: number; retained_seconds: number; touch_pairs: number; known_with_retention_seconds: number; coverage_with_retention_pct: number | null; definition: string;
    by_team: { team: string; retained_seconds: number; with_retention_seconds: number; share_with_retention_pct: number | null; share_of_non_dead_with_retention_pct: number | null }[] };
};

const STATES = Object.keys(STATE_LABEL) as SequenceState[];
const hasAssumed = (s: TemporalSequences) => Boolean(s.retention && s.retention.retained_seconds > 0);

/** One sentence that answers "how much can we say about possession?" using only reported fields. */
export function possessionTakeaway(s: TemporalSequences | undefined): string {
  if (!s) return "This older analysis has no possession estimate.";
  if (!(s.known_seconds > 0)) return "The model could not support a possession estimate anywhere in this clip. Possession is unknown, not an even split.";
  const base = `The model can support a possession estimate for ${pct(s.coverage_pct)} of playing time (${secs(s.known_seconds)} of ${secs(s.classified_non_dead_seconds)}).`;
  const assumed = hasAssumed(s) ? ` Assuming a team keeps the ball between its own touches raises that to ${pct(s.retention!.coverage_with_retention_pct)}.` : "";
  return `${base}${assumed} The rest is unknown.`;
}

/**
 * Playing time as one bar: each team's supported time (solid) and assumed time (hatched), then
 * contested time. Everything left over is unknown and stays the colour of the ground.
 */
export function PossessionBar({ sequences: s, size = "tall" }: { sequences: TemporalSequences; size?: "tall" | "thin" }) {
  const kits = useKits();
  const segments: BarSegment[] = [];
  s.by_team.forEach(team => {
    const tone = kits.tone(team.team);
    const assumed = s.retention?.by_team.find(item => item.team === team.team);
    segments.push({ key: `${team.team}-supported`, value: team.estimated_seconds, fill: "solid", tone, title: `${kits.label(team.team)} · supported · ${secs(team.estimated_seconds)}` });
    if (assumed) segments.push({ key: `${team.team}-assumed`, value: assumed.retained_seconds, fill: "hatch", tone, title: `${kits.label(team.team)} · assumed between touches · ${secs(assumed.retained_seconds)}` });
  });
  segments.push({ key: "contested", value: s.contested_seconds, fill: "contested", title: `Contested · ${secs(s.contested_seconds)}` });
  const label = `Playing time of ${secs(s.classified_non_dead_seconds)}. ${s.by_team.map(team => `${kits.label(team.team)}: ${secs(team.estimated_seconds)} supported`).join(". ")}. Unknown: ${secs(s.unknown_seconds)}.`;
  return <StackedBar segments={segments} total={s.classified_non_dead_seconds} label={label} size={size} />;
}

/** Per-team rows: share of playing time we can support, and the share once assumed time is added. */
export function PossessionTeams({ sequences: s, detail = false }: { sequences: TemporalSequences; detail?: boolean }) {
  const kits = useKits();
  const assumedShown = hasAssumed(s);
  return (
    <ul className="teamrows">
      {s.by_team.map(team => {
        const tone = kits.tone(team.team);
        const assumed = s.retention?.by_team.find(item => item.team === team.team);
        return (
          <li key={team.team} className="teamrow">
            <div className="teamrow__name"><TeamName id={team.team} /></div>
            <div className="teamrow__figure" title="Technical name: share of non-dead time, from classified known time">
              <strong>{pct(team.share_of_non_dead_pct)}</strong>
              <span>of playing time, supported · {secs(team.estimated_seconds)}</span>
            </div>
            {assumedShown && (
              <div className="teamrow__figure teamrow__figure--assumed" title="Technical name: share of non-dead time with possession retained between touches (inferred)">
                <strong>{pct(assumed?.share_of_non_dead_with_retention_pct)}</strong>
                <span>including assumed · {secs(assumed?.with_retention_seconds)}</span>
              </div>
            )}
            <div className="teamrow__bar">
              <StackedBar
                size="thin"
                total={s.classified_non_dead_seconds}
                label={`${kits.label(team.team)}: ${secs(team.estimated_seconds)} supported${assumed ? `, ${secs(assumed.retained_seconds)} assumed` : ""} of ${secs(s.classified_non_dead_seconds)} playing time`}
                segments={[
                  { key: "supported", value: team.estimated_seconds, fill: "solid", tone },
                  { key: "assumed", value: assumed?.retained_seconds ?? 0, fill: "hatch", tone },
                ]}
              />
            </div>
            {detail && (
              <p className="teamrow__detail">
                On the ball {secs(team.observed_control_seconds)} · ball travelling and brief gaps {secs(team.flight_and_gap_seconds)}, including {secs(team.bounded_missing_ball_seconds)} with the ball briefly out of sight
                {assumed ? ` · kept between touches (assumed) ${secs(assumed.retained_seconds)}` : ""}.
              </p>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function BarKey({ sequences: s }: { sequences: TemporalSequences }) {
  return (
    <ul className="keyline">
      <li><Swatch fill="solid" />Supported estimate</li>
      {hasAssumed(s) && <li><Swatch fill="hatch" />Assumed between touches</li>}
      {s.contested_seconds > 0 && <li><Swatch fill="contested" />Contested</li>}
      <li><Swatch fill="unknown" />Unknown</li>
    </ul>
  );
}

/** The split between the teams, over only the time that has an estimate. */
function TeamSplit({ sequences: s }: { sequences: TemporalSequences }) {
  const kits = useKits();
  const row = (title: string, seconds: number, fill: "solid" | "hatch", share: (team: string) => number | null | undefined, hint: string) => (
    <div className="split" title={hint}>
      <p className="split__label">{title} <span className="mono">{secs(seconds)}</span></p>
      <StackedBar
        total={100}
        label={`${title}: ${s.by_team.map(team => `${kits.label(team.team)} ${pct(share(team.team))}`).join(", ")}`}
        segments={s.by_team.map(team => ({ key: team.team, value: share(team.team) ?? 0, fill, tone: kits.tone(team.team) }))}
      />
      <ul className="split__values">
        {s.by_team.map(team => <li key={team.team}><TeamName id={team.team} /> <b className="mono">{pct(share(team.team))}</b></li>)}
      </ul>
    </div>
  );
  return (
    <div className="splits">
      {row("Split of supported time", s.known_seconds, "solid", team => s.by_team.find(item => item.team === team)?.share_of_known_pct, "Technical name: share of known classified time")}
      {hasAssumed(s) && row("Split including assumed time", s.retention!.known_with_retention_seconds, "hatch", team => s.retention!.by_team.find(item => item.team === team)?.share_with_retention_pct, "Technical name: share of supported plus retained time")}
    </div>
  );
}

/** Summary-sized view: the bar, its key and the per-team shares. */
export function PossessionOverview({ sequences: s }: { sequences?: TemporalSequences }) {
  if (!s) return <EmptyState title="No possession estimate for this clip">This older analysis has no possession estimate. Nothing is assumed in its place.</EmptyState>;
  return (
    <div className="stack">
      <PossessionBar sequences={s} />
      <BarKey sequences={s} />
      {s.known_seconds > 0 && s.by_team.length
        ? <PossessionTeams sequences={s} />
        : <p className="note" role="status">No team possession is supported. Team shares are unknown, not an even split.</p>}
    </div>
  );
}

/** The full possession estimate for the Possession tab. */
export function TemporalPossession({ sequences: s }: { sequences?: TemporalSequences }) {
  const kits = useKits();
  const [state, setState] = useState("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [selectedBin, setSelectedBin] = useState(0);

  // Time bins keep the mix of states inside each slice, for inspecting exact durations.
  const bins = useMemo(() => {
    if (!s || s.source_duration_seconds <= 0) return [];
    const count = Math.min(120, Math.max(1, Math.ceil(s.source_duration_seconds * 4)));
    return Array.from({ length: count }, (_, index) => {
      const start = index * s.source_duration_seconds / count, end = (index + 1) * s.source_duration_seconds / count;
      const parts = new Map<string, { state: SequenceState; team: string | null; seconds: number }>();
      s.intervals.forEach(row => {
        const seconds = Math.max(0, Math.min(end, row.end_seconds) - Math.max(start, row.start_seconds));
        if (!seconds) return;
        const visualState = row.state === "controlled" && !row.team ? "unknown" : row.state;
        const key = `${visualState}/${row.team ?? ""}`;
        const old = parts.get(key);
        parts.set(key, { state: visualState, team: row.team, seconds: (old?.seconds ?? 0) + seconds });
      });
      const observed = [...parts.values()].reduce((sum, part) => sum + part.seconds, 0);
      if (observed < end - start - 1e-6) {
        const old = parts.get("unknown/");
        parts.set("unknown/", { state: "unknown", team: null, seconds: (old?.seconds ?? 0) + end - start - observed });
      }
      return { start, end, parts: [...parts.values()] };
    });
  }, [s]);

  const intervals = s?.intervals;
  const nowIndex = usePlaybackValue(useCallback((seconds: number) => intervals ? intervals.findIndex(row => seconds >= row.start_seconds && seconds < row.end_seconds) : -1, [intervals]));

  if (!s) {
    return (
      <EmptyState title="No possession estimate for this clip">
        This older analysis has no possession estimate over time. The strict near-foot observations below are a separate rule and are not team possession.
      </EmptyState>
    );
  }

  const selected = bins[Math.min(selectedBin, bins.length - 1)];
  const matches = s.intervals.filter(row => (state === "all" || row.state === state) && `${row.team ?? "unknown"} ${row.kit ?? ""} ${row.track_id ?? ""} ${row.support} ${kits.label(row.team, "")}`.toLowerCase().includes(search.trim().toLowerCase()));
  const pages = Math.max(1, Math.ceil(matches.length / 8));
  const currentPage = Math.min(page, pages - 1);
  const now = nowIndex >= 0 ? s.intervals[nowIndex] : undefined;

  return (
    <div className="stack stack--lg">
      <div className="tiles tiles--four">
        <StatTile label="Possession we can support" value={num(s.known_seconds)} unit="s" evidence="estimated" title="Technical name: classified known time (temporal classified team time)" context={`${pct(s.coverage_pct)} of ${secs(s.classified_non_dead_seconds)} playing time`} />
        <StatTile
          label="Including assumed possession between touches"
          value={s.retention ? num(s.retention.known_with_retention_seconds) : "Unknown"}
          unit="s"
          evidence="assumed"
          title="Technical name: with possession retained between touches (inferred)"
          context={s.retention ? `${pct(s.retention.coverage_with_retention_pct)} of playing time · adds ${secs(s.retention.retained_seconds)} from ${s.retention.touch_pairs} touch pairs` : "Not calculated for this analysis"}
        />
        <StatTile label="Unknown" value={num(s.unknown_seconds)} unit="s" evidence="unknown" title="Technical name: unknown / unassigned non-dead time" context={hasAssumed(s) ? "Stays in playing time · includes the assumed time" : "Stays in playing time; never shared out"} />
        <StatTile label="Contested" value={num(s.contested_seconds)} unit="s" chip={<span className="chip" title="Both teams are at the ball; no team is credited"><Swatch fill="contested" />Contested</span>} context="No team is credited" />
      </div>

      <p className="facts">
        <span>Clip length<b>{secs(s.source_duration_seconds)}</b></span>
        <span title="Technical name: non-dead time, the denominator for every share">Playing time (clip minus confirmed stoppages)<b>{secs(s.classified_non_dead_seconds)}</b></span>
        <span title="Technical name: confirmed stoppages (dead-ball time)">Stoppages (confirmed)<b>{secs(s.dead_ball_seconds)}</b></span>
      </p>

      <div className="stack">
        <div className="row row--between">
          <h4>Playing time, by team and certainty</h4>
          <BarKey sequences={s} />
        </div>
        <PossessionBar sequences={s} />
        {s.known_seconds > 0 && s.by_team.length ? <>
          <PossessionTeams sequences={s} detail />
          <TeamSplit sequences={s} />
          <p className="note">The splits compare the teams over the supported time only. They do not describe the {secs(s.unknown_seconds)} that is unknown.</p>
        </> : <p className="note note--boxed" role="status">No team possession is supported. Team shares are unknown, not an even possession split.</p>}
      </div>

      <div className="stack">
        <div className="row row--between">
          <h4>Intervals</h4>
          <span className="note" role="status">
            {now ? <>At the playhead: {STATE_LABEL[now.state]}{now.team ? <> · <TeamName id={now.team} /></> : ""}</> : "The playhead is outside every interval"}
          </span>
        </div>
        <div className="toolbar">
          <Field label="State" className="field--fixed">
            <select value={state} onChange={event => { setState(event.target.value); setPage(0); }}>
              <option value="all">All intervals</option>
              {STATES.map(key => <option key={key} value={key}>{STATE_LABEL[key]}</option>)}
            </select>
          </Field>
          <Field label="Find an interval">
            <input type="search" value={search} onChange={event => { setSearch(event.target.value); setPage(0); }} placeholder="Team, kit group, track or support" />
          </Field>
        </div>
        {matches.length > 0 ? (
          <DataTable label="Possession intervals in source time" stack>
            <thead><tr><th scope="col">Time</th><th scope="col">State</th><th scope="col">Team</th><th scope="col">Support</th><th scope="col"><span className="sr-only">Video</span></th></tr></thead>
            <tbody>
              {matches.slice(currentPage * 8, (currentPage + 1) * 8).map(row => {
                const kind = segmentKind(row.state, row.team);
                return (
                  <tr key={`${row.start_seconds}-${row.state}`} className={row === now ? "is-now" : undefined} aria-current={row === now ? "true" : undefined}>
                    <th scope="row" className="timecell">{range(row.start_seconds, row.end_seconds)}<small>{secs(row.end_seconds - row.start_seconds, 2)} long</small></th>
                    <td data-label="State" title={`Technical name: ${STATE_TECHNICAL[row.state]}`}>
                      <span className="statecell"><Swatch fill={KIND_FILL[kind]} tone={kits.tone(row.team)} />{STATE_LABEL[row.state]}</span>
                    </td>
                    <td data-label="Team"><div><TeamName id={row.team} />{row.kit && row.kit !== row.team && <small title={row.kit}>{kits.anonymous(row.kit)}</small>}{row.track_id && <small>Track {row.track_id}</small>}</div></td>
                    <td data-label="Support">{sentence(row.support)}</td>
                    <td className="actions"><WatchButton at={row.start_seconds} describe="Watch interval" /></td>
                  </tr>
                );
              })}
            </tbody>
          </DataTable>
        ) : <p className="note" role="status">No intervals match these filters.</p>}
        <Pager page={currentPage} pages={pages} count={matches.length} noun="intervals" onPage={setPage} />
      </div>

      {selected && (
        <Disclosure summary="Inspect a slice of the clip" meta={`${bins.length} time bins`}>
          <p>Each bin keeps the exact mix of states inside it. Move the slider to read the durations for one slice.</p>
          <Field label="Time bin">
            <input type="range" min="0" max={bins.length - 1} step="1" value={Math.min(selectedBin, bins.length - 1)} onChange={event => setSelectedBin(Number(event.target.value))} />
          </Field>
          <div className="row row--between" role="status">
            <div>
              <strong className="mono">{range(selected.start, selected.end)}</strong>
              <ul>
                {selected.parts.map((part, index) => <li key={index}>{STATE_LABEL[part.state]} · {part.team ? kits.label(part.team) : "No team"}: {secs(part.seconds, 2)}</li>)}
              </ul>
            </div>
            <WatchButton at={selected.start} label="Watch this slice" />
          </div>
        </Disclosure>
      )}

      <Disclosure summary="How this is measured" meta="Definitions and limits">
        <p>
          {s.mapping_status === "user_confirmed" ? "Uses user-confirmed team context." : "Teams are anonymous scene-scoped kit groups; no Home/Away or real-team identity is assumed."}{" "}
          Estimates include supported flight and short gaps, without inventing ball coordinates. They are model estimates and have not been reviewed.
        </p>
        <p title={`Technical name: context revision ${s.context_revision}`}>
          <strong>Context applied. </strong>
          {s.mapping_status === "user_confirmed" ? "Team names confirmed by a person" : "No confirmed team names"}, {s.dead_ball_seconds > 0 ? `${secs(s.dead_ball_seconds, 2)} of confirmed stoppages` : "no confirmed stoppages"} (saved context version {s.context_revision}).
        </p>
        <p><strong>Limits. </strong>{s.limitations}</p>
        <p>
          Full source: {secs(s.source_duration_seconds, 2)}. Playing time (the non-dead denominator): {secs(s.classified_non_dead_seconds, 2)}. Team splits divide only the {secs(s.known_seconds, 2)} of possession
          we can support (classified known time); shares of playing time keep all non-dead time in the denominator.
        </p>
        {s.retention && (
          <p>
            <strong>Assumed possession. </strong>
            {secs(s.retention.known_with_retention_seconds, 2)} ({pct(s.retention.coverage_with_retention_pct, 2)} of playing time) once a team is assumed to keep the ball between two of its own consecutive touches no more than{" "}
            {secs(s.retention.max_gap_seconds, 2)} apart. This adds {secs(s.retention.retained_seconds, 2)} from {s.retention.touch_pairs} touch pairs. It is an assumption, not an observation: an opponent touch the model missed
            would make it wrong, and this time still counts as unknown in the totals above. {s.retention.definition}
          </p>
        )}
        <p>
          {s.diagnostics.anchors} supported touches (anchors), including {s.diagnostics.motion_supported_touches} motion-supported touches.
          {Object.keys(s.diagnostics.rejected_transitions ?? {}).length > 0 && <> Rejected transitions: {Object.entries(s.diagnostics.rejected_transitions).map(([reason, count]) => `${words(reason)} ${count}`).join(" · ")}.</>}
          {" "}{bins.length} time bins preserve duration mixes; exact intervals and touches are in the exports. No coordinates are generated where the ball is missing.
        </p>
        <div>
          <h5>Names used here and in the data</h5>
          <dl className="deflist">
            <div><dt>Possession we can support</dt><dd>classified known time</dd></div>
            <div><dt>Playing time</dt><dd>non-dead time</dd></div>
            {STATES.map(key => <div key={key}><dt>{STATE_LABEL[key]}</dt><dd>{STATE_TECHNICAL[key]}</dd></div>)}
          </dl>
        </div>
        <div>
          <h5>Parameters</h5>
          <dl className="deflist">{Object.entries(s.parameters).map(([key, value]) => <div key={key}><dt>{words(key)}</dt><dd>{String(value)}</dd></div>)}</dl>
        </div>
      </Disclosure>

    </div>
  );
}
