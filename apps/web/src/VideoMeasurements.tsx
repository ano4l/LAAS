import { useCallback, useState } from "react";
import { num, pct, range, secs, sentence } from "./format";
import { TeamName, useKits } from "./kits";
import { usePlaybackValue } from "./playback";
import { DataTable, Disclosure, EmptyState, Field, Panel, WatchButton, cssVars } from "./ui";

export type VideoMeasurements = {
  schema_version: string; source: string; review_status: string;
  analyzed_frames: number; source_duration_seconds: number; analyzed_frame_pct: number | null;
  mean_visible_players: number | null; max_visible_players: number; ball_coverage_pct: number | null;
  ball_detected_frames: number; ambiguous_ball_frames: number; mean_ball_confidence: number | null;
  longest_ball_detection_gap_seconds: number; ball_proximity_frames: number; proximity_association_coverage_pct: number | null;
  selected_ball_candidate_frames?: number; selected_ball_candidate_coverage_pct?: number | null;
  provisional_kit_analytics?: { kit: string; player_observations: number; near_ball_frames: number; near_ball_seconds: number;
    share_of_known_kit_proximity_pct: number | null; image_shape: { centroid_x_pct: number; centroid_y_pct: number; width_pct: number; depth_pct: number } | null }[];
  event_candidates?: { type: string; timestamp_seconds: number; from_track: string; to_track: string; kit: string;
    image_separation_px: number; review_status: string; source: string }[];
  tracks: { track_id: string; object_type: string; observations: number; observed_seconds: number; visibility_pct: number | null;
    first_seen_seconds: number; last_seen_seconds: number; image_displacement_px: number; excluded_movement_steps: number;
    average_image_x_pct: number; average_image_y_pct: number; near_ball_frames: number }[];
  timeline: { start_seconds: number; end_seconds: number; analyzed_frames: number; ball_coverage_pct: number | null; mean_visible_players: number }[];
  image_occupancy_grid: number[][]; definitions: Record<string, string>;
};

/* ---------- Ball detected through the clip ---------- */

export function BallVisibility({ measurements: m }: { measurements: VideoMeasurements }) {
  const [selectedSegment, setSelectedSegment] = useState(0);
  const segment = m.timeline[Math.min(selectedSegment, m.timeline.length - 1)];
  const timeline = m.timeline;
  const nowIndex = usePlaybackValue(useCallback((seconds: number) => timeline.findIndex(item => seconds >= item.start_seconds && seconds < item.end_seconds), [timeline]));

  return (
    <Panel title="Ball detected through the clip" subtitle="Share of analysed frames with a ball candidate in each stretch of the clip. A candidate is not a confirmed ball.">
      {m.timeline.length ? <>
        <div className="colchart" role="group" aria-label="Select a ball coverage time segment">
          {m.timeline.map((item, index) => (
            <button
              type="button"
              key={item.start_seconds}
              className={`colchart__col ${nowIndex === index ? "is-now" : ""}`}
              aria-pressed={selectedSegment === index}
              aria-label={`${num(item.start_seconds)} to ${num(item.end_seconds)} seconds, ball detected in ${pct(item.ball_coverage_pct)} of frames. Select segment.`}
              onClick={() => setSelectedSegment(index)}
            >
              <span className="colchart__value">{item.ball_coverage_pct === null ? "—" : `${num(item.ball_coverage_pct, 0)}%`}</span>
              <span className={`colchart__track ${item.ball_coverage_pct === null ? "is-unknown" : ""}`}><span style={{ height: item.ball_coverage_pct === null ? undefined : `${item.ball_coverage_pct}%` }} /></span>
              <span className="colchart__time">{num(item.start_seconds)} s</span>
            </button>
          ))}
        </div>
        {segment && (
          <div className="row row--between" role="status">
            <span><strong className="mono">{range(segment.start_seconds, segment.end_seconds, 1)}</strong> <span className="note">ball detected in {pct(segment.ball_coverage_pct)} of frames · {num(segment.mean_visible_players)} players in view on average · {segment.analyzed_frames} analysed frames</span></span>
            <WatchButton at={segment.start_seconds} label="Watch this segment" />
          </div>
        )}
        <Disclosure summary="Segment measurements" meta={`${m.timeline.length} segments`}>
          <DataTable label="Segment measurements table; scroll horizontally to view all columns">
            <thead><tr><th scope="col">Seconds</th><th scope="col" className="num">Frames</th><th scope="col" className="num">Ball detected</th><th scope="col" className="num">Mean visible players</th></tr></thead>
            <tbody>
              {m.timeline.map((item, index) => (
                <tr key={item.start_seconds} className={nowIndex === index ? "is-now" : undefined}>
                  <th scope="row" className="timecell">{num(item.start_seconds)}–{num(item.end_seconds)}</th>
                  <td className="num">{item.analyzed_frames}</td>
                  <td className="num">{pct(item.ball_coverage_pct)}</td>
                  <td className="num">{num(item.mean_visible_players)}</td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        </Disclosure>
      </> : <EmptyState title="No segment measurements for this clip" />}
    </Panel>
  );
}

/* ---------- Where players were seen ---------- */

export function ImageOccupancy({ measurements: m }: { measurements: VideoMeasurements }) {
  const [cell, setCell] = useState({ row: 0, column: 0 });
  const actualMaximum = Math.max(0, ...m.image_occupancy_grid.flat());
  const maxOccupancy = Math.max(1, actualMaximum);
  const rows = m.image_occupancy_grid.length;

  return (
    <Panel title="Where players were seen" subtitle="Player foot-point observations on a 12 × 8 grid of the camera image, all groups together.">
      <div className="chartpair chartpair--even">
        <div className="stack stack--sm">
          <div className="heatgrid" role="img" aria-label="Image occupancy: brighter cells have more player foot-point observations. Use the row and column controls to inspect exact values. Camera image coordinates, not calibrated pitch coordinates.">
            {m.image_occupancy_grid.flatMap((row, y) => row.map((count, x) => (
              <span key={`${x}-${y}`} className={cell.row === y && cell.column === x ? "is-selected" : ""} title={`Image row ${y + 1}, column ${x + 1}: ${count} observations`} style={cssVars({ "--v": 0.05 + .95 * count / maxOccupancy })} />
            )))}
          </div>
          <p className="row row--between note"><span>Top of the image is the top row</span><span className="heatgrid__scale">0 <i aria-hidden="true" /> {num(actualMaximum, 0)} observations</span></p>
        </div>
        <div className="stack stack--sm">
          <div className="toolbar">
            <Field label="Image row">
              <select value={cell.row} onChange={event => setCell({ ...cell, row: Number(event.target.value) })}>
                {m.image_occupancy_grid.map((_, row) => <option key={row} value={row}>{row + 1} {row === 0 ? "· top" : row === rows - 1 ? "· bottom" : ""}</option>)}
              </select>
            </Field>
            <Field label="Image column">
              <select value={cell.column} onChange={event => setCell({ ...cell, column: Number(event.target.value) })}>
                {Array.from({ length: 12 }, (_, column) => <option key={column} value={column}>{column + 1}</option>)}
              </select>
            </Field>
          </div>
          <output aria-live="polite" className="bigout"><strong>{num(m.image_occupancy_grid[cell.row]?.[cell.column] ?? 0, 0)}</strong> observations in this cell</output>
          <p className="note">{m.definitions.occupancy}</p>
        </div>
      </div>
    </Panel>
  );
}

/* ---------- Kit groups (provisional) ---------- */

export function ProvisionalKitPanel({ measurements: m }: { measurements: VideoMeasurements }) {
  const kitNames = useKits();
  const [selectedScene, setSelectedScene] = useState("all");
  const kits = m.provisional_kit_analytics ?? [];
  const sceneOf = (kit: string) => kit.includes("/") ? kit.split("/")[0] : "Unknown scene";
  const sceneLabel = (scene: string) => { const match = /^scene-(\d+)$/.exec(scene); return match ? `Scene ${Number(match[1]) + 1}` : scene; };
  const scenes = [...new Set([...kits.map(kit => sceneOf(kit.kit)), ...(m.event_candidates ?? []).map(event => sceneOf(event.kit))])];
  const scene = scenes.includes(selectedScene) ? selectedScene : "all";
  const visibleKits = kits.filter(kit => scene === "all" || sceneOf(kit.kit) === scene);
  const knownProximityFrames = kits.reduce((total, kit) => total + kit.near_ball_frames, 0);
  const knownProximitySeconds = kits.reduce((total, kit) => total + kit.near_ball_seconds, 0);
  const knownProximityCoverage = m.analyzed_frames > 0 ? 100 * knownProximityFrames / m.analyzed_frames : null;
  const candidates = (m.event_candidates ?? []).filter(event => scene === "all" || sceneOf(event.kit) === scene);

  return (
    <Panel
      title="Kit groups"
      subtitle="Jersey-colour groups, worked out separately for each scene. They are not verified teams or players; the same label in another scene may be different people."
      aside={scenes.length > 0 && (
        <Field label="Scene" className="field--inline">
          <select value={scene} onChange={event => setSelectedScene(event.target.value)}>
            <option value="all">All scenes · groups stay separate</option>
            {scenes.map(item => <option key={item} value={item}>{sceneLabel(item)}</option>)}
          </select>
        </Field>
      )}
    >
      <p className="row row--between">
        <span>{m.provisional_kit_analytics === undefined ? "Group count unknown for this run" : <><strong>{visibleKits.length}</strong> anonymous {visibleKits.length === 1 ? "group" : "groups"} in this view</>}</span>
        <span className="note">Time near the ball is not possession.</span>
      </p>
      {visibleKits.length > 0 ? (
        <DataTable label="Anonymous kit measurements table; scroll horizontally to view image shape columns" stack>
          <thead><tr><th scope="col">Kit group</th><th scope="col" className="num">Player observations</th><th scope="col" className="num">Near the ball</th><th scope="col" className="num">Share of near-ball observations</th><th scope="col" className="num">Image centre x / y</th><th scope="col" className="num">Image width / depth</th></tr></thead>
          <tbody>
            {visibleKits.map(kit => (
              <tr key={kit.kit}>
                <th scope="row"><TeamName id={kit.kit} />{!kitNames.multiScene && scenes.length > 1 && <small>{sceneLabel(sceneOf(kit.kit))}</small>}</th>
                <td className="num" data-label="Player observations">{num(kit.player_observations, 0)}</td>
                <td className="num" data-label="Near the ball"><div>{secs(kit.near_ball_seconds, 2)}<small>{num(kit.near_ball_frames, 0)} frames</small></div></td>
                <td className="num" data-label="Share of near-ball observations">{kit.share_of_known_kit_proximity_pct === null ? "Unknown" : pct(kit.share_of_known_kit_proximity_pct)}</td>
                <td className="num" data-label="Image centre x / y">{kit.image_shape ? `${num(kit.image_shape.centroid_x_pct)}% / ${num(kit.image_shape.centroid_y_pct)}%` : "Unknown"}</td>
                <td className="num" data-label="Image width / depth">{kit.image_shape ? `${num(kit.image_shape.width_pct)}% / ${num(kit.image_shape.depth_pct)}%` : "Unknown"}</td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      ) : <EmptyState title={m.provisional_kit_analytics === undefined ? "Kit groups are unknown for this run" : "No kit groups in this view"}>{m.provisional_kit_analytics === undefined ? "This analysis did not record kit observations." : "No anonymous kit groups were established here."}</EmptyState>}

      <div className="stack stack--sm">
        <div className="row row--between">
          <h4>To review: possible same-kit transfers</h4>
          <span className="chip chip--unknown"><i className="chip__swatch" aria-hidden="true" />Unreviewed · not counted</span>
        </div>
        <p className="note">A change in which track is nearest the ball, within one kit group. These are prompts to check, not completed passes, touches or possession statistics.</p>
        {candidates.length > 0 ? (
          <DataTable label="Possible same-kit transfer candidates; scroll to view all columns" stack>
            <thead><tr><th scope="col">Time</th><th scope="col">Candidate</th><th scope="col">Kit group</th><th scope="col">From → to track</th><th scope="col" className="num">Image separation</th><th scope="col">Review</th><th scope="col"><span className="sr-only">Video</span></th></tr></thead>
            <tbody>
              {candidates.map((event, index) => (
                <tr key={`${event.timestamp_seconds}-${event.from_track}-${event.to_track}-${index}`}>
                  <th scope="row" className="timecell">{secs(event.timestamp_seconds, 2)}</th>
                  <td data-label="Candidate">Possible same-kit transfer</td>
                  <td data-label="Kit group"><TeamName id={event.kit} /></td>
                  <td className="mono" data-label="From → to track">{event.from_track} → {event.to_track}</td>
                  <td className="num" data-label="Image separation">{num(event.image_separation_px)} px</td>
                  <td data-label="Review"><div>Unreviewed<small>Ball-proximity heuristic</small></div></td>
                  <td className="actions"><WatchButton at={event.timestamp_seconds} /></td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        ) : <p className="note" role="status">{m.event_candidates === undefined ? "Transfer candidates are unknown for this run." : "No possible same-kit transfers in this view. Completed pass statistics remain unknown."}</p>}
      </div>

      <Disclosure summary="How this is measured">
        {m.provisional_kit_analytics !== undefined && (
          <p>
            <strong>Share denominator across the entire clip: </strong>{num(knownProximityFrames, 0)} known-kit proximity frames / {num(m.analyzed_frames, 0)} analysed frames
            ({knownProximityCoverage === null ? "coverage unknown" : `${num(knownProximityCoverage)}% coverage`}), approximately {secs(knownProximitySeconds, 2)}. Shares exclude unknown associations and may cover only a small fraction of the clip.
          </p>
        )}
        <p>Share uses all known kit proximity observations in this run. Selecting a scene does not recalculate shares or the support above. Shape averages require at least three visible grouped players. Percentages refer to camera-image coordinates and include camera effects.</p>
        {m.definitions.kits && <p><strong>Kits: </strong>{m.definitions.kits}</p>}
        {m.definitions.event_candidates && <p><strong>Event candidates: </strong>{m.definitions.event_candidates}</p>}
        {m.definitions.proximity && <p><strong>Proximity: </strong>{m.definitions.proximity}</p>}
      </Disclosure>
    </Panel>
  );
}

/** Every measurement definition from the run, optionally limited to some keys. */
export function MeasurementDefinitions({ measurements: m, only }: { measurements: VideoMeasurements; only?: string[] }) {
  return <>{Object.entries(m.definitions).filter(([key]) => !only || only.includes(key)).map(([key, description]) => <p key={key}><strong>{sentence(key)}: </strong>{description}</p>)}</>;
}
