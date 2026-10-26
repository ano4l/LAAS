import { useId, useState } from "react";

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

const number = (value: number | null, digits = 1) => value === null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
const percent = (value: number | null) => value === null ? "Unknown" : `${number(value)}%`;

export function VideoMeasurementsPanel({ measurements: m, onSeek, showTracks = true }: { measurements: VideoMeasurements; onSeek?: (seconds: number) => void; showTracks?: boolean }) {
  const instanceId = useId();
  const [query, setQuery] = useState("");
  const [selectedSegment, setSelectedSegment] = useState(0);
  const [cell, setCell] = useState({ row: 0, column: 0 });
  const segment = m.timeline[Math.min(selectedSegment, m.timeline.length - 1)];
  const actualMaximum = Math.max(0, ...m.image_occupancy_grid.flat());
  const maxOccupancy = Math.max(1, actualMaximum);
  const matches = m.tracks.filter(track => `${track.track_id} ${track.object_type}`.toLowerCase().includes(query.trim().toLowerCase()));
  return <section className="video-measurements" aria-labelledby={instanceId + "-video-measurements-title"}>
    <header><div><p className="eyebrow">Measured from your video · unreviewed</p><h3 id={instanceId + "-video-measurements-title"}>Video observations</h3></div><span className="video-measurements__badge">Camera-relative</span></header>
    <p>These measurements describe visible detections across {number(m.source_duration_seconds)} seconds. Track IDs are temporary; they are not verified player identities.</p>
    <div className="video-measurements__summary">
      <article><span>Analyzed frames</span><strong>{number(m.analyzed_frames, 0)}</strong><small>{m.analyzed_frame_pct === null ? "Source coverage unknown" : `${percent(m.analyzed_frame_pct)} of source frames`}</small></article>
      <article><span>Mean visible players</span><strong>{number(m.mean_visible_players)}</strong><small>Maximum {number(m.max_visible_players, 0)} in an analyzed frame</small></article>
      <article><span>Ball candidate coverage</span><strong>{number(m.ball_coverage_pct)}{m.ball_coverage_pct !== null && "%"}</strong><small>{number(m.ball_detected_frames, 0)} frames · {number(m.ambiguous_ball_frames, 0)} with multiple candidates</small></article>
      <article><span>Longest ball detection gap</span><strong>{number(m.longest_ball_detection_gap_seconds)}s</strong><small>Across analyzed intervals</small></article>
    </div>
    <div className="video-measurements__visuals">
      <section className="video-measurements__timeline" aria-labelledby={instanceId + "-observation-timeline-title"}><h4 id={instanceId + "-observation-timeline-title"}>Ball visibility through the clip</h4><p>Height shows candidate coverage in each time segment.</p>
        <div className="video-measurements__bars" role="group" aria-label="Select a ball coverage time segment">{m.timeline.map((item, index) => <button type="button" key={item.start_seconds} className="video-measurements__segment" aria-pressed={selectedSegment === index} aria-label={`${number(item.start_seconds)} to ${number(item.end_seconds)} seconds, ball coverage ${percent(item.ball_coverage_pct)}. Select segment.`} onClick={() => setSelectedSegment(index)}>
          <span className="video-measurements__bar-value">{number(item.ball_coverage_pct, 0)}{item.ball_coverage_pct !== null && "%"}</span>
          <span className={`video-measurements__bar-track ${item.ball_coverage_pct === null ? "unknown-coverage" : ""}`}><span style={{ height: item.ball_coverage_pct === null ? undefined : `${item.ball_coverage_pct}%` }} /></span>
          <span className="video-measurements__bar-time">{number(item.start_seconds)}s</span>
        </button>)}</div>
        {segment && <div className="studio-segment-detail" role="status"><div><strong>{number(segment.start_seconds)}–{number(segment.end_seconds)}s</strong><span>{percent(segment.ball_coverage_pct)} ball candidate coverage · {number(segment.mean_visible_players)} mean visible players · {segment.analyzed_frames} analyzed frames</span></div>{onSeek && <button type="button" className="ghost" onClick={() => onSeek(segment.start_seconds)}>Watch this segment →</button>}</div>}
        <details><summary>View segment measurements</summary><div className="video-measurements__scroll" tabIndex={0} role="region" aria-label="Segment measurements table; scroll horizontally to view all columns"><table><thead><tr><th scope="col">Seconds</th><th scope="col">Frames</th><th scope="col">Ball coverage</th><th scope="col">Mean visible players</th></tr></thead><tbody>{m.timeline.map(segment => <tr key={segment.start_seconds}><th scope="row">{number(segment.start_seconds)}–{number(segment.end_seconds)}</th><td>{segment.analyzed_frames}</td><td>{percent(segment.ball_coverage_pct)}</td><td>{number(segment.mean_visible_players)}</td></tr>)}</tbody></table></div></details>
      </section>
      <section className="video-measurements__occupancy" aria-labelledby={instanceId + "-image-occupancy-title"}><h4 id={instanceId + "-image-occupancy-title"}>Image occupancy · 12 × 8</h4><p>Player foot-point observations in the camera image.</p>
        <div className="video-measurements__grid" role="img" aria-label="Image occupancy: brighter cells have more player foot-point observations. Use the row and column controls below to inspect exact values. Camera image coordinates, not calibrated pitch coordinates.">{m.image_occupancy_grid.flatMap((row, y) => row.map((count, x) => <span key={`${x}-${y}`} className={cell.row === y && cell.column === x ? "selected-cell" : ""} title={`Image row ${y + 1}, column ${x + 1}: ${count} observations`} style={{ background: `rgba(167, 145, 255, ${0.04 + .92 * count / maxOccupancy})` }} />))}</div>
        <div className="video-measurements__grid-legend"><span>Image top → bottom</span><span>0 <i aria-hidden="true" /> {actualMaximum} observations</span></div>
        <div className="studio-cell-inspector"><label>Image row<select value={cell.row} onChange={e => setCell({ ...cell, row: Number(e.target.value) })}>{m.image_occupancy_grid.map((_, row) => <option key={row} value={row}>{row + 1} {row === 0 ? "· top" : row === 7 ? "· bottom" : ""}</option>)}</select></label><label>Image column<select value={cell.column} onChange={e => setCell({ ...cell, column: Number(e.target.value) })}>{Array.from({ length: 12 }, (_, column) => <option key={column} value={column}>{column + 1}</option>)}</select></label><output aria-live="polite"><strong>{m.image_occupancy_grid[cell.row]?.[cell.column] ?? 0}</strong> observations</output></div><p className="video-measurements__caution">{m.definitions.occupancy}</p>
      </section>
    </div>
    {(m.selected_ball_candidate_frames !== undefined || m.selected_ball_candidate_coverage_pct !== undefined) && <section className="video-measurements__selection" aria-labelledby={instanceId + "-selected-ball-title"}>
      <div><h4 id={instanceId + "-selected-ball-title"}>Selected ball candidate coverage</h4><p>One observed candidate retained after confidence and continuity checks. Selection is not a verified detection or an accuracy score.</p></div>
      <div><strong>{number(m.selected_ball_candidate_coverage_pct ?? null)}{m.selected_ball_candidate_coverage_pct != null && "%"}</strong><span>{m.selected_ball_candidate_frames === undefined ? "Observed frame count unknown" : `${number(m.selected_ball_candidate_frames, 0)} analyzed frames with a selected candidate`}</span></div>
    </section>}
    {(m.provisional_kit_analytics !== undefined || m.event_candidates !== undefined) && <ProvisionalKitPanel measurements={m} />}
    {showTracks && <section className="video-measurements__tracks" aria-labelledby={instanceId + "-observed-tracks-title"}><div className="video-measurements__track-heading"><div><h4 id={instanceId + "-observed-tracks-title"}>Observed tracks</h4><p>Ordered by observed frames. Visibility uses all analyzed intervals.</p></div><label>Find an observed track<input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Track ID or class" /></label></div>
      <div className="video-measurements__scroll" tabIndex={0} role="region" aria-label="Observed tracks table; scroll to view all tracks and columns"><table><thead><tr><th scope="col">Track / class</th><th scope="col">Observed</th><th scope="col">Visibility</th><th scope="col">Image displacement</th><th scope="col">Near-ball frames</th></tr></thead><tbody>{matches.slice(0, 100).map(track => <tr key={track.track_id}><th scope="row">{track.track_id}<small>{track.object_type}</small></th><td>{number(track.observed_seconds)}s</td><td>{number(track.visibility_pct)}%</td><td>{number(track.image_displacement_px)} px<small>{track.excluded_movement_steps} excluded steps</small></td><td>{number(track.near_ball_frames, 0)}</td></tr>)}</tbody></table></div>
      {!matches.length && <p role="status">No matching observed tracks.</p>}<p>Showing {Math.min(100, matches.length)} of {matches.length} matching tracks. Full measurements are in the analysis JSON.</p>
    </section>}
    <aside className="video-measurements__notes"><h4>Read these measurements with their limits</h4><p><strong>Ball proximity is not possession or touches.</strong> {number(m.ball_proximity_frames, 0)} frames meet the proximity rule ({percent(m.proximity_association_coverage_pct)} of analyzed frames). Mean ball confidence: {m.mean_ball_confidence === null ? "Unknown" : percent(m.mean_ball_confidence * 100)}. Model confidence is not measured accuracy.</p>
      <details><summary>Measurement definitions</summary>{Object.entries(m.definitions).map(([key, description]) => <p key={key}><strong>{key[0].toUpperCase() + key.slice(1)}: </strong>{description}</p>)}</details>
      <p>Image displacement includes camera motion and tracking errors. It is not distance run or speed on the pitch.</p>
    </aside>
  </section>;
}

function ProvisionalKitPanel({ measurements: m }: { measurements: VideoMeasurements }) {
  const instanceId = useId();
  const [selectedScene, setSelectedScene] = useState("all");
  const kits = m.provisional_kit_analytics ?? [];
  const sceneOf = (kit: string) => kit.includes("/") ? kit.split("/")[0] : "Unknown scene";
  const scenes = [...new Set([...kits.map(kit => sceneOf(kit.kit)), ...(m.event_candidates ?? []).map(event => sceneOf(event.kit))])];
  const scene = scenes.includes(selectedScene) ? selectedScene : "all";
  const visibleKits = kits.filter(kit => scene === "all" || sceneOf(kit.kit) === scene);
  const knownProximityFrames = kits.reduce((total, kit) => total + kit.near_ball_frames, 0);
  const knownProximitySeconds = kits.reduce((total, kit) => total + kit.near_ball_seconds, 0);
  const knownProximityCoverage = m.analyzed_frames > 0 ? 100 * knownProximityFrames / m.analyzed_frames : null;
  const candidates = (m.event_candidates ?? []).filter(event => scene === "all" || sceneOf(event.kit) === scene);
  return <section className="video-measurements__kits" aria-labelledby={instanceId + "-kit-observations-title"}>
    <div className="video-measurements__kit-heading"><div><p className="eyebrow">Provisional grouping · unreviewed</p><h4 id={instanceId + "-kit-observations-title"}>Anonymous kit observations</h4></div>
      {scenes.length > 0 && <label>Scene scope<select value={scene} onChange={event => setSelectedScene(event.target.value)}><option value="all">All scenes · groups stay separate</option>{scenes.map(item => <option key={item} value={item}>{item}</option>)}</select></label>}
    </div>
    <p>Jersey-colour groups are scoped to each scene. They are not verified teams or players; the same group label in another scene may refer to different people.</p>
    <p className="video-measurements__kit-total">{m.provisional_kit_analytics === undefined ? "Group count unknown for this run" : <><strong>{visibleKits.length}</strong> anonymous {visibleKits.length === 1 ? "group" : "groups"} in this view</>}<span>Near-ball observation share is not possession.</span></p>
    {visibleKits.length > 0 ? <div className="video-measurements__scroll" tabIndex={0} role="region" aria-label="Anonymous kit measurements table; scroll horizontally to view image shape columns"><table className="video-measurements__kit-table"><thead><tr><th scope="col">Scene / kit group</th><th scope="col">Player observations</th><th scope="col">Near-ball seconds</th><th scope="col">Near-ball observation share</th><th scope="col">Image centroid x / y</th><th scope="col">Image width / depth</th></tr></thead><tbody>{visibleKits.map(kit => <tr key={kit.kit}>
      <th scope="row">{kit.kit}</th><td>{number(kit.player_observations, 0)}</td><td>{number(kit.near_ball_seconds)}s<small>{number(kit.near_ball_frames, 0)} frames</small></td><td>{kit.share_of_known_kit_proximity_pct === null ? "Unknown" : `${number(kit.share_of_known_kit_proximity_pct)}%`}</td><td>{kit.image_shape ? `${number(kit.image_shape.centroid_x_pct)}% / ${number(kit.image_shape.centroid_y_pct)}%` : "Unknown"}</td><td>{kit.image_shape ? `${number(kit.image_shape.width_pct)}% / ${number(kit.image_shape.depth_pct)}%` : "Unknown"}</td>
    </tr>)}</tbody></table></div> : <p className="video-measurements__empty" role="status">{m.provisional_kit_analytics === undefined ? "Kit observations are unknown for this run." : "No anonymous kit groups were established in this view."}</p>}
    {m.provisional_kit_analytics !== undefined && <p className="video-measurements__support"><strong>Share denominator across the entire clip:</strong> {number(knownProximityFrames, 0)} known-kit proximity frames / {number(m.analyzed_frames, 0)} analyzed frames ({knownProximityCoverage === null ? "coverage unknown" : `${number(knownProximityCoverage)}% coverage`}), approximately {number(knownProximitySeconds, 3)} seconds. Shares exclude unknown associations and may cover only a small fraction of the clip.</p>}
    <p>Share uses all known kit proximity observations in this run. Selecting a scene does not recalculate shares or the support above. Shape averages require at least three visible grouped players. Percentages refer to camera-image coordinates and include camera effects.</p>
    <div className="video-measurements__candidate-heading"><h4>Possible same-kit transfer candidates</h4><span className="video-measurements__badge">Unreviewed</span></div>
    <p>Changes in ball-proximity association between temporary tracks are review prompts. They are not completed passes, touches or possession statistics.</p>
    {candidates.length > 0 ? <div className="video-measurements__scroll" tabIndex={0} role="region" aria-label="Possible same-kit transfer candidates; scroll to view all columns"><table><thead><tr><th scope="col">Time</th><th scope="col">Candidate</th><th scope="col">Scene / kit</th><th scope="col">From → to track</th><th scope="col">Image separation</th><th scope="col">Review</th></tr></thead><tbody>{candidates.map((event, index) => <tr key={`${event.timestamp_seconds}-${event.from_track}-${event.to_track}-${index}`}><th scope="row">{number(event.timestamp_seconds)}s</th><td>Possible same-kit transfer</td><td>{event.kit}</td><td>{event.from_track} → {event.to_track}</td><td>{number(event.image_separation_px)} px</td><td>Unreviewed<small>Ball-proximity heuristic</small></td></tr>)}</tbody></table></div> : <p className="video-measurements__empty" role="status">{m.event_candidates === undefined ? "Transfer candidates are unknown for this run." : "No possible same-kit transfer candidates in this view. Completed pass statistics remain unknown."}</p>}
  </section>;
}
