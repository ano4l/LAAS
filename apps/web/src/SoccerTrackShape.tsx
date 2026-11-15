import { useCallback, useId, useState } from "react";
import { num, pct, range, secs } from "./format";
import { TeamName, useKits } from "./kits";
import { usePlaybackValue } from "./playback";
import { PitchBoard, type BoardGroup } from "./PitchBoard";
import { ShapeStage, type StageGroup } from "./ShapeStage";
import { DataTable, Disclosure, EmptyState, EvidenceChip, Field, Pager, Panel, StatTile, WatchButton } from "./ui";

export type ShapeEllipse = { centroid_x_pct: number; centroid_y_pct: number; major_radius_px: number; minor_radius_px: number; angle_degrees: number; area_image_pct: number; rms_spread_px: number; point_count: number };
/** Estimated positions on an assumed pitch, present only when that frame passed automatic calibration. */
type PitchSnapshot = { players: { track_id: string; x_m: number; y_m: number }[]; players_not_placed: number; landmarks_used: number; fit_error_m: number; stabilised?: boolean; visible_area_m: [number, number][]; length_extent_m: number | null; width_extent_m: number | null; hull_area_m2: number | null };
type PitchStatus = { status: string; reason?: string | null; assumed_pitch?: { length_m: number; width_m: number; measured: boolean }; frames_tried?: number; frames_calibrated?: number; frames_stabilised?: number; limitations?: string[] };
type MotionTrack = { track_id: string; measured_seconds: number; distance_m: number; mean_speed_m_s: number; fastest_hop_m_s: number; rejected_hops: number };
/** Whole-clip movement estimates on the assumed pitch; only time tracked in a calibrated frame counts. */
type PitchMotion = { status: string; reason?: string | null; calibrated_seconds?: number; coverage_seconds?: number[][]; grid: { columns: number; rows: number; cell_length_m: number; cell_width_m: number }; parameters: { position_average_seconds: number; hop_seconds: number; max_speed_m_s: number; min_listed_track_seconds: number }; limitations: string[]; teams: { kit: string; scene_id: number; placed_player_seconds: number; heatmap_player_seconds: number[][]; measured_player_seconds: number; distance_m: number; mean_speed_m_s: number | null; fastest_hop_m_s: number | null; fastest_hop_track: string | null; tracks_measured: number; tracks: MotionTrack[]; tracks_listed_total: number }[] };
type Snapshot = { timestamp_seconds: number; players: { track_id: string; x_pct: number; y_pct: number }[]; ellipse: ShapeEllipse; pitch?: PitchSnapshot };
export type DynamicsWindow = { start_seconds: number; end_seconds: number; supported_seconds: number; observed_frames: number; mean_visible_players: number; mean_area_image_pct: number; centroid_x_pct: number; centroid_y_pct: number; motion_alignment: number | null; alignment_supported_seconds: number; snapshot: Snapshot };
export type CollectiveDynamics = { schema_version: string; reference: { title: string; url: string; version: number }; pitch?: PitchStatus; pitch_motion?: PitchMotion; coordinate_system: string; review_status: string; source_duration_seconds: number; formation: { status: string; label: string | null; reason: string }; teams: { kit: string; scene_id: number; supported_seconds: number; source_coverage_pct: number; mean_visible_players: number; mean_area_image_pct: number; mean_alignment: number | null; timeline: DynamicsWindow[]; activity_ellipses_total?: number; activity_ellipses: { track_id: string; observed_seconds: number; observed_frames: number; ellipse: ShapeEllipse }[] }[]; parameters: Record<string, string | number>; limitations: string[] };

const FORMATION_PLAIN = "Naming a formation such as 4‑3‑3 or 4‑4‑2 needs calibrated pitch coordinates, every outfield player in view and reliable identities. This view shows the shape of the players the camera could see.";

/** Collective shape of each visible kit group, measured in the camera image (after the SoccerTrack notebook). */
export function SoccerTrackShape({ data, imageWidth, imageHeight, duration, cleanSrc, annotatedSrc }: { data?: CollectiveDynamics; imageWidth?: number; imageHeight?: number; duration: number; cleanSrc?: string; annotatedSrc?: string }) {
  const id = useId();
  const kits = useKits();
  const [kitId, setKitId] = useState(data?.teams[0]?.kit ?? "");
  const [second, setSecond] = useState(0);
  const [follow, setFollow] = useState(true);
  const [activityPage, setActivityPage] = useState(0);
  const [footage, setFootage] = useState(true);
  // The original upload gives a clean frame. If the browser cannot play it, the annotated video is used instead.
  const [cleanFailed, setCleanFailed] = useState<string | null>(null);
  const usingClean = Boolean(cleanSrc) && cleanFailed !== cleanSrc;
  const videoSrc = usingClean ? cleanSrc : annotatedSrc;
  const [showOther, setShowOther] = useState(true);
  const [showIds, setShowIds] = useState(false);
  const pitchAvailable = data?.pitch?.status === "estimated";
  const [view, setView] = useState<"pitch" | "heat" | "camera">(pitchAvailable ? "pitch" : "camera");
  const motionData = data?.pitch_motion?.status === "estimated" ? data.pitch_motion : undefined;
  const pitchLength = data?.pitch?.assumed_pitch?.length_m ?? 105, pitchWidth = data?.pitch?.assumed_pitch?.width_m ?? 68;
  const team = data?.teams.find(group => group.kit === kitId) ?? data?.teams[0];
  const sourceDuration = data?.source_duration_seconds ?? duration;
  const seconds = Math.max(1, Math.ceil(sourceDuration));
  const videoSecond = usePlaybackValue(useCallback((time: number) => Math.floor(time), []));
  const selectedSecond = Math.min(follow ? videoSecond : second, seconds - 1);
  const chosen = team?.timeline.find(window => window.start_seconds <= selectedSecond && window.end_seconds > selectedSecond);
  const width = imageWidth && imageWidth > 0 ? imageWidth : undefined, height = imageHeight && imageHeight > 0 ? imageHeight : undefined;
  const snapshot = chosen?.snapshot;
  const ellipse = snapshot?.ellipse;
  const activities = team?.activity_ellipses ?? [];
  const pages = Math.max(1, Math.ceil(activities.length / 8));
  const page = Math.min(activityPage, pages - 1);
  const maxArea = Math.max(.01, ...(team?.timeline.map(window => window.mean_area_image_pct) ?? []));
  const tone = kits.tone(team?.kit);
  // The other kit group in the same scene, drawn alongside only when its snapshot is from the very same moment.
  const other = team ? data?.teams.find(group => group.kit !== team.kit && group.scene_id === team.scene_id) : undefined;
  const otherSnapshot = other?.timeline.find(window => window.start_seconds <= selectedSecond && window.end_seconds > selectedSecond)?.snapshot;
  const sameMoment = Boolean(snapshot && otherSnapshot && Math.abs(otherSnapshot.timestamp_seconds - snapshot.timestamp_seconds) < .001);
  const stageGroups: StageGroup[] = snapshot && team ? [
    ...(showOther && other && otherSnapshot && sameMoment ? [{ kit: other.kit, label: kits.label(other.kit), tone: kits.tone(other.kit), primary: false, players: otherSnapshot.players, ellipse: otherSnapshot.ellipse }] : []),
    { kit: team.kit, label: kits.label(team.kit), tone, primary: true, players: snapshot.players, ellipse: snapshot.ellipse },
  ] : [];
  const pitch = snapshot?.pitch;
  const boardGroups: BoardGroup[] = pitch && team ? [
    ...(showOther && other && otherSnapshot?.pitch && sameMoment ? [{ kit: other.kit, label: kits.label(other.kit), tone: kits.tone(other.kit), primary: false, players: otherSnapshot.pitch.players }] : []),
    { kit: team.kit, label: kits.label(team.kit), tone, primary: true, players: pitch.players },
  ] : [];
  const motion = motionData && team ? motionData.teams.find(group => group.kit === team.kit && group.scene_id === team.scene_id) : undefined;
  const kmh = (speed: number) => num(speed * 3.6);
  const notPlaced = (pitch?.players_not_placed ?? 0) + (boardGroups.length > 1 ? otherSnapshot?.pitch?.players_not_placed ?? 0 : 0);

  return (
    <Panel
      title="Team shape"
      subtitle="How each visible group spreads, moves and changes shape through the clip, measured in the camera image."
      aside={data && data.teams.length > 0 && (
        <Field label="Shape group" className="field--inline">
          <select value={team?.kit ?? ""} onChange={event => { setKitId(event.target.value); setActivityPage(0); }}>
            {data.teams.map(group => <option key={group.kit} value={group.kit}>{kits.label(group.kit)}</option>)}
          </select>
        </Field>
      )}
    >
      <p className="note note--boxed" title={data?.formation.reason ? `Analysis wording: ${data.formation.reason}` : undefined}><strong>Named tactical formation unavailable. </strong>{FORMATION_PLAIN}</p>

      {!data ? <EmptyState title="No shape measurements for this clip">Collective shape was not calculated in this older analysis. No player positions or formation are assumed.</EmptyState>
        : !data.teams.length || !team ? <EmptyState title="Shape unknown for this clip">No group has enough supported player positions for a shape ellipse. Missing evidence stays unknown.</EmptyState>
        : <div className={`stack stack--lg ${tone ? `tone-${tone}` : ""}`}>
          <div className="tiles tiles--compact tiles--four">
            <StatTile label="Shape evidence" value={num(team.supported_seconds)} unit="s" evidence="seen" context={`${pct(team.source_coverage_pct)} of the clip · ${secs(Math.max(0, sourceDuration - team.supported_seconds))} unsupported`} />
            <StatTile label="Players in the group, on average" value={num(team.mean_visible_players)} evidence="seen" title="Technical name: mean visible group" context="Observed players, not a full line-up" />
            <StatTile label="Average spread" value={num(team.mean_area_image_pct, 2)} unit="%" evidence="estimated" title="Technical name: mean characteristic area, the product of the ellipse's one-standard-deviation radii" context="Share of the camera-image area" />
            <StatTile label="Moving together" value={team.mean_alignment == null ? "Unknown" : num(team.mean_alignment, 3)} evidence="estimated" title="Technical name: camera-image alignment (0–1), supported moving tracks only" context="0–1 · image motion, not proven coordination" />
          </div>

          <div className="shapepair">
            <div className="stack stack--sm">
              <p className="row row--between note">
                <span><TeamName id={team.kit} /> · scene {team.scene_id + 1} · {view === "camera" ? "camera frame, x right, y down" : "pitch seen from above, far touchline at the top"}</span>
                <span className="mono">{view === "heat" ? "Whole clip" : snapshot ? `Snapshot ${secs(snapshot.timestamp_seconds, 2)}` : `${range(selectedSecond, Math.min(sourceDuration, selectedSecond + 1), 1)} · no snapshot`}</span>
              </p>
              {pitchAvailable && (
                <div className="viewswitch" role="group" aria-label="Diagram view">
                  <button type="button" aria-pressed={view === "pitch"} onClick={() => setView("pitch")}>Pitch map <EvidenceChip level="estimated" /></button>
                  {motionData && <button type="button" aria-pressed={view === "heat"} onClick={() => setView("heat")}>Heat map <EvidenceChip level="estimated" /></button>}
                  <button type="button" aria-pressed={view === "camera"} onClick={() => setView("camera")}>Camera view <EvidenceChip level="seen" /></button>
                </div>
              )}
              {view === "heat" && motionData ? (motion ? (
                <PitchBoard
                  length={pitchLength}
                  width={pitchWidth}
                  groups={[]}
                  visibleArea={[]}
                  showIds={false}
                  heat={{ cells: motion.heatmap_player_seconds, coverage: motionData.coverage_seconds ?? [], tone, label: kits.label(team.kit) }}
                  label={`Estimated heat map for ${kits.label(team.kit)} over the whole clip, seen from above: where its tracked players were, in ${motionData.grid.columns} by ${motionData.grid.rows} cells, from ${num(motion.placed_player_seconds)} player-seconds placed on an assumed ${pitchLength} by ${pitchWidth} metre pitch. Dark cells were never in the camera’s view. An unreviewed estimate.`}
                />
              ) : (
                <div className="pitchboard pitchboard--empty" style={{ aspectRatio: `${pitchLength + 9}/${pitchWidth + 9}` }}>
                  <p className="camera__message" role="status">No player of this group was placed on the pitch map, so there is no heat map for it.</p>
                </div>
              )) : view === "pitch" && pitchAvailable ? (pitch ? (
                <PitchBoard
                  length={pitchLength}
                  width={pitchWidth}
                  groups={boardGroups}
                  visibleArea={pitch.visible_area_m}
                  showIds={showIds}
                  label={`Estimated positions of ${pitch.players.length} ${kits.label(team.kit)} players on the pitch at ${num(snapshot!.timestamp_seconds, 2)} seconds, seen from above${boardGroups.length > 1 && other && otherSnapshot?.pitch ? `, with ${otherSnapshot.pitch.players.length} players of ${kits.label(other.kit)}` : ""}. Mapped from the camera frame onto an assumed ${pitchLength} by ${pitchWidth} metre pitch; an unreviewed estimate. The dimmed part of the pitch was outside the camera frame.`}
                />
              ) : (
                <div className="pitchboard pitchboard--empty" style={{ aspectRatio: `${pitchLength + 9}/${pitchWidth + 9}` }}>
                  <p className="camera__message" role="status">{snapshot ? "No pitch map for this second. The frame did not pass the calibration checks, so no positions are placed. The camera view still shows what was seen." : "No supported group shape in this second. No positions are filled in."}</p>
                </div>
              )) : snapshot && ellipse && width && height ? (
                <ShapeStage
                  width={width}
                  height={height}
                  time={snapshot.timestamp_seconds}
                  videoSrc={videoSrc}
                  onFrameError={usingClean && annotatedSrc ? () => setCleanFailed(cleanSrc ?? null) : undefined}
                  footage={footage}
                  showIds={showIds}
                  groups={stageGroups}
                  label={`Observed ${snapshot.players.length}-player shape for ${kits.label(team.kit)} at ${num(snapshot.timestamp_seconds, 2)} seconds${stageGroups.length > 1 && other && otherSnapshot ? `, with ${otherSnapshot.players.length} players of ${kits.label(other.kit)}` : ""}. Outline of the outermost players, typical-spread ellipse and player footpoints in the camera image${footage && videoSrc ? ", drawn over the video frame from that moment" : ""}; not a pitch map.`}
                />
              ) : (
                <div className="camera camera--shape" style={width && height ? { aspectRatio: `${width}/${height}` } : undefined}>
                  <p className="camera__message" role="status">{snapshot ? "Source image dimensions are unavailable. Inspect the numeric snapshot below." : "No supported group shape in this second. No positions are filled in."}</p>
                </div>
              )}
              {view !== "heat" && <div className="row shapestage-options">
                {videoSrc && view === "camera" && <label className="check"><input type="checkbox" checked={footage} onChange={event => setFootage(event.target.checked)} />Real footage behind</label>}
                {other && <label className="check"><input type="checkbox" checked={showOther} onChange={event => setShowOther(event.target.checked)} />Show {kits.label(other.kit)} too</label>}
                <label className="check"><input type="checkbox" checked={showIds} onChange={event => setShowIds(event.target.checked)} />Track IDs</label>
              </div>}
              {view === "heat" && motionData ? <>
                <ul className="keyline">
                  <li><i className="shape__key shape__key--heat" aria-hidden="true" />Less to more time spent here</li>
                  <li><i className="shape__key shape__key--unseen" aria-hidden="true" />Never in the camera’s view: unknown</li>
                </ul>
                <p className="note">
                  <strong>Estimate, not a measurement. </strong>
                  Each cell is about {num(motionData.grid.cell_length_m)} × {num(motionData.grid.cell_width_m)} m and adds up the time this group’s tracked players were placed in it, across the {secs(motionData.calibrated_seconds ?? 0)} of the clip that could be mapped onto the pitch.
                  A cell with no colour was in view but no player of this group was placed there. The camera follows play, so parts of the pitch it rarely showed look emptier than they were.
                </p>
              </> : view === "pitch" && pitchAvailable ? <>
                <ul className="keyline">
                  <li title="Technical name: detection foot point projected through a per-frame homography"><i className="shape__key shape__key--dot" aria-hidden="true" />Estimated player position</li>
                  <li title="Technical name: convex hull of the estimated positions"><i className="shape__key shape__key--hull" aria-hidden="true" />Outline of the group (outermost players)</li>
                  <li><i className="shape__key shape__key--unseen" aria-hidden="true" />Outside the camera frame: nothing known</li>
                </ul>
                {pitch && <p className="note">
                  <strong>Estimate, not a measurement. </strong>
                  Mapped from {pitch.landmarks_used} pitch markings found in the frame at {secs(snapshot!.timestamp_seconds, 2)}, which {pitch.stabilised ? "the steadied mapping matches" : "agree with each other"} to about {num(pitch.fit_error_m, 2)} m.
                  The pitch is assumed to be {pitchLength} × {pitchWidth} m, and positions have not been checked against measured ones.
                  {notPlaced > 0 ? ` ${notPlaced} ${notPlaced === 1 ? "player" : "players"} seen in the frame did not land on the pitch and ${notPlaced === 1 ? "is" : "are"} not drawn.` : ""}
                </p>}
              </> : <>
              <ul className="keyline">
                <li title="Technical name: observed player footpoint"><i className="shape__key shape__key--ring" aria-hidden="true" />Where a player stood</li>
                <li title="Technical name: convex hull of the observed footpoints"><i className="shape__key shape__key--hull" aria-hidden="true" />Outline of the group (outermost players)</li>
                <li title="Technical name: one-standard-deviation (1σ) covariance ellipse"><i className="shape__key shape__key--ellipse" aria-hidden="true" />Typical spread</li>
                <li><i className="shape__key shape__key--centre" aria-hidden="true" />Centre of the group</li>
              </ul>
              {footage && videoSrc && snapshot && <p className="note">The picture behind is the {usingClean ? "original video" : "annotated video"} frame at {secs(snapshot.timestamp_seconds, 2)}, the same moment the positions were measured.{usingClean ? "" : " The boxes and numbers in it are the tracker's own drawing."} Rings are drawn at each player's feet.</p>}
              </>}

              <div className="scrub">
                <Field label="Shape at this second">
                  <input type="range" min="0" max={seconds - 1} step="1" value={selectedSecond} onChange={event => { setFollow(false); setSecond(Number(event.target.value)); }} />
                </Field>
                <div className="row row--between" role="status">
                  <span><strong className="mono">{range(selectedSecond, Math.min(sourceDuration, selectedSecond + 1), 1)}</strong> <span className="note">{chosen ? `${secs(chosen.supported_seconds, 2)} support · ${chosen.observed_frames} observed frames` : "No supported shape in this second"}</span></span>
                  <WatchButton at={snapshot?.timestamp_seconds ?? selectedSecond} label={snapshot ? "Watch shape snapshot" : "Watch this second"} />
                </div>
                <label className="check"><input type="checkbox" checked={follow} onChange={event => { setFollow(event.target.checked); setSecond(selectedSecond); }} />Follow the video</label>
              </div>
            </div>

            <div className="stack stack--sm">
              <h4>{chosen ? "This second, weighted means" : "This second: no evidence"}</h4>
              {chosen ? (
                <dl className="deflist">
                  <div><dt>Players in the group</dt><dd>{num(chosen.mean_visible_players, 2)}</dd></div>
                  <div><dt>Spread (characteristic area)</dt><dd>{pct(chosen.mean_area_image_pct, 2)} of image</dd></div>
                  <div><dt>Group centre</dt><dd>x {pct(chosen.centroid_x_pct)} · y {pct(chosen.centroid_y_pct)}</dd></div>
                  <div><dt>Moving together (alignment)</dt><dd>{chosen.motion_alignment == null ? "Unavailable" : num(chosen.motion_alignment, 3)}<small>{secs(chosen.alignment_supported_seconds, 2)} motion support</small></dd></div>
                </dl>
              ) : <p className="note">No mean or alignment is assigned here. The full clip stays in the support denominator.</p>}
              {pitchAvailable && snapshot && <>
                <h4 className="row">On the pitch, at the snapshot <EvidenceChip level="estimated" /></h4>
                {pitch ? (
                  <dl className="deflist">
                    <div><dt>Goal-to-goal depth</dt><dd>{pitch.length_extent_m == null ? "Unknown" : `about ${num(pitch.length_extent_m)} m`}</dd></div>
                    <div><dt>Side-to-side width</dt><dd>{pitch.width_extent_m == null ? "Unknown" : `about ${num(pitch.width_extent_m)} m`}</dd></div>
                    <div><dt>Area inside the outline</dt><dd>{pitch.hull_area_m2 == null ? "Unknown" : `about ${num(pitch.hull_area_m2, 0)} m²`}</dd></div>
                    <div><dt>Players placed</dt><dd>{pitch.players.length} of {snapshot.players.length}<small>visible players of this group only</small></dd></div>
                  </dl>
                ) : <p className="note">Unknown for this second: the frame did not pass the pitch calibration checks.</p>}
              </>}
              {motionData && <>
                <h4 className="row">Movement on the pitch, whole clip <EvidenceChip level="estimated" /></h4>
                {motion && motion.measured_player_seconds > 0 && motion.mean_speed_m_s != null ? (
                  <dl className="deflist">
                    <div><dt>Distance while tracked</dt><dd>about {num(motion.distance_m, 0)} m<small>{motion.tracks_measured} {motion.tracks_measured === 1 ? "track" : "tracks"} added together, {secs(motion.measured_player_seconds, 0)} of player time</small></dd></div>
                    <div><dt>Average speed</dt><dd>about {num(motion.mean_speed_m_s)} m/s<small>{kmh(motion.mean_speed_m_s)} km/h, while tracked</small></dd></div>
                    <div><dt>Fastest half-second</dt><dd>{motion.fastest_hop_m_s == null ? "Unknown" : <>about {num(motion.fastest_hop_m_s)} m/s<small>{kmh(motion.fastest_hop_m_s)} km/h · track {motion.fastest_hop_track}</small></>}</dd></div>
                  </dl>
                ) : <p className="note">Unknown: no track of this group stayed on the pitch map long enough to measure.</p>}
                <p className="note">Not match totals. Only time a player was tracked in a mapped frame counts, and a track is not a verified player.</p>
              </>}
              {motion && motion.tracks.length > 0 && (
                <Disclosure summary="Movement by track" meta={`${motion.tracks_listed_total} tracks`}>
                  <DataTable label="Estimated movement of each track on the assumed pitch">
                    <thead><tr><th scope="col">Track</th><th scope="col" className="num">Measured</th><th scope="col" className="num">Distance · m</th><th scope="col" className="num">Average · m/s</th><th scope="col" className="num">Fastest ½ s · m/s</th></tr></thead>
                    <tbody>{motion.tracks.map(row => <tr key={row.track_id}><th scope="row" className="mono">{row.track_id}</th><td className="num">{secs(row.measured_seconds)}</td><td className="num">{num(row.distance_m)}</td><td className="num">{num(row.mean_speed_m_s)}</td><td className="num">{num(row.fastest_hop_m_s)}</td></tr>)}</tbody>
                  </DataTable>
                  <p>Tracks measured for at least {motionData?.parameters.min_listed_track_seconds} s{motion.tracks_listed_total > motion.tracks.length ? `; the ${motion.tracks.length} longest are shown` : ""}. Estimates on an assumed {pitchLength} × {pitchWidth} m pitch.</p>
                </Disclosure>
              )}
              {ellipse && (
                <Disclosure summary="Exact snapshot measurements">
                  <dl className="deflist deflist--single">
                    <div><dt>Observed points</dt><dd>{ellipse.point_count}</dd></div>
                    <div><dt>Snapshot area</dt><dd>{pct(ellipse.area_image_pct, 2)} of image</dd></div>
                    <div><dt>Major / minor radius</dt><dd>{num(ellipse.major_radius_px, 2)} / {num(ellipse.minor_radius_px, 2)} px</dd></div>
                    <div><dt>Ellipse angle</dt><dd>{num(ellipse.angle_degrees, 2)}° in image axes</dd></div>
                    <div><dt>RMS spread</dt><dd>{num(ellipse.rms_spread_px, 2)} px</dd></div>
                  </dl>
                </Disclosure>
              )}
              {pitch && (
                <Disclosure summary="Estimated pitch positions" meta={`${pitch.players.length} tracks`}>
                  <DataTable label="Estimated snapshot player positions on the assumed pitch">
                    <thead><tr><th scope="col">Track</th><th scope="col" className="num">Along · m</th><th scope="col" className="num">Across · m</th></tr></thead>
                    <tbody>{pitch.players.map(player => <tr key={player.track_id}><th scope="row" className="mono">{player.track_id}</th><td className="num">{num(player.x_m)}</td><td className="num">{num(player.y_m)}</td></tr>)}</tbody>
                  </DataTable>
                  <p>Along runs from the left goal line as the main camera sees it; across runs from the far touchline. Estimates on an assumed {pitchLength} × {pitchWidth} m pitch.</p>
                </Disclosure>
              )}
              {snapshot && (
                <Disclosure summary="Snapshot positions" meta={`${snapshot.players.length} tracks`}>
                  <DataTable label="Observed snapshot player footpoints">
                    <thead><tr><th scope="col">Track</th><th scope="col" className="num">X image %</th><th scope="col" className="num">Y image %</th></tr></thead>
                    <tbody>{snapshot.players.map(player => <tr key={player.track_id}><th scope="row" className="mono">{player.track_id}</th><td className="num">{num(player.x_pct, 2)}</td><td className="num">{num(player.y_pct, 2)}</td></tr>)}</tbody>
                  </DataTable>
                </Disclosure>
              )}
            </div>
          </div>

          <div className="stack stack--sm">
            <h4>Spreading out and closing up through the clip</h4>
            <p className="note">Average spread per second, as a share of the camera image. Gaps are seconds with no supported shape; a mark on the baseline is a supported shape with zero area.</p>
            <figure className="areachart">
              <svg viewBox="0 0 600 100" preserveAspectRatio="none" role="img" aria-label={`Characteristic area across the full ${num(sourceDuration)} second source for ${kits.label(team.kit)}. Separate bars show supported intervals; empty slots are unknown. Inspect with the time control for exact values.`}>
                {team.timeline.map(window => window.mean_area_image_pct === 0
                  ? <line key={window.start_seconds} className="areachart__zero" x1={600 * window.start_seconds / sourceDuration} x2={600 * window.end_seconds / sourceDuration - 1} y1="96" y2="96" />
                  : <rect key={window.start_seconds} x={600 * window.start_seconds / sourceDuration} y={96 - 88 * window.mean_area_image_pct / maxArea} width={Math.max(.3, 600 * (window.end_seconds - window.start_seconds) / sourceDuration - 1)} height={88 * window.mean_area_image_pct / maxArea} className={selectedSecond >= window.start_seconds && selectedSecond < window.end_seconds ? "is-selected" : undefined} />)}
                <line className="areachart__now" x1={600 * selectedSecond / sourceDuration} x2={600 * selectedSecond / sourceDuration} y1="0" y2="100" />
              </svg>
              <figcaption><span>0 s</span><span>Scale 0–{pct(maxArea, 2)}</span><span>{secs(sourceDuration)}</span></figcaption>
            </figure>
          </div>

          <Disclosure summary="Shape data by second" meta={`${team.timeline.length} supported seconds`}>
            <DataTable label="Selected group source-time shape data" stack>
              <thead><tr><th scope="col">Time</th><th scope="col" className="num">Support</th><th scope="col" className="num">Players</th><th scope="col" className="num">Area · image %</th><th scope="col" className="num">Alignment</th><th scope="col"><span className="sr-only">Video</span></th></tr></thead>
              <tbody>
                {team.timeline.filter(window => Math.abs(window.start_seconds - selectedSecond) <= 4).map(window => (
                  <tr key={window.start_seconds} className={window === chosen ? "is-now" : undefined}>
                    <th scope="row" className="timecell">{range(window.start_seconds, window.end_seconds)}</th>
                    <td className="num" data-label="Support">{secs(window.supported_seconds, 2)}</td>
                    <td className="num" data-label="Players">{num(window.mean_visible_players, 2)}</td>
                    <td className="num" data-label="Area · image %">{num(window.mean_area_image_pct, 2)}</td>
                    <td className="num" data-label="Alignment">{window.motion_alignment == null ? "Unavailable" : num(window.motion_alignment, 3)}</td>
                    <td className="actions"><WatchButton at={window.snapshot.timestamp_seconds} /></td>
                  </tr>
                ))}
              </tbody>
            </DataTable>
            <p>Showing supported seconds within four seconds of the selected time. All of them are in Insights JSON and CSV.</p>
          </Disclosure>

          <Disclosure summary="Where each track spent its time" meta={`${activities.length} tracks`}>
            <p>Per-track positional spread across observed time in this scene (temporary-track activity ellipses). Includes camera motion; this is not physical activity territory or a confidence interval. One player can be split across several tracks.</p>
            {activities.length ? <>
              <DataTable label="Temporary-track activity ellipse data" stack>
                <thead><tr><th scope="col">Track</th><th scope="col" className="num">Observed</th><th scope="col" className="num">Mean image position</th><th scope="col" className="num">Area · image %</th><th scope="col" className="num">Radii · px</th></tr></thead>
                <tbody>
                  {activities.slice(page * 8, (page + 1) * 8).map(activity => (
                    <tr key={activity.track_id}>
                      <th scope="row" className="mono">{activity.track_id}</th>
                      <td className="num" data-label="Observed">{secs(activity.observed_seconds, 2)} · {activity.observed_frames} frames</td>
                      <td className="num" data-label="Mean image position">x {pct(activity.ellipse.centroid_x_pct)} · y {pct(activity.ellipse.centroid_y_pct)}</td>
                      <td className="num" data-label="Area · image %">{num(activity.ellipse.area_image_pct, 2)}</td>
                      <td className="num" data-label="Radii · px">{num(activity.ellipse.major_radius_px)} / {num(activity.ellipse.minor_radius_px)}</td>
                    </tr>
                  ))}
                </tbody>
              </DataTable>
              <Pager page={page} pages={pages} count={activities.length} noun="activity tracks" onPage={setActivityPage} />
            </> : <p>No track has enough supported observations for an activity ellipse.</p>}
          </Disclosure>
        </div>}

      <Disclosure summary="How this is measured" meta="Method and limits">
        <p>
          Adapted from <a href="https://www.kaggle.com/code/chaozhuang/soccertrack-collective-dynamics-analysis" target="_blank" rel="noreferrer">Chao Zhuang’s SoccerTrack: Collective Dynamics Analysis</a>, version {data?.reference.version ?? 21}.
          Group centres and sample-covariance ellipses describe visible spatial dispersion. Characteristic area (shown as “spread”) uses the product of the ellipse’s one-standard-deviation radii.
        </p>
        <p>Observed footpoints replace annotated box centres; missing observations remain missing. Named tactical formations are not classified by this method.</p>
        {data?.formation.reason && <p><strong>Formation status, in the analysis's wording: </strong>{data.formation.reason}</p>}
        <p>In the diagram, the solid outline joins the outermost observed players (a convex hull), the dashed outline is a one-standard-deviation (1σ) covariance ellipse of the observed footpoints, and the cross is their mean. The second group is drawn only when its snapshot is from the same moment.</p>
        <p>With footage on, the background is the frame at the snapshot's timestamp, taken from the original upload when it is stored and playable in this browser, otherwise from the annotated video. Ring size grows towards the bottom of the image as a drawing aid; it is not a measured player size.</p>
        <p>The diagram is one actual snapshot near the middle of the second. Summary values average supported observations. Camera motion, perspective and missing players change this shape.</p>
        <p>Alignment (“moving together”) describes image motion, not proven tactical coordination. The camera view is not a pitch map, occupied territory or square metres.</p>
        {data?.pitch && (data.pitch.status === "estimated" ? <>
          <p><strong>Pitch map. </strong>A model marks known pitch markings in a frame, and the frame is mapped onto the pitch from them. {data.pitch.frames_calibrated} of {data.pitch.frames_tried} tried frames passed the checks, and a frame that fails has no pitch positions. {data.pitch.frames_stabilised ? `${data.pitch.frames_stabilised} of those were then steadied: the camera is taken to stand in one place for the whole clip, and its turn and zoom are smoothed over about half a second. The rest keep their own frame-by-frame mapping.` : "Each frame is mapped on its own."} When several frames in a second qualify, the snapshot is the calibrated frame nearest the middle of the second.</p>
          {data.pitch.limitations?.length ? <ul>{data.pitch.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul> : null}
        </> : <p><strong>Pitch map unavailable. </strong>{data.pitch.reason}</p>)}
        {motionData && <>
          <p><strong>Heat map, distance and speed. </strong>Each tracked player’s feet are placed on the pitch in every mapped frame. Positions are averaged over {motionData.parameters.position_average_seconds} s and distance is added up in {motionData.parameters.hop_seconds} s steps, so box jitter is not counted as running. A step faster than {motionData.parameters.max_speed_m_s} m/s is treated as an error and left out. A gap in tracking or mapping ends the stretch; nothing is filled in across it.</p>
          <ul>{motionData.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul>
        </>}
        {data?.limitations.length ? <ul>{data.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul> : null}
      </Disclosure>
    </Panel>
  );
}
