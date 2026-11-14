import { useCallback, useId, useState } from "react";
import { num, pct, range, secs } from "./format";
import { TeamName, useKits } from "./kits";
import { usePlaybackValue } from "./playback";
import { DataTable, Disclosure, EmptyState, Field, Pager, Panel, StatTile, WatchButton } from "./ui";

export type ShapeEllipse = { centroid_x_pct: number; centroid_y_pct: number; major_radius_px: number; minor_radius_px: number; angle_degrees: number; area_image_pct: number; rms_spread_px: number; point_count: number };
type Snapshot = { timestamp_seconds: number; players: { track_id: string; x_pct: number; y_pct: number }[]; ellipse: ShapeEllipse };
export type DynamicsWindow = { start_seconds: number; end_seconds: number; supported_seconds: number; observed_frames: number; mean_visible_players: number; mean_area_image_pct: number; centroid_x_pct: number; centroid_y_pct: number; motion_alignment: number | null; alignment_supported_seconds: number; snapshot: Snapshot };
export type CollectiveDynamics = { schema_version: string; reference: { title: string; url: string; version: number }; coordinate_system: string; review_status: string; source_duration_seconds: number; formation: { status: string; label: string | null; reason: string }; teams: { kit: string; scene_id: number; supported_seconds: number; source_coverage_pct: number; mean_visible_players: number; mean_area_image_pct: number; mean_alignment: number | null; timeline: DynamicsWindow[]; activity_ellipses_total?: number; activity_ellipses: { track_id: string; observed_seconds: number; observed_frames: number; ellipse: ShapeEllipse }[] }[]; parameters: Record<string, string | number>; limitations: string[] };

const FORMATION_PLAIN = "Naming a formation such as 4‑3‑3 or 4‑4‑2 needs calibrated pitch coordinates, every outfield player in view and reliable identities. This view shows the shape of the players the camera could see.";

/** Collective shape of each visible kit group, measured in the camera image (after the SoccerTrack notebook). */
export function SoccerTrackShape({ data, imageWidth, imageHeight, duration }: { data?: CollectiveDynamics; imageWidth?: number; imageHeight?: number; duration: number }) {
  const id = useId();
  const kits = useKits();
  const [kitId, setKitId] = useState(data?.teams[0]?.kit ?? "");
  const [second, setSecond] = useState(0);
  const [follow, setFollow] = useState(true);
  const [activityPage, setActivityPage] = useState(0);
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
                <span><TeamName id={team.kit} /> · scene {team.scene_id + 1} · camera frame, x right, y down</span>
                <span className="mono">{snapshot ? `Snapshot ${secs(snapshot.timestamp_seconds, 2)}` : `${range(selectedSecond, Math.min(sourceDuration, selectedSecond + 1), 1)} · no snapshot`}</span>
              </p>
              <div className="camera camera--shape" style={width && height ? { aspectRatio: `${width}/${height}` } : undefined}>
                {snapshot && ellipse && width && height ? (
                  <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={`Observed ${snapshot.players.length}-player shape for ${kits.label(team.kit)} at ${num(snapshot.timestamp_seconds, 2)} seconds. One-standard-deviation covariance ellipse and player footpoints in the camera image; not a pitch map.`}>
                    <defs><clipPath id={`${id}-clip`}><rect width={width} height={height} /></clipPath></defs>
                    <g className="camera__grid" aria-hidden="true"><line x1={width / 2} x2={width / 2} y1="0" y2={height} /><line x1="0" x2={width} y1={height / 2} y2={height / 2} /></g>
                    <g clipPath={`url(#${id}-clip)`}>
                      <ellipse className="shape__ellipse" cx={ellipse.centroid_x_pct * width / 100} cy={ellipse.centroid_y_pct * height / 100} rx={ellipse.major_radius_px} ry={ellipse.minor_radius_px} transform={`rotate(${ellipse.angle_degrees} ${ellipse.centroid_x_pct * width / 100} ${ellipse.centroid_y_pct * height / 100})`} />
                      {snapshot.players.map(player => (
                        <g key={player.track_id}>
                          <circle className="shape__player" cx={player.x_pct * width / 100} cy={player.y_pct * height / 100} r={width * .006} />
                          <text className="shape__id" aria-hidden="true" x={player.x_pct * width / 100 + width * .01} y={player.y_pct * height / 100 - width * .007} fontSize={width * .018}>{player.track_id}</text>
                        </g>
                      ))}
                      <path className="shape__centre" d={`M${ellipse.centroid_x_pct * width / 100 - width * .01},${ellipse.centroid_y_pct * height / 100}h${width * .02}M${ellipse.centroid_x_pct * width / 100},${ellipse.centroid_y_pct * height / 100 - width * .01}v${width * .02}`} />
                    </g>
                  </svg>
                ) : <p className="camera__message" role="status">{snapshot ? "Source image dimensions are unavailable. Inspect the numeric snapshot below." : "No supported group shape in this second. No positions are filled in."}</p>}
              </div>
              <ul className="keyline">
                <li title="Technical name: observed player footpoint"><i className="shape__key shape__key--point" aria-hidden="true" />Where a player stood</li>
                <li title="Technical name: one-standard-deviation (1σ) covariance ellipse"><i className="shape__key shape__key--ellipse" aria-hidden="true" />Typical spread of the group</li>
                <li><i className="shape__key shape__key--centre" aria-hidden="true" />Centre of the group</li>
              </ul>

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
                <dl className="deflist deflist--single">
                  <div><dt>Players in the group</dt><dd>{num(chosen.mean_visible_players, 2)}</dd></div>
                  <div><dt>Spread (characteristic area)</dt><dd>{pct(chosen.mean_area_image_pct, 2)} of image</dd></div>
                  <div><dt>Group centre</dt><dd>x {pct(chosen.centroid_x_pct)} · y {pct(chosen.centroid_y_pct)}</dd></div>
                  <div><dt>Moving together (alignment)</dt><dd>{chosen.motion_alignment == null ? "Unavailable" : num(chosen.motion_alignment, 3)}<small>{secs(chosen.alignment_supported_seconds, 2)} motion support</small></dd></div>
                </dl>
              ) : <p className="note">No mean or alignment is assigned here. The full clip stays in the support denominator.</p>}
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
        <p>In the diagram, the outline is a one-standard-deviation (1σ) covariance ellipse of the observed footpoints, and the cross is their mean.</p>
        <p>The diagram is one actual snapshot near the middle of the second. Summary values average supported observations. Camera motion, perspective and missing players change this shape.</p>
        <p>Alignment (“moving together”) describes image motion, not proven tactical coordination. This is not a pitch map, occupied territory or square metres.</p>
        {data?.limitations.length ? <ul>{data.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul> : null}
      </Disclosure>
    </Panel>
  );
}
