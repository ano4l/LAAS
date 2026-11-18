import { useCallback, useId, useState } from "react";
import { ArrowDownToLine, ArrowRight, Play } from "lucide-react";
import type { TemporalSequences } from "./TemporalPossession";
import type { CollectiveDynamics } from "./SoccerTrackShape";
import { num, pct, range, secs, sentence, words } from "./format";
import { TeamName, useKits } from "./kits";
import { usePlayback, usePlaybackValue } from "./playback";
import { DataTable, Disclosure, EmptyState, Field, Meter, Pager, Panel, WatchButton, cssVars } from "./ui";

type PathPoint = { timestamp_seconds: number; x_pct: number; y_pct: number; scene_id: number; segment_id: number };
type ShapeWindow = { start_seconds: number; end_seconds: number; supported_frames: number; supported_seconds: number; mean_visible_players: number; centroid_x_pct: number | null; centroid_y_pct: number | null; width_pct: number | null; depth_pct: number | null };
type Kit = { kit: string; scene_id: number; player_observations: number; visible_frames: number; visible_seconds: number; image_thirds_pct: (number | null)[]; image_occupancy_grid: number[][]; shape_timeline: ShapeWindow[] };
type Spell = { id: string; scene_id: number; kit: string | null; track_id: string; start_seconds: number; end_seconds: number; observed_seconds: number; observed_frames: number; mean_ball_confidence: number; mean_foot_distance_heights: number; image_start: { x_pct: number; y_pct: number }; image_end: { x_pct: number; y_pct: number }; review_status: string };
export type Candidate = { id: string; type: string; timestamp_seconds: number; start_seconds: number; end_seconds: number; evidence_start_seconds: number; evidence_end_seconds: number; scene_id: number; from_track: string | null; to_track: string | null; from_kit: string | null; to_kit: string | null; image_separation_px?: number; supported_control_seconds?: number; observed_bridge_frames?: number; review_status: string; evidence: string; from_team?: string; to_team?: string; ball_support_pct?: number | null; missing_ball_seconds?: number; max_missing_ball_gap_seconds?: number; receiver_support?: string; quality_flags?: string[]; evidence_level?: string; possession_included?: boolean;
  /** Set on rule-based shots, tackles, duels and dribbles, and names the Review event type a person would record. */
  suggested_review_type?: string; basis?: string; ball_speed_heights_per_second?: number; goalkeeper_track?: string; estimated_distance_m?: number;
  players_closest_heights?: number; ball_travel_heights?: number; contested_seconds?: number; next_touch_track?: string | null; next_touch_team?: string | null;
  opponent_closest_heights?: number; carried_seconds?: number };
/** What the Review form is started with when a person chooses to draft a candidate. */
export type CandidateDraft = { timestamp_seconds: number; type: string; notes: string };
export type VideoInsights = {
  schema_version: string; source: string; review_status: string; coordinate_system: string;
  source_video?: { filename?: string; sha256?: string; width?: number; height?: number; fps?: number; frame_count?: number };
  sample_fps?: number;
  source_duration_seconds: number; analyzed_frames: number; analyzed_seconds: number;
  quality: { selected_ball_seconds: number; selected_ball_coverage_pct: number | null; grouped_player_observations: number; player_observations: number; kit_assignment_coverage_pct: number | null; accuracy_evaluated: boolean };
  control: { estimated_control_seconds: number; unknown_seconds: number; coverage_pct: number | null; grouped_control_seconds: number; grouped_coverage_pct: number | null; spells: Spell[]; by_kit: { kit: string; estimated_seconds: number; share_of_known_control_pct: number | null; clip_share_pct: number | null }[] };
  candidates: Candidate[];
  timeline: { start_seconds: number; end_seconds: number; analyzed_frames: number; ball_observed_seconds: number; ball_coverage_pct: number | null; estimated_control_seconds: number; unknown_seconds: number; kit_control_seconds: Record<string, number> }[];
  kits: Kit[];
  ball: { observed_seconds: number; image_thirds_pct: (number | null)[]; image_occupancy_grid: number[][]; path: PathPoint[]; path_observations: number; path_display_stride: number; linked_motion_steps: number; excluded_motion_steps: number; linked_image_displacement_px: number };
  findings: { kind: string; text: string }[];
  bookmarks?: { id: string; type: string; label: string; start_seconds: number; end_seconds: number; detail: string }[];
  parameters: Record<string, string | number>; definitions: Record<string, string>;
  sequences?: TemporalSequences;
  collective_dynamics?: CollectiveDynamics;
};

const THIRDS = ["Left third of the image", "Centre third of the image", "Right third of the image"];
const CANDIDATE_NAMES: Record<string, [string, string]> = {
  possible_pass: ["Possible pass", "Possible passes"], possible_turnover: ["Possible turnover", "Possible turnovers"], possible_shot: ["Possible shot", "Possible shots"],
  possible_tackle: ["Possible tackle", "Possible tackles"], possible_duel: ["Possible duel", "Possible duels"], possible_dribble: ["Possible dribble", "Possible dribbles"],
};
export const candidateName = (type: string) => CANDIDATE_NAMES[type]?.[0] ?? sentence(type);
/** Who did what, in words that fit the kind of event. `name` turns a track into what the reader should see. */
export function candidatePlayers(event: Candidate, name: (track: string) => string = track => `track ${track}`) {
  const a = event.from_track ? name(event.from_track) : "an untracked player", b = event.to_track ? name(event.to_track) : null;
  switch (event.type) {
    case "possible_shot": return b ? `${a} · next seen with ${b}` : a;
    case "possible_tackle": return `${b ?? "an untracked player"} took the ball from ${a}`;
    case "possible_duel": return `${a} against ${b ?? "an untracked player"}`;
    case "possible_dribble": return `${a} past ${b ?? "an untracked player"}`;
    default: return `${a} → ${b ?? "an untracked player"}`;
  }
}
function candidateDetail(event: Candidate) {
  switch (event.type) {
    case "possible_shot": return `Ball left at about ${num(event.ball_speed_heights_per_second)} player heights a second · ${event.basis === "towards_goalkeeper" ? `heading at the goalkeeper (track ${event.goalkeeper_track})` : "heading at a goal on the estimated pitch map"}${event.estimated_distance_m != null ? ` · about ${num(event.estimated_distance_m, 0)} m out on the pitch map` : ""}`;
    case "possible_tackle": return `Players ${num(event.players_closest_heights)} player heights apart at their closest · ball moved ${num(event.ball_travel_heights)} heights between them`;
    case "possible_duel": return `Ball about equally close to both for ${secs(event.contested_seconds ?? 0, 2)} · ${event.next_touch_track ? `next controlled touch: track ${event.next_touch_track}` : "no controlled touch soon after"}`;
    case "possible_dribble": return `Ball kept for ${secs(event.carried_seconds ?? 0, 2)} · opponent came within ${num(event.opponent_closest_heights)} player heights`;
    default: return null;
  }
}

/** Insights JSON and CSV for this run. The export format is unchanged; only the buttons are restyled. */
export function VideoInsightsDownloads({ insights, runId }: { insights: VideoInsights; runId: string }) {
  function download(format: "json" | "csv") {
    const columns = ["record_type", "source_run_id", "source_duration_seconds", "sequence_schema_version", "context_revision", "mapping_status", "scene_id", "segment_id", "team", "from_team", "to_team", "kit", "track_id", "start_seconds", "end_seconds", "interval_seconds", "state", "support", "x_pct", "y_pct", "width_pct", "depth_pct", "supported_seconds", "unknown_seconds", "contested_seconds", "dead_ball_seconds", "classified_non_dead_seconds", "observed_seconds", "observed_frames", "share_of_known_group_control_pct", "clip_share_pct", "share_of_known_sequence_pct", "share_of_non_dead_pct", "observed_control_seconds", "flight_and_gap_seconds", "bounded_missing_ball_seconds", "ball_support_pct", "missing_ball_seconds", "max_missing_ball_gap_seconds", "receiver_support", "quality_flags", "evidence_level", "possession_included", "event_type", "from_track", "to_track", "review_status"];
    columns.push("source_filename", "source_sha256", "shape_schema_version", "shape_coordinate_system", "formation_status", "formation_label", "mean_visible_players", "area_image_pct", "major_radius_px", "minor_radius_px", "angle_degrees", "rms_spread_px", "point_count", "motion_alignment", "alignment_supported_seconds", "snapshot_seconds");
    const rows: (string | number | null)[][] = [];
    const add = (type: string, values: Record<string, string | number | null>) => rows.push(columns.map(key => key === "record_type" ? type : key === "source_run_id" ? runId : key === "source_duration_seconds" ? insights.source_duration_seconds : key === "source_filename" ? insights.source_video?.filename ?? null : key === "source_sha256" ? insights.source_video?.sha256 ?? null : key === "shape_schema_version" ? insights.collective_dynamics?.schema_version ?? null : key === "shape_coordinate_system" ? insights.collective_dynamics?.coordinate_system ?? null : key === "sequence_schema_version" ? insights.sequences?.schema_version ?? null : key === "context_revision" ? insights.sequences?.context_revision ?? null : key === "mapping_status" ? insights.sequences?.mapping_status ?? null : key === "review_status" ? "unreviewed" : values[key] ?? null));
    add("strict_near_foot_control_summary", { start_seconds: 0, end_seconds: insights.source_duration_seconds, supported_seconds: insights.control.estimated_control_seconds, unknown_seconds: insights.control.unknown_seconds });
    const sequence = insights.sequences;
    if (sequence) {
      add("temporal_possession_summary", { start_seconds: 0, end_seconds: sequence.source_duration_seconds, supported_seconds: sequence.known_seconds, unknown_seconds: sequence.unknown_seconds, contested_seconds: sequence.contested_seconds, dead_ball_seconds: sequence.dead_ball_seconds, classified_non_dead_seconds: sequence.classified_non_dead_seconds });
      sequence.by_team.forEach(team => add("temporal_team_estimate", { team: team.team, supported_seconds: team.estimated_seconds, share_of_known_sequence_pct: team.share_of_known_pct, share_of_non_dead_pct: team.share_of_non_dead_pct, observed_control_seconds: team.observed_control_seconds, flight_and_gap_seconds: team.flight_and_gap_seconds, bounded_missing_ball_seconds: team.bounded_missing_ball_seconds }));
      sequence.intervals.forEach(interval => add("temporal_source_interval", { start_seconds: interval.start_seconds, end_seconds: interval.end_seconds, state: interval.state, team: interval.team, kit: interval.kit, track_id: interval.track_id, support: interval.support, interval_seconds: interval.end_seconds - interval.start_seconds }));
      sequence.anchors.forEach(anchor => add("temporal_control_anchor", { start_seconds: anchor.start_seconds, end_seconds: anchor.end_seconds, scene_id: anchor.scene_id, team: anchor.team, kit: anchor.kit, track_id: anchor.track_id, observed_seconds: anchor.observed_seconds, observed_frames: anchor.observed_frames, support: anchor.support_type }));
    }
    const dynamics = insights.collective_dynamics;
    if (dynamics) {
      add("collective_formation_status", { formation_status: dynamics.formation.status, formation_label: dynamics.formation.label });
      dynamics.teams.forEach(team => {
        add("collective_group_summary", { kit: team.kit, scene_id: team.scene_id, supported_seconds: team.supported_seconds, clip_share_pct: team.source_coverage_pct, mean_visible_players: team.mean_visible_players, area_image_pct: team.mean_area_image_pct, motion_alignment: team.mean_alignment });
        team.timeline.forEach(window => {
          const ellipse = window.snapshot.ellipse;
          add("collective_source_second", { kit: team.kit, scene_id: team.scene_id, start_seconds: window.start_seconds, end_seconds: window.end_seconds, supported_seconds: window.supported_seconds, observed_frames: window.observed_frames, mean_visible_players: window.mean_visible_players, area_image_pct: window.mean_area_image_pct, x_pct: window.centroid_x_pct, y_pct: window.centroid_y_pct, motion_alignment: window.motion_alignment, alignment_supported_seconds: window.alignment_supported_seconds, snapshot_seconds: window.snapshot.timestamp_seconds });
          add("collective_snapshot_ellipse", { kit: team.kit, scene_id: team.scene_id, snapshot_seconds: window.snapshot.timestamp_seconds, x_pct: ellipse.centroid_x_pct, y_pct: ellipse.centroid_y_pct, area_image_pct: ellipse.area_image_pct, major_radius_px: ellipse.major_radius_px, minor_radius_px: ellipse.minor_radius_px, angle_degrees: ellipse.angle_degrees, rms_spread_px: ellipse.rms_spread_px, point_count: ellipse.point_count });
          window.snapshot.players.forEach(player => add("collective_snapshot_player", { kit: team.kit, scene_id: team.scene_id, snapshot_seconds: window.snapshot.timestamp_seconds, track_id: player.track_id, x_pct: player.x_pct, y_pct: player.y_pct }));
        });
        team.activity_ellipses.forEach(activity => add("collective_track_activity", { kit: team.kit, scene_id: team.scene_id, track_id: activity.track_id, observed_seconds: activity.observed_seconds, observed_frames: activity.observed_frames, x_pct: activity.ellipse.centroid_x_pct, y_pct: activity.ellipse.centroid_y_pct, area_image_pct: activity.ellipse.area_image_pct, major_radius_px: activity.ellipse.major_radius_px, minor_radius_px: activity.ellipse.minor_radius_px, angle_degrees: activity.ellipse.angle_degrees, rms_spread_px: activity.ellipse.rms_spread_px, point_count: activity.ellipse.point_count }));
      });
    }
    add("full_clip_ball_observation_summary", { observed_seconds: insights.ball.observed_seconds, clip_share_pct: insights.quality.selected_ball_coverage_pct });
    insights.control.by_kit.forEach(kit => add("group_assigned_control_summary", { kit: kit.kit, supported_seconds: kit.estimated_seconds, share_of_known_group_control_pct: kit.share_of_known_control_pct, clip_share_pct: kit.clip_share_pct }));
    insights.ball.path.forEach(point => add("sampled_ball_path_point", { scene_id: point.scene_id, segment_id: point.segment_id, start_seconds: point.timestamp_seconds, x_pct: point.x_pct, y_pct: point.y_pct }));
    insights.kits.forEach(kit => kit.shape_timeline.forEach(window => add("kit_image_shape_window", { scene_id: kit.scene_id, kit: kit.kit, start_seconds: window.start_seconds, end_seconds: window.end_seconds, x_pct: window.centroid_x_pct, y_pct: window.centroid_y_pct, width_pct: window.width_pct, depth_pct: window.depth_pct, supported_seconds: window.supported_seconds, observed_frames: window.supported_frames })));
    insights.control.spells.forEach(spell => add("estimated_control_spell", { scene_id: spell.scene_id, kit: spell.kit, track_id: spell.track_id, start_seconds: spell.start_seconds, end_seconds: spell.end_seconds, supported_seconds: spell.observed_seconds, observed_frames: spell.observed_frames }));
    insights.candidates.forEach(event => add("possible_event_candidate", { scene_id: event.scene_id, kit: `${event.from_kit ?? "unknown"} → ${event.to_kit ?? "unknown"}`, from_team: event.from_team ?? null, to_team: event.to_team ?? null, start_seconds: event.start_seconds, end_seconds: event.end_seconds, event_type: event.type, from_track: event.from_track, to_track: event.to_track, supported_seconds: event.supported_control_seconds ?? null, observed_frames: event.observed_bridge_frames ?? null, ball_support_pct: event.ball_support_pct ?? null, missing_ball_seconds: event.missing_ball_seconds ?? null, max_missing_ball_gap_seconds: event.max_missing_ball_gap_seconds ?? null, receiver_support: event.receiver_support ?? null, quality_flags: event.quality_flags?.join("; ") ?? null, evidence_level: event.evidence_level ?? null, possession_included: event.possession_included === undefined ? null : String(event.possession_included) }));
    const cell = (value: string | number | null) => { const text = value == null ? "" : String(value); const safe = typeof value === "string" && /^(\s*[=+\-@]|[\t\r])/.test(text) ? `'${text}` : text; return `"${safe.replaceAll('"', '""')}"`; };
    const contents = format === "json" ? JSON.stringify({ run_id: runId, video_insights: insights }, null, 2) : [columns, ...rows].map(row => row.map(cell).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([contents], { type: format === "json" ? "application/json" : "text/csv;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = `${runId}-unreviewed-video-insights.${format}`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <div className="row">
      <button type="button" className="btn btn--secondary btn--sm" onClick={() => download("json")}><ArrowDownToLine size={14} aria-hidden="true" /> Insights JSON</button>
      <button type="button" className="btn btn--secondary btn--sm" onClick={() => download("csv")}><ArrowDownToLine size={14} aria-hidden="true" /> Insights CSV</button>
    </div>
  );
}

/* ---------- Ball journey ---------- */

export function BallJourney({ data }: { data: VideoInsights }) {
  const scenes = [...new Set(data.ball.path.map(point => point.scene_id))].sort((a, b) => a - b);
  const [selectedScene, setSelectedScene] = useState(scenes[0] ?? 0);
  const [pointIndex, setPointIndex] = useState(0);
  const [follow, setFollow] = useState(true);
  const points = data.ball.path.filter(point => point.scene_id === selectedScene);
  const segments = new Map<number, PathPoint[]>();
  points.forEach(point => segments.set(point.segment_id, [...(segments.get(point.segment_id) ?? []), point]));

  // While following, the marked point is the latest observed position at or before the playhead.
  const path = data.ball.path;
  const followed = usePlaybackValue(useCallback((seconds: number) => {
    let found = -1;
    for (let index = 0; index < path.length; index++) {
      if (path[index].timestamp_seconds > seconds + 1e-6) break;
      if (path[index].scene_id === selectedScene) found++;
    }
    return found;
  }, [path, selectedScene]));
  const current = Math.min(follow ? Math.max(0, followed) : pointIndex, Math.max(0, points.length - 1));
  const point = points[current];
  const aspect = data.source_video?.width && data.source_video?.height ? `${data.source_video.width} / ${data.source_video.height}` : "16 / 9";

  return (
    <Panel
      title="Ball journey through the image"
      subtitle="Observed positions only. Separate path segments stop at gaps, cuts and jumps; no missing positions are filled."
      aside={scenes.length > 1 && (
        <Field label="Ball path scene" className="field--inline">
          <select value={selectedScene} onChange={event => { setSelectedScene(Number(event.target.value)); setPointIndex(0); }}>
            {scenes.map(scene => <option key={scene} value={scene}>Scene {scene + 1}</option>)}
          </select>
        </Field>
      )}
    >
      <div className="chartpair">
        <div className="stack stack--sm">
          <figure className="camera" style={{ aspectRatio: aspect }}>
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`Observed ball path in scene ${selectedScene + 1}. X and Y are camera-image percentages. Inspect exact points with the time control below.`}>
              <g className="camera__grid"><line x1="33.333" x2="33.333" y1="0" y2="100" /><line x1="66.667" x2="66.667" y1="0" y2="100" /><line x1="0" x2="100" y1="50" y2="50" /></g>
              {[...segments].map(([segment, values]) => values.length > 1
                ? <polyline className="camera__path" key={segment} points={values.map(p => `${p.x_pct},${p.y_pct}`).join(" ")} />
                : <circle className="camera__dot" key={segment} cx={values[0].x_pct} cy={values[0].y_pct} r=".6" />)}
            </svg>
            {point && <i className="camera__ball" style={{ left: `${point.x_pct}%`, top: `${point.y_pct}%` }} aria-hidden="true" />}
            <figcaption className="camera__axis"><span>Left · x 0%</span><span>Camera image, top is y 0%</span><span>x 100% · right</span></figcaption>
          </figure>

          {point ? (
            <div className="scrub">
              <Field label="Observed ball position">
                <input type="range" min="0" max={Math.max(0, points.length - 1)} step="1" value={current} onChange={event => { setFollow(false); setPointIndex(Number(event.target.value)); }} />
              </Field>
              <div className="row row--between" role="status">
                <span><strong className="mono">{secs(point.timestamp_seconds, 2)}</strong> <span className="note">x {pct(point.x_pct)} · y {pct(point.y_pct)} · segment {point.segment_id}</span></span>
                <WatchButton at={point.timestamp_seconds} label="Watch ball position" />
              </div>
              <label className="check"><input type="checkbox" checked={follow} onChange={event => { setFollow(event.target.checked); setPointIndex(current); }} />Follow the video</label>
            </div>
          ) : <EmptyState title="No ball path in this scene">No eligible observed ball positions. Nothing is drawn in their place.</EmptyState>}
          <p className="note">Scene {selectedScene + 1}: {points.length} displayed points across {segments.size} separate segments. Displayed points are sampled; the whole-clip totals do not change with the scene.</p>
        </div>

        <div className="stack">
          <div className="stack stack--sm">
            <h4>Where the ball was seen</h4>
            <p className="note">Share of the {secs(data.ball.observed_seconds)} of observed ball time, across all scenes.</p>
            <Thirds values={data.ball.image_thirds_pct} />
            <p className="note">Thirds of the camera image, not pitch zones or attack direction.</p>
          </div>
          {points.length > 0 && (
            <Disclosure summary="Ball path data">
              <DataTable label="Selected scene sampled ball positions">
                <thead><tr><th scope="col">Time</th><th scope="col" className="num">X image %</th><th scope="col" className="num">Y image %</th><th scope="col" className="num">Segment</th></tr></thead>
                <tbody>
                  {points.slice(Math.max(0, current - 5), current + 6).map(p => (
                    <tr key={`${p.timestamp_seconds}-${p.segment_id}`} className={p === point ? "is-now" : undefined}>
                      <th scope="row" className="timecell">{secs(p.timestamp_seconds, 2)}</th><td className="num">{pct(p.x_pct, 2)}</td><td className="num">{pct(p.y_pct, 2)}</td><td className="num">{p.segment_id}</td>
                    </tr>
                  ))}
                </tbody>
              </DataTable>
              <p>Up to 11 points around the marked time. The full sampled path is in Insights JSON and CSV.</p>
            </Disclosure>
          )}
        </div>
      </div>
    </Panel>
  );
}

export function Thirds({ values, tone }: { values: (number | null)[]; tone?: "a" | "b" | "x" | null }) {
  return (
    <ul className="thirds">
      {THIRDS.map((label, index) => (
        <li key={label}>
          <span>{label}</span>
          <strong className="mono">{pct(values[index])}</strong>
          <Meter value={values[index]} tone={tone} />
        </li>
      ))}
    </ul>
  );
}

/* ---------- Kit-group shape over time ---------- */

export function KitEvolution({ data }: { data: VideoInsights }) {
  const kits = useKits();
  const [kitId, setKitId] = useState(data.kits[0]?.kit ?? "");
  const [measure, setMeasure] = useState<"width_pct" | "depth_pct" | "centroid_x_pct" | "centroid_y_pct">("width_pct");
  const kit = data.kits.find(item => item.kit === kitId);
  const [windowIndex, setWindowIndex] = useState(0);
  const [cell, setCell] = useState({ row: 0, column: 0 });
  const shape = kit?.shape_timeline ?? [];
  const chosen = shape[Math.min(windowIndex, Math.max(0, shape.length - 1))];
  const paths: ShapeWindow[][] = [];
  shape.forEach(window => {
    if (window[measure] == null) return;
    const last = paths[paths.length - 1];
    if (last && Math.abs(last[last.length - 1].end_seconds - window.start_seconds) < .001) last.push(window); else paths.push([window]);
  });
  const totalSupport = shape.reduce((sum, window) => sum + window.supported_seconds, 0);
  const maxCell = Math.max(1, ...(kit?.image_occupancy_grid.flat() ?? []));
  const names = { width_pct: "Image width", depth_pct: "Image depth", centroid_x_pct: "Centre x", centroid_y_pct: "Centre y" };
  const tone = kits.tone(kit?.kit);
  const nowSeconds = usePlaybackValue(useCallback((seconds: number) => Math.round(seconds * 4) / 4, []));
  const x = (window: ShapeWindow) => 100 * ((window.start_seconds + window.end_seconds) / 2) / data.source_duration_seconds;

  return (
    <Panel
      title="Kit-group shape over time"
      subtitle="Image geometry of at least three visible grouped players, in five-second windows. No formation or physical distance is inferred."
      aside={data.kits.length > 0 && (
        <Field label="Kit group" className="field--inline">
          <select value={kitId} onChange={event => { setKitId(event.target.value); setWindowIndex(0); }}>
            {data.kits.map(item => <option value={item.kit} key={item.kit}>{kits.label(item.kit)}</option>)}
          </select>
        </Field>
      )}
    >
      {!kit ? <EmptyState title="No kit groups for this clip">No supported anonymous kit grouping is available. Shape by group is unknown.</EmptyState> : <>
        <p className="row"><TeamName id={kit.kit} /><span className="note">{num(kit.player_observations, 0)} player observations · visible in {num(kit.visible_frames, 0)} frames / {secs(kit.visible_seconds)} · shape support {secs(totalSupport)}</span></p>
        <div className={`chartpair ${tone ? `tone-${tone}` : ""}`}>
          <div className="stack stack--sm">
            <Field label="Shape measure" className="field--fixed">
              <select value={measure} onChange={event => setMeasure(event.target.value as typeof measure)}>
                {Object.entries(names).map(([key, label]) => <option value={key} key={key}>{label} · image %</option>)}
              </select>
            </Field>
            {shape.length > 0 ? <>
              <figure className="linechart">
                <span className="linechart__max">100%</span>
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${names[measure]} across five-second windows for ${kits.label(kit.kit)}. Exact values and support are available below.`}>
                  <line className="linechart__mid" x1="0" x2="100" y1="50" y2="50" />
                  {paths.map((path, index) => path.length > 1 && <polyline key={index} points={path.map(window => `${x(window)},${100 - window[measure]!}`).join(" ")} />)}
                  <line className="linechart__now" x1={100 * nowSeconds / data.source_duration_seconds} x2={100 * nowSeconds / data.source_duration_seconds} y1="0" y2="100" />
                </svg>
                {paths.flat().map(window => <i key={window.start_seconds} className={`linechart__dot ${window === chosen ? "is-selected" : ""}`} style={{ left: `${x(window)}%`, top: `${100 - window[measure]!}%` }} aria-hidden="true" />)}
                <figcaption><span>0% · 0 s</span><span>{secs(data.source_duration_seconds)} · full source time</span></figcaption>
              </figure>
              <Field label="Five-second window">
                <select value={Math.min(windowIndex, shape.length - 1)} onChange={event => setWindowIndex(Number(event.target.value))}>
                  {shape.map((window, index) => <option key={window.start_seconds} value={index}>{range(window.start_seconds, window.end_seconds)}</option>)}
                </select>
              </Field>
              {chosen && (
                <div className="row row--between" role="status">
                  <span><strong>{names[measure]}: {pct(chosen[measure])}</strong><br /><span className="note">Supported by {secs(chosen.supported_seconds, 2)} / {chosen.supported_frames} frames, with {num(chosen.mean_visible_players)} mean grouped visible players.</span></span>
                  <WatchButton at={chosen.start_seconds} label="Watch shape window" />
                </div>
              )}
            </> : <EmptyState title="Shape unknown for this group">Fewer than three grouped players were sufficiently visible.</EmptyState>}
          </div>

          <div className="stack stack--sm">
            <h4>Where this group was seen</h4>
            <div className="heatgrid" role="img" aria-label={`Image occupancy for ${kits.label(kit.kit)}. Brighter cells contain more observations; inspect values with the row and column selectors.`}>
              {kit.image_occupancy_grid.flatMap((row, y) => row.map((count, column) => (
                <span key={`${column}-${y}`} className={cell.row === y && cell.column === column ? "is-selected" : ""} title={`Row ${y + 1}, column ${column + 1}: ${count} observations`} style={cssVars({ "--v": .08 + .92 * count / maxCell })} />
              )))}
            </div>
            <div className="toolbar">
              <Field label="Row"><select aria-label="Kit image row" value={cell.row} onChange={event => setCell({ ...cell, row: Number(event.target.value) })}>{kit.image_occupancy_grid.map((_, row) => <option value={row} key={row}>{row + 1}</option>)}</select></Field>
              <Field label="Column"><select aria-label="Kit image column" value={cell.column} onChange={event => setCell({ ...cell, column: Number(event.target.value) })}>{Array.from({ length: 12 }, (_, col) => <option value={col} key={col}>{col + 1}</option>)}</select></Field>
              <output aria-live="polite">{kit.image_occupancy_grid[cell.row]?.[cell.column] ?? "Unknown"} observations</output>
            </div>
            <Thirds values={kit.image_thirds_pct} tone={tone} />
            <p className="note">Share of {num(kit.player_observations, 0)} grouped foot-point observations in this scene and kit group; not time or pitch territory.</p>
          </div>
        </div>
        {shape.length > 0 && (
          <Disclosure summary="Exact shape windows and support" meta={`${shape.length} windows`}>
            <DataTable label="Anonymous kit image shape and support by window" stack>
              <thead><tr><th scope="col">Window</th><th scope="col" className="num">Centre x / y</th><th scope="col" className="num">Width / depth</th><th scope="col" className="num">Support</th><th scope="col" className="num">Mean grouped players</th></tr></thead>
              <tbody>
                {shape.map(window => (
                  <tr key={window.start_seconds} className={nowSeconds >= window.start_seconds && nowSeconds < window.end_seconds ? "is-now" : undefined}>
                    <th scope="row" className="timecell">{range(window.start_seconds, window.end_seconds)}</th>
                    <td className="num" data-label="Centre x / y">{pct(window.centroid_x_pct, 2)} / {pct(window.centroid_y_pct, 2)}</td>
                    <td className="num" data-label="Width / depth">{pct(window.width_pct, 2)} / {pct(window.depth_pct, 2)}</td>
                    <td className="num" data-label="Support">{secs(window.supported_seconds, 2)} / {window.supported_frames} frames</td>
                    <td className="num" data-label="Mean grouped players">{num(window.mean_visible_players, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </DataTable>
          </Disclosure>
        )}
      </>}
    </Panel>
  );
}

/* ---------- Strict near-foot observations ---------- */

export function ControlEstimates({ data }: { data: VideoInsights }) {
  const kits = useKits();
  const playback = usePlayback();
  const [page, setPage] = useState(0);
  const [kit, setKit] = useState("all");
  const spells = data.control.spells.filter(spell => kit === "all" || (kit === "unknown" ? !spell.kit : spell.kit === kit));
  const pages = Math.max(1, Math.ceil(spells.length / 6));
  const current = Math.min(page, pages - 1);
  const unknownKitSeconds = Math.max(0, data.control.estimated_control_seconds - data.control.grouped_control_seconds);

  return (
    <div className="stack">
      <p>A stricter, separate rule: the ball clearly nearest one player's feet for a sustained run of frames. It is <strong>not</strong> match possession, and unknown intervals stay unassigned.</p>
      <p className="facts">
        <span>Supported<b>{secs(data.control.estimated_control_seconds)}</b></span>
        <span>Unknown<b>{secs(data.control.unknown_seconds)}</b></span>
        <span>Full source<b>{secs(data.source_duration_seconds)}</b></span>
      </p>
      {data.control.estimated_control_seconds > 0 ? <>
        <p className="note note--boxed">
          <strong>{secs(data.control.grouped_control_seconds)} assigned to kit groups</strong> · {pct(data.control.grouped_coverage_pct)} of the full clip · {secs(unknownKitSeconds)} of estimated control with an unknown kit · {secs(data.control.unknown_seconds)} with no supported controller.
        </p>
        {data.control.by_kit.length > 0 ? (
          <ul className="teamrows">
            {data.control.by_kit.map(group => (
              <li key={group.kit} className="teamrow">
                <div className="teamrow__name"><TeamName id={group.kit} /></div>
                <div className="teamrow__figure"><strong>{secs(group.estimated_seconds)}</strong><span>{pct(group.share_of_known_control_pct)} of known grouped control · {pct(group.clip_share_pct)} of the full clip</span></div>
                <div className="teamrow__bar"><Meter value={group.share_of_known_control_pct} tone={kits.tone(group.kit)} /></div>
              </li>
            ))}
          </ul>
        ) : <p className="note">Supported control has no reliable kit assignment. Group shares are unknown.</p>}
        <Field label="Filter supported spells" className="field--fixed">
          <select value={kit} onChange={event => { setKit(event.target.value); setPage(0); }}>
            <option value="all">All supported spells</option>
            <option value="unknown">Unknown kit only</option>
            {data.control.by_kit.map(group => <option value={group.kit} key={group.kit}>{kits.label(group.kit)}</option>)}
          </select>
        </Field>
        <ul className="rowlist">
          {spells.slice(current * 6, (current + 1) * 6).map(spell => (
            <li key={spell.id}>
              <button type="button" className="rowlink" onClick={() => playback.seek(spell.start_seconds)}>
                <Play size={13} fill="currentColor" aria-hidden="true" />
                <span><strong><span className="mono">{range(spell.start_seconds, spell.end_seconds)}</span> · {kits.label(spell.kit, "Unknown kit")}</strong><small>Track {spell.track_id} · {secs(spell.observed_seconds, 2)} / {spell.observed_frames} observed frames · scene {spell.scene_id + 1}</small></span>
                <ArrowRight size={15} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ul>
        {!spells.length && <p className="note" role="status">No supported spells match this kit selection.</p>}
        <Pager page={current} pages={pages} count={spells.length} noun="spells" onPage={setPage} />
      </> : <EmptyState title="Near-foot control is unknown throughout this clip">No run of consecutive frames met the minimum support rule, so no kit shares or possession split can be established.</EmptyState>}
      <p className="note">
        Kit shares divide only the {secs(data.control.grouped_control_seconds, 2)} of group-assigned control. Support requires at least {secs(Number(data.parameters.min_control_seconds ?? .24), 2)} and{" "}
        {num(Number(data.parameters.min_control_frames ?? 3), 0)} observed frames. This is an image-based estimate for review.
      </p>
    </div>
  );
}

/* ---------- To review: possible events ---------- */

export function CandidateQueue({ data, onDraft }: { data: VideoInsights; onDraft?: (draft: CandidateDraft) => void }) {
  const kits = useKits();
  const id = useId();
  const [type, setType] = useState("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const matches = data.candidates.filter(event => (type === "all" || event.type === type) && `${event.from_track} ${event.to_track} ${event.from_kit} ${event.to_kit} ${event.evidence} ${event.from_kit ? kits.label(event.from_kit) : ""} ${event.to_kit ? kits.label(event.to_kit) : ""} ${candidateName(event.type)}`.toLowerCase().includes(search.trim().toLowerCase()));
  const types = Object.keys(CANDIDATE_NAMES).filter(key => data.candidates.some(event => event.type === key));
  const ruleBased = data.candidates.some(event => event.evidence_level === "rule_based_review_only");
  const draft = (event: Candidate) => onDraft?.({
    timestamp_seconds: event.timestamp_seconds, type: event.suggested_review_type ?? "pass",
    notes: `Started from the automatic candidate ${event.id} (${candidateName(event.type).toLowerCase()}): ${candidatePlayers(event)}${event.from_kit ? ` · ${kits.label(event.from_kit)}` : ""}${event.to_kit && event.to_kit !== event.from_kit ? ` and ${kits.label(event.to_kit)}` : ""}.`,
  });
  const pages = Math.max(1, Math.ceil(matches.length / 6));
  const current = Math.min(page, pages - 1);

  return (
    <Panel
      title="To review: possible events"
      subtitle="Possible passes, turnovers, shots, tackles, duels and dribbles, for a person to check against the footage. Nothing here is counted or approved automatically."
      aside={<span className="chip chip--plain" title="Technical name: evidence queue · not approved events">{data.candidates.length} to review · not counted</span>}
    >
      <div className="toolbar">
        <Field label="Type" className="field--fixed">
          <select value={type} onChange={event => { setType(event.target.value); setPage(0); }}>
            <option value="all">All possible events</option>
            {types.map(key => <option key={key} value={key}>{CANDIDATE_NAMES[key][1]}</option>)}
          </select>
        </Field>
        <Field label="Find a possible event">
          <input type="search" value={search} onChange={event => { setSearch(event.target.value); setPage(0); }} placeholder="Track or kit group" />
        </Field>
      </div>
      {ruleBased && <p className="note note--boxed">Shots, tackles, duels and dribbles come from simple rules over the tracked ball and players. They have not been checked against labelled events, they miss most real ones, and some will be wrong. Watch each one before recording it.</p>}
      {matches.length ? (
        <ul className="cards" aria-labelledby={id}>
          {matches.slice(current * 6, (current + 1) * 6).map(event => (
            <li key={event.id} className={`card ${event.evidence_level === "partial_tracking_review_only" ? "card--partial" : ""}`}>
              <header className="row row--between">
                <span className="chip chip--unknown"><i className="chip__swatch" aria-hidden="true" />{candidateName(event.type)} · unreviewed</span>
                <strong className="mono">{secs(event.timestamp_seconds, 2)}</strong>
              </header>
              {event.evidence_level === "partial_tracking_review_only" && <p className="note note--boxed">Partial tracking: a review-only proposal. Excluded from possession estimates.</p>}
              <p><strong>{sentence(candidatePlayers(event))}</strong>{event.from_kit && <> · <TeamName id={event.from_kit} /></>}{event.to_kit && <> → <TeamName id={event.to_kit} /></>}</p>
              {event.supported_control_seconds !== undefined
                ? <p className="note">{secs(event.supported_control_seconds, 2)} supported control at both ends · {event.observed_bridge_frames} observed bridge frames · scene {event.scene_id + 1}</p>
                : <p className="note">{candidateDetail(event)} · scene {event.scene_id + 1}</p>}
              {event.from_team && event.supported_control_seconds !== undefined && <p className="note">Estimated group or team: {kits.label(event.from_team)} → {kits.label(event.to_team, "Unknown")}</p>}
              {event.ball_support_pct !== undefined && (
                <p className="note">
                  Ball seen for {pct(event.ball_support_pct)} of it · ball missing for {secs(event.missing_ball_seconds, 2)}
                  {event.max_missing_ball_gap_seconds !== undefined ? ` (longest gap ${secs(event.max_missing_ball_gap_seconds, 2)})` : ""} · receiver: {event.receiver_support ? words(event.receiver_support) : "Unknown"}
                </p>
              )}
              {Boolean(event.quality_flags?.length) && <p className="note">Review flags: {event.quality_flags!.map(words).join(" · ")}</p>}
              <p className="note">{event.evidence}</p>
              <div className="row">
                <WatchButton at={event.evidence_start_seconds} label={`Review evidence ${range(event.evidence_start_seconds, event.evidence_end_seconds)}`} />
                {onDraft && event.suggested_review_type && <button type="button" className="btn btn--secondary btn--sm" title="Opens the Review form filled in with this time and event type. Nothing is saved or approved until you do it there." onClick={() => draft(event)}>Draft under Review</button>}
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <EmptyState title={data.candidates.length ? "No possible events match these filters" : "Nothing to review here"}>
          {data.candidates.length ? "Try another type or search." : "No possible events have sufficient supporting evidence. That does not mean none happened: event totals remain unknown. Inspect the footage and add verified annotations under Review."}
        </EmptyState>
      )}
      <span id={id} className="sr-only">Possible events to review</span>
      <Pager page={current} pages={pages} count={matches.length} noun="possible events" onPage={setPage} />
    </Panel>
  );
}

/* ---------- Definitions ---------- */

/** The pipeline's own definitions, optionally limited to some keys. Parameters are listed in full. */
export function InsightDefinitions({ data, only, parameters = false }: { data: VideoInsights; only?: string[]; parameters?: boolean }) {
  const entries = Object.entries(data.definitions).filter(([name]) => !only || only.includes(name));
  return <>
    {entries.map(([name, text]) => <p key={name}><strong>{sentence(name)}: </strong>{text}</p>)}
    {parameters && <div><h5>Parameters</h5><dl className="deflist">{Object.entries(data.parameters).map(([key, value]) => <div key={key}><dt>{words(key)}</dt><dd>{String(value)}</dd></div>)}</dl></div>}
  </>;
}
