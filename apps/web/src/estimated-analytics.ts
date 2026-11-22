// What the video analysis itself can say for the requested analytics, where no person has approved events yet.
// Every figure here is the model's estimate from the tracked ball and players. It is shown as an estimate,
// kept apart from figures counted from approved events, and it undercounts wherever the ball or a touch was missed.
import type { AnalysisResult } from "./main";

export type Estimate = { lines: string[]; note: string };
type Label = (kit: string | null | undefined, fallback?: string) => string;

const n = (value: number, digits = 0) => value.toLocaleString(undefined, { maximumFractionDigits: digits });
const plural = (count: number, one: string, many = `${one}s`) => `${n(count)} ${count === 1 ? one : many}`;
const BANDS: [string, string][] = [["walking", "Standing or walking"], ["jogging", "Jogging"], ["running", "Running"], ["fast_running", "Fast running"]];
const CANDIDATES: Record<string, [string, string, string]> = {
  "shots-goals": ["possible_shot", "possible shot", "A fast ball heading at a goalkeeper or a goal on the pitch map. Whether it was a shot, and how it ended, is for a person to confirm."],
  tackles: ["possible_tackle", "possible tackle", "The ball changing team at close range. Who won it cleanly, or whether it was a foul, is for a person to confirm."],
  duels: ["possible_duel", "possible duel", "Two opponents about equally close to the ball. The outcome is for a person to confirm."],
  dribbles: ["possible_dribble", "possible dribble", "A player keeping the ball while an opponent came close and fell away. Whether it was a take-on is for a person to confirm."],
  interceptions: ["possible_turnover", "possible interception", "The ball going from one team’s player to the other team’s after travelling. Whether it was an interception or a loose pass is for a person to confirm."],
};
const TOUCH_NOTE = "From touches the model saw: the ball staying at one player’s feet. Two touches in a row by the same team’s players are taken as a pass, and by different teams as a change of possession. Touches the model missed are not counted, so these are low, and a pass is assumed, not seen.";

/** Everything the estimates draw on, worked out once per clip. */
function context(result: AnalysisResult) {
  const insights = result.metrics.video_insights, sequences = insights?.sequences;
  const anchors = [...(sequences?.anchors ?? [])].sort((a, b) => a.start_seconds - b.start_seconds);
  const maxGap = sequences?.retention?.max_gap_seconds ?? 5;
  const touches = new Map<string, number>(), passes = new Map<string, number>(), lost = new Map<string, number>(), won = new Map<string, number>(), byTrack = new Map<string, { team: string | null; touches: number }>();
  const bump = (map: Map<string, number>, key: string) => map.set(key, (map.get(key) ?? 0) + 1);
  anchors.forEach((anchor, index) => {
    const team = anchor.team ?? anchor.kit;
    if (team) bump(touches, team);
    const mine = byTrack.get(anchor.track_id) ?? { team, touches: 0 };
    byTrack.set(anchor.track_id, { team: mine.team ?? team, touches: mine.touches + 1 });
    const next = anchors[index + 1], nextTeam = next ? next.team ?? next.kit : null;
    if (!next || !team || !nextTeam || next.scene_id !== anchor.scene_id || next.track_id === anchor.track_id || next.start_seconds - anchor.end_seconds > maxGap) return;
    if (nextTeam === team) bump(passes, team); else { bump(lost, team); bump(won, nextTeam); }
  });
  const motion = insights?.collective_dynamics?.pitch_motion;
  const pitch = insights?.collective_dynamics?.pitch;
  const moving = motion?.status === "estimated" ? motion : undefined;
  const teams = [...new Set([...touches.keys(), ...(sequences?.by_team ?? []).map(team => team.team), ...(moving?.teams ?? []).map(team => team.kit)])].sort();
  return { insights, sequences, touches, passes, lost, won, byTrack, moving, pitch, teams };
}

const pitchNote = (pitch: ReturnType<typeof context>["pitch"]) =>
  pitch?.line_check?.status === "lines_match" ? "In this clip the pitch lines drawn from the calibration sit on the lines in the video, which is the best case for these figures."
    : pitch?.line_check?.warning ? `${pitch.line_check.warning}` : "The pitch calibration was not checked against the lines in this clip.";

/** The model's estimate for one catalogue entry, or null when the video analysis has nothing to say about it. */
export function estimatedMetric(id: string, result: AnalysisResult, label: Label): Estimate | null {
  const c = context(result);
  if (!c.insights) return null;
  const name = (team: string) => label(team, team);
  if (CANDIDATES[id]) {
    const [type, word, note] = CANDIDATES[id];
    const found = c.insights.candidates.filter(candidate => candidate.type === type);
    if (!found.length) return null;
    return { lines: [`${plural(found.length, word)} proposed by the model, at ${found.slice(0, 6).map(candidate => `${n(candidate.timestamp_seconds, 1)} s`).join(", ")}${found.length > 6 ? " and later" : ""}`], note: `${note} They are listed with footage under Ball.` };
  }
  switch (id) {
    case "passes": {
      if (!c.touches.size) return null;
      return { lines: c.teams.filter(team => c.touches.has(team)).map(team => `${name(team)}: about ${plural(c.passes.get(team) ?? 0, "pass", "passes")} between its own players, from ${plural(c.touches.get(team)!, "touch", "touches")} the model saw`), note: `${TOUCH_NOTE} Completion cannot be estimated: a pass that went astray is not seen as a pass.` };
    }
    case "possession": {
      const s = c.sequences;
      if (!s || !(s.known_seconds > 0)) return null;
      const assumed = new Map((s.retention?.by_team ?? []).map(team => [team.team, team]));
      const lines = s.by_team.map(team => `${name(team.team)}: ${n(team.estimated_seconds, 1)} s supported (${n(team.share_of_non_dead_pct ?? 0, 1)}% of playing time)${assumed.get(team.team) ? ` · ${n(assumed.get(team.team)!.share_of_non_dead_with_retention_pct ?? 0, 1)}% assuming it kept the ball between its own touches` : ""}`);
      c.teams.forEach(team => { if (c.lost.get(team) || c.won.get(team)) lines.push(`${name(team)}: lost the ball to the other team about ${plural(c.lost.get(team) ?? 0, "time")}, won it about ${plural(c.won.get(team) ?? 0, "time")}`); });
      return { lines, note: `Possession is supported for ${n(s.coverage_pct ?? 0, 1)}% of playing time${s.retention?.coverage_with_retention_pct != null ? `, or ${n(s.retention.coverage_with_retention_pct, 1)}% with the assumption` : ""}; the rest is unknown and is not shared out. The intervals are on the timeline and under Possession.` };
    }
    case "home-away": {
      if (c.teams.length < 2) return null;
      const [a, b] = c.teams, retention = new Map((c.sequences?.retention?.by_team ?? []).map(team => [team.team, team.share_with_retention_pct]));
      const motion = new Map((c.moving?.teams ?? []).map(team => [team.kit, team]));
      const pair = (text: string, pick: (team: string) => string | null) => { const x = pick(a), y = pick(b); return x === null && y === null ? null : `${text}: ${x ?? "unknown"} – ${y ?? "unknown"}`; };
      const lines = [
        `${name(a)} – ${name(b)}`,
        pair("Share of known possession, with the assumption", team => retention.get(team) == null ? null : `${n(retention.get(team)!, 1)}%`),
        pair("Touches the model saw", team => c.touches.has(team) ? n(c.touches.get(team)!) : null),
        pair("Passes between own players, about", team => c.touches.has(team) ? n(c.passes.get(team) ?? 0) : null),
        pair("Distance covered by tracked players", team => motion.get(team) ? `${n(motion.get(team)!.distance_m)} m` : null),
        pair("Average speed", team => motion.get(team)?.mean_speed_m_s != null ? `${n(motion.get(team)!.mean_speed_m_s!, 1)} m/s` : null),
      ].filter((line): line is string => line !== null);
      return { lines, note: "The two sides are the model’s kit-colour groups, not confirmed home and away teams. Distance adds up every tracked player for the time each was measured, so a side with more players in view covers more." };
    }
    case "key-players":
    case "player-statistics": {
      const moved = new Map((c.moving?.players ?? []).map(player => [player.track_id, player]));
      const tracks = [...new Set([...c.byTrack.keys(), ...moved.keys()])].map(track => ({ track, touches: c.byTrack.get(track)?.touches ?? 0, team: c.byTrack.get(track)?.team ?? moved.get(track)?.kit ?? null, motion: moved.get(track) }));
      if (!tracks.length) return null;
      const ranked = tracks.sort((x, y) => y.touches - x.touches || (y.motion?.distance_m ?? 0) - (x.motion?.distance_m ?? 0));
      const line = (row: typeof ranked[number]) => `${row.track} (${label(row.team, "no kit group")}): ${plural(row.touches, "touch", "touches")}${row.motion ? ` · ${n(row.motion.distance_m)} m in ${n(row.motion.measured_seconds, 1)} s · ${n(row.motion.mean_speed_m_s, 1)} m/s average · fastest ${n(row.motion.fastest_hop_m_s, 1)} m/s` : ""}`;
      return id === "key-players"
        ? { lines: ranked.slice(0, 3).map(line), note: "Tracked identities ranked by touches the model saw, then by distance. They are not named players: one player can be more than one identity. Name them under Players." }
        : { lines: [`${plural(tracks.length, "tracked identity", "tracked identities")} with touches or measured movement`, ...ranked.slice(0, 5).map(line)], note: `${ranked.length > 5 ? `Showing 5 of ${ranked.length}; every profile is under Players. ` : ""}Tracked identities, not named players. Distance and speed are for the time each was measured on the pitch map.` };
    }
    case "movement": {
      if (!c.moving) return null;
      const players = c.moving.players ?? [];
      return { lines: [`${plural(players.length, "player")} measured on the estimated pitch map`, ...c.moving.teams.map(team => `${name(team.kit)}: ${n(team.distance_m)} m covered in total · ${team.mean_speed_m_s == null ? "unknown" : `${n(team.mean_speed_m_s, 1)} m/s`} average · fastest half-second ${team.fastest_hop_m_s == null ? "unknown" : `${n(team.fastest_hop_m_s, 1)} m/s`}`)],
        note: `Estimated on an assumed 105 × 68 m pitch, for the time each player was tracked. ${pitchNote(c.pitch)} On labelled clips from another league distances came out about 1.35 times too long. Each player’s figures are under Players.` };
    }
    case "heatmaps": {
      if (!c.moving) return null;
      const shaped = c.insights.collective_dynamics?.teams ?? [];
      return { lines: [`Heat maps for ${plural((c.moving.players ?? []).length, "player")} and ${plural(c.moving.teams.length, "kit group")}`, ...shaped.map(team => `${name(team.kit)}: team shape for ${n(team.supported_seconds, 1)} s of the clip`)],
        note: `The maps are under Players and Shape. ${pitchNote(c.pitch)} A named formation is not estimated.` };
    }
    case "intensity": {
      const totals = new Map<string, number>();
      (c.moving?.players ?? []).forEach(player => Object.entries(player.speed_band_seconds ?? {}).forEach(([band, seconds]) => totals.set(band, (totals.get(band) ?? 0) + seconds)));
      const all = [...totals.values()].reduce((sum, seconds) => sum + seconds, 0);
      if (!all) return null;
      return { lines: BANDS.map(([band, text]) => `${text}: ${n(totals.get(band) ?? 0)} player-seconds (${n(100 * (totals.get(band) ?? 0) / all)}%)`),
        note: "From the video, added over every tracked player for the time each was measured. It is not a GPS load, and on labelled clips time at fast-running speed came out too high. Not for workload decisions." };
    }
    default: return null;
  }
}
