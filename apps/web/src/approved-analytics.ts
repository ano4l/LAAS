// Turns the saved, approved review summary into values for the requested-analytics catalogue.
// Nothing here estimates: a figure is shown only when approved records for it exist, and every
// other entry keeps its reason for being unavailable.
import type { Review, TeamSummary } from "./ReviewPanel";

export type MetricState = "calculated" | "partial" | "unavailable";
/** `lines` are the figures; `note` says what is still missing or why nothing can be shown. */
export type MetricValue = { state: MetricState; lines: string[]; note?: string };
export type Side = "home" | "away";

const SIDES: Side[] = ["home", "away"];
const other = (side: Side): Side => side === "home" ? "away" : "home";
const n = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 1 });
const rate = (value: number | null | undefined) => value == null ? "" : ` (${n(value)}%)`;
const plural = (count: number, one: string, many = `${one}s`) => `${n(count)} ${count === 1 ? one : many}`;
const unavailable = (note: string): MetricValue => ({ state: "unavailable", lines: [], note });
const NONE = "No approved records of this kind are saved for this clip. Record and approve them under Review.";

/** One line per team for an event type, or unavailable when neither team has any. */
function perTeam(saved: Review, describe: (team: TeamSummary) => { recorded: number; text: string }): MetricValue {
  let any = false;
  const lines = SIDES.map(side => {
    const team = saved.summary.teams[side];
    if (!team) return `${saved.teams[side]}: no approved records`;
    const { recorded, text } = describe(team);
    any ||= recorded > 0;
    return `${saved.teams[side]}: ${recorded > 0 ? text : "none recorded"}`;
  });
  return any ? { state: "calculated", lines } : unavailable(NONE);
}

/** "3 recorded · 2 won, 1 lost (66.7%)", with the losing count and rate left out when an older server did not send them. */
function decided(group: { recorded: number; known_outcomes?: number }, good: number, goodWord: string, bad: number | undefined, badWord: string, pct: number | null | undefined) {
  const unknown = group.known_outcomes === undefined ? 0 : group.recorded - group.known_outcomes;
  return `${n(group.recorded)} recorded · ${n(good)} ${goodWord}${bad === undefined ? "" : `, ${n(bad)} ${badWord}`}${rate(pct)}${unknown > 0 ? ` · ${n(unknown)} with unknown outcome` : ""}`;
}

const passes = (team: TeamSummary) => ({
  recorded: team.passes.recorded,
  text: `${n(team.passes.recorded)} recorded · ${n(team.passes.completed)} of ${n(team.passes.known_outcomes)} with a known outcome completed${rate(team.passes.completion_pct)}`
    + (team.passes.recorded > team.passes.known_outcomes ? ` · ${n(team.passes.recorded - team.passes.known_outcomes)} with unknown outcome` : ""),
});
const shots = (team: TeamSummary) => ({
  recorded: team.shots.recorded,
  text: `${plural(team.shots.recorded, "shot")} · ${n(team.shots.on_target)} on target · ${plural(team.shots.goals, "goal")}`
    + (team.shots.blocked ? ` · ${n(team.shots.blocked)} blocked` : "") + (team.shots.off_target ? ` · ${n(team.shots.off_target)} off target` : ""),
});

function pair(saved: Review, first: Side, label: string, pick: (team: TeamSummary) => string | number | null) {
  const a = saved.summary.teams[first], b = saved.summary.teams[other(first)];
  const show = (team: TeamSummary | null) => { const value = team ? pick(team) : null; return value === null ? "unknown" : typeof value === "number" ? n(value) : value; };
  return `${label}: ${show(a)} – ${show(b)}`;
}

function comparison(saved: Review, first: Side): string[] {
  return [
    pair(saved, first, "Approved events", team => team.event_count),
    pair(saved, first, "Shots", team => team.shots.recorded),
    pair(saved, first, "Goals", team => team.shots.goals),
    pair(saved, first, "Pass completion", team => team.passes.completion_pct === null ? null : `${n(team.passes.completion_pct)}%`),
    pair(saved, first, "Tackles won", team => team.tackles.won),
    pair(saved, first, "Interceptions", team => team.interceptions),
    pair(saved, first, "Dribbles successful", team => team.dribbles.successful),
    pair(saved, first, "Duels won", team => team.duels.won),
  ];
}

const playerLine = (saved: Review, player: Review["summary"]["players"][number]) =>
  `${player.player} (${saved.teams[player.team]}): ${plural(player.event_count, "approved event")} · ${Object.entries(player.counts).map(([type, total]) => `${type} ${total}`).join(", ")}`;

/**
 * The value for one catalogue entry, or null when approved events cannot supply it at all
 * (the catalogue then shows what data that entry needs).
 */
export function approvedMetric(id: string, saved: Review, ownSide: Side): MetricValue | null {
  const s = saved.summary, own = s.teams[ownSide], opponentSide = other(ownSide), opponent = s.teams[opponentSide];
  const ownName = saved.teams[ownSide], opponentName = saved.teams[opponentSide];
  switch (id) {
    case "passes": return perTeam(saved, passes);
    case "tackles": return perTeam(saved, team => ({ recorded: team.tackles.recorded, text: decided(team.tackles, team.tackles.won, "won", team.tackles.lost, "lost", team.tackles.win_pct) }));
    case "interceptions": return perTeam(saved, team => ({ recorded: team.interceptions, text: `${n(team.interceptions)} recorded` }));
    case "dribbles": return perTeam(saved, team => ({ recorded: team.dribbles.recorded, text: decided(team.dribbles, team.dribbles.successful, "successful", team.dribbles.unsuccessful, "unsuccessful", team.dribbles.success_pct) }));
    case "shots-goals": return perTeam(saved, shots);
    case "duels": return perTeam(saved, team => ({ recorded: team.duels.recorded, text: decided(team.duels, team.duels.won, "won", team.duels.lost, "lost", team.duels.win_pct) }));
    case "possession": {
      const known = s.possession.home_seconds + s.possession.away_seconds;
      const turnovers = SIDES.some(side => (s.teams[side]?.turnovers?.conceded ?? 0) + (s.teams[side]?.turnovers?.won ?? 0) > 0);
      if (!known && !turnovers) return unavailable("No approved possession intervals, and no approved events that change possession, are saved for this clip.");
      const lines: string[] = [];
      if (known) {
        SIDES.forEach(side => lines.push(`${saved.teams[side]}: ${n(s.possession[`${side}_seconds`])} s of approved possession${rate(s.possession[`${side}_share_pct`])}`));
        lines.push(`Possession is known for ${s.possession.known_coverage_pct === null ? "an unknown share" : `${n(s.possession.known_coverage_pct)}%`} of the clip; the share is of known time only.`);
      }
      if (turnovers) SIDES.forEach(side => { const t = s.teams[side]?.turnovers; if (t) lines.push(`${saved.teams[side]}: lost possession ${plural(t.conceded, "time")}, won it ${plural(t.won, "time")}`); });
      return { state: known && turnovers ? "calculated" : "partial", lines,
        note: !known ? "Possession share needs approved possession intervals." : !turnovers ? "Turnovers need approved passes, dribbles, tackles or interceptions." : s.definitions?.turnovers };
    }
    case "home-away": {
      if (!s.teams.home && !s.teams.away) return unavailable(NONE);
      if (!s.teams.home || !s.teams.away) return { state: "partial", lines: comparison(saved, "home"), note: `Only ${saved.teams[s.teams.home ? "home" : "away"]} has approved records, so there is nothing to compare yet.` };
      return { state: "calculated", lines: [`${saved.teams.home} – ${saved.teams.away}`, ...comparison(saved, "home")] };
    }
    case "key-players": {
      if (!s.players.length) return unavailable("No approved event names a player. Type a player on each event under Review.");
      const ranked = [...s.players].sort((a, b) => b.event_count - a.event_count);
      return { state: "partial", lines: ranked.slice(0, 3).map(player => playerLine(saved, player)),
        note: "Ranked by number of approved events only. Player names are typed by the reviewer, not checked against a roster. In-game problems are not calculated." };
    }
    case "opponent-threats": {
      if (!opponent) return unavailable(`No approved events are saved for ${opponentName}.`);
      const shooters = s.players.filter(player => player.team === opponentSide && player.counts.shot).sort((a, b) => b.counts.shot - a.counts.shot);
      return { state: "partial", lines: [
        `${opponentName}: ${shots(opponent).text}`,
        ...(opponent.dribbles.recorded ? [`${opponentName}: ${n(opponent.dribbles.successful)} of ${plural(opponent.dribbles.recorded, "dribble")} successful`] : []),
        ...(opponent.passes.completed ? [`${opponentName}: ${plural(opponent.passes.completed, "completed pass", "completed passes")}`] : []),
        ...(shooters.length ? [`Shots by player: ${shooters.slice(0, 3).map(player => `${player.player} ${player.counts.shot}`).join(", ")}`] : []),
      ], note: "Counts only. Where threats come from, and expected threat, need a pitch location on each event." };
    }
    case "opponent-weaknesses": {
      if (!opponent && !own) return unavailable(NONE);
      const lines: string[] = [];
      if (opponent?.turnovers?.conceded) lines.push(`${opponentName} lost possession ${plural(opponent.turnovers.conceded, "time")}: ${plural(opponent.turnovers.incomplete_passes, "incomplete pass", "incomplete passes")}, ${plural(opponent.turnovers.unsuccessful_dribbles, "unsuccessful dribble")}`);
      if (opponent?.duels.lost) lines.push(`${opponentName} lost ${n(opponent.duels.lost)} of ${n(opponent.duels.known_outcomes ?? 0)} duels with a known outcome`);
      if (own?.shots.recorded) lines.push(`${ownName} against them: ${shots(own).text}`);
      if (!lines.length) return unavailable(`No approved records show ${opponentName} losing the ball, losing a duel or conceding a shot.`);
      return { state: "partial", lines, note: "Counts only. Which areas of the pitch are weak needs a pitch location on each event." };
    }
    case "opponent-compare": {
      if (!own && !opponent) return unavailable(NONE);
      const lines = [`${ownName} – ${opponentName}`, ...comparison(saved, ownSide)];
      return own && opponent ? { state: "calculated", lines } : { state: "partial", lines, note: `Only ${own ? ownName : opponentName} has approved records, so there is nothing to compare yet.` };
    }
    case "player-statistics": {
      if (!s.players.length) return unavailable("No approved event names a player. Type a player on each event under Review.");
      const ranked = [...s.players].sort((a, b) => b.event_count - a.event_count);
      return { state: "partial", lines: [`${plural(s.players.length, "named player")} with approved events in this clip`, ...ranked.slice(0, 5).map(player => playerLine(saved, player))],
        note: `${ranked.length > 5 ? `Showing 5 of ${ranked.length}; all are listed under Review. ` : ""}This clip only. Season performance needs saved matches over time, and names are reviewer labels, not a verified roster.` };
    }
    default: return null;
  }
}
