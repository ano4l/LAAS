import { useEffect, useId, useMemo, useRef, useState } from "react";
import { AlertTriangle, ArrowDownToLine } from "lucide-react";
import type { AnalysisResult } from "./main";
import { DASH, num, range, secs, sentence } from "./format";
import { TeamName, useKits } from "./kits";
import { PitchBoard } from "./PitchBoard";
import { candidateName, candidatePlayers } from "./VideoInsights";
import type { MotionPlayer } from "./SoccerTrackShape";
import { DataTable, Disclosure, EmptyState, Field, Meter, Panel, StatTile, WatchButton } from "./ui";

/** A name a person gave to one or more tracked identities. Nothing here is recognised automatically. */
/** `lineupId` points at an entry in the line-up typed under Review; the name and number then come from there. */
type NamedPlayer = { key: string; name: string; number: string; tracks: string[]; lineupId: string | null };
/** One person in the video's shared roster: the line-up, the tracked identities named for them, and their approved events. */
type RosterRow = { source: "lineup" | "tracked_only" | "event_only"; lineup_id: string | null; team: "home" | "away" | null; name: string; number: string | null;
  position: string | null; minutes: number | null; tracks: string[]; event_count: number; counts: Record<string, number>; per_90: Record<string, number> | null };
type VideoRoster = { teams: { home: string; away: string }; players: RosterRow[] };
type Labels = { schema_version: string; run_id: string; source_sha256: string | null; revision: number; players: { id: string; name: string; number: string | null; tracks: string[]; lineup_id?: string | null; lineup_missing?: boolean }[]; overlap_checked: boolean | null };
type Identity = { track_id: string; object_type: string; team: string | null; first_seen_ms: number; last_seen_ms: number; fragments: number };
/** Two identities that could be one player, for a person to check. Never applied automatically. */
type Suggestion = { from_track: string; to_track: string; gap_seconds: number; from_last_seen_seconds: number; to_first_seen_seconds: number; same_kit: boolean | null; basis: "pitch" | "image"; distance_m: number | null; distance_heights: number | null; rivals: number; fit?: number; close_fit?: boolean };
/** A shirt number an experimental text reader saw repeatedly on one identity. A question for a person, not a recognition. */
type NumberProposal = { track_id: string; proposed_number: string; agreement: number; confident_reads: number; crops_read: number };
/** What an identity's shirt-number reads say once only numbers on the saved line-up can be right. Still a question for a person. */
type LineupMatch = { track_id: string; status: "match" | "ambiguous_team" | "too_few_reads" | "not_on_sheet" | "keeper_number_on_outfield_player";
  number: string; reads: number; confident_reads: number; one_digit?: boolean; team: "home" | "away" | null; team_basis?: string | null;
  candidates: { team: "home" | "away"; lineup_id: string; name: string; number: string; position: string | null }[]; same_player_tracks?: string[]; kit?: string | null };
type Suggestions = { suggestions: Suggestion[]; suggestions_total: number; appearances?: Record<string, [number, number][]>; reason?: string; shirt_numbers?: { reader: string | null; proposals: NumberProposal[]; lineup?: { status: string; matches: LineupMatch[] } } };
type RosterEntry = { name: string; number: string | null; clips: number };

const BANDS: [string, string][] = [["walking", "Standing or walking (under 2 m/s)"], ["jogging", "Jogging (2 to 4 m/s)"], ["running", "Running (4 to 5.5 m/s)"], ["fast_running", "Fast running (5.5 m/s and over)"]];
const comparable = (players: NamedPlayer[]) => JSON.stringify(players.map(player => [player.name.trim(), player.number.trim(), [...player.tracks].sort(), player.lineupId]));
const fromSaved = (labels: Labels): NamedPlayer[] => labels.players.map(player => ({ key: player.id, name: player.name, number: player.number ?? "", tracks: player.tracks,
  // A link whose line-up entry was removed is kept as a typed name; saving again drops the broken link.
  lineupId: player.lineup_id && !player.lineup_missing ? player.lineup_id : null }));
const title = (player: NamedPlayer) => `${player.number.trim() ? `${player.number.trim()} · ` : ""}${player.name.trim() || "Unnamed"}`;
let nextKey = 0;

/**
 * Everything measured for one player: either a single tracked identity, or the identities a person has
 * said are the same player. Totals add up only the time those identities were tracked.
 */
export function PlayerProfiles({ result, api, authHeaders, selected, onSelect, reviewRevision }: {
  result: AnalysisResult; api: string; authHeaders: () => Record<string, string>; selected: string | null; onSelect: (trackId: string) => void;
  /** Changes when the review, and so possibly the line-up, is saved. */
  reviewRevision?: number;
}) {
  const kits = useKits();
  const rosterId = useId();
  const runId = result.run_id, sourceSha = result.metadata.sha256, duration = result.metadata.duration_seconds;
  const epoch = useRef(0);
  const [saved, setSaved] = useState<Labels | null>(null);
  const [players, setPlayers] = useState<NamedPlayer[]>([]);
  const [pending, setPending] = useState(false); const [error, setError] = useState(""); const [conflict, setConflict] = useState(false);
  const [confirmed, setConfirmed] = useState(false); const [reload, setReload] = useState(0);
  const [newName, setNewName] = useState(""); const [newNumber, setNewNumber] = useState("");
  const [suggested, setSuggested] = useState<Suggestions | null>(null);
  const [looseFor, setLooseFor] = useState<string | null>(null); // the player whose looser suggestions are shown
  const [roster, setRoster] = useState<RosterEntry[]>([]);
  const [compare, setCompare] = useState("");

  const identities = useMemo<Identity[]>(() => {
    const linked = result.detection_summary?.linked_tracks;
    const rows = linked
      ? linked.map(t => ({ track_id: t.linked_track_id, object_type: t.object_type, team: t.team, first_seen_ms: t.first_seen_ms, last_seen_ms: t.last_seen_ms, fragments: t.track_ids.length }))
      : (result.detection_summary?.tracks ?? []).map(t => ({ track_id: t.track_id, object_type: t.object_type, team: null, first_seen_ms: t.first_seen_ms, last_seen_ms: t.last_seen_ms, fragments: 1 }));
    return rows.filter(row => row.object_type === "player" || row.object_type === "goalkeeper").sort((a, b) => a.first_seen_ms - b.first_seen_ms);
  }, [result]);
  const byId = useMemo(() => new Map(identities.map(identity => [identity.track_id, identity])), [identities]);
  const measured = useMemo(() => new Map(result.metrics.video_measurements?.tracks.map(track => [track.track_id, track]) ?? []), [result]);
  const motionData = result.metrics.video_insights?.collective_dynamics?.pitch_motion;
  const motion = motionData?.status === "estimated" ? motionData : undefined;
  // Set when most checked frames' pitch lines, drawn from the calibration, miss the lines in the video.
  const lineWarning = motion ? result.metrics.video_insights?.collective_dynamics?.pitch?.line_check?.warning ?? null : null;
  const motionTracks = useMemo(() => new Map<string, MotionPlayer>(motion?.players?.map(track => [track.track_id, track] as [string, MotionPlayer]) ?? []), [motion]);
  const spells = result.metrics.video_insights?.control.spells ?? [];
  const candidates = result.metrics.video_insights?.candidates ?? [];

  useEffect(() => {
    const request = ++epoch.current; const abort = new AbortController(); setPending(true); setError(""); setConflict(false); setConfirmed(false);
    fetch(`${api}/runs/${encodeURIComponent(runId)}/players`, { headers: authHeaders(), signal: abort.signal })
      .then(async response => { const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Player names could not be loaded (${response.status}).`); if (body.run_id !== runId || (sourceSha && body.source_sha256 !== sourceSha)) throw new Error("Saved player names belong to a different clip. Nothing was loaded."); return body as Labels; })
      .then(body => { if (request !== epoch.current) return; setSaved(body); setPlayers(fromSaved(body)); })
      .catch(cause => { if (!abort.signal.aborted && request === epoch.current) setError(String(cause instanceof Error ? cause.message : cause)); })
      .finally(() => { if (request === epoch.current) setPending(false); });
    return () => { abort.abort(); epoch.current++; };
  }, [runId, sourceSha, api, reload]);

  // Suggestions and the list of names saved on other clips are optional extras: the profile works without them.
  useEffect(() => {
    const abort = new AbortController();
    fetch(`${api}/runs/${encodeURIComponent(runId)}/players/suggestions`, { headers: authHeaders(), signal: abort.signal })
      .then(response => response.ok ? response.json() : null).then(body => { if (body && Array.isArray(body.suggestions)) setSuggested(body as Suggestions); }).catch(() => undefined);
    return () => abort.abort();
  }, [runId, api]);
  useEffect(() => {
    const abort = new AbortController();
    fetch(`${api}/roster`, { headers: authHeaders(), signal: abort.signal })
      .then(response => response.ok ? response.json() : null).then(body => { if (body && Array.isArray(body.players)) setRoster(body.players as RosterEntry[]); }).catch(() => undefined);
    return () => abort.abort();
  }, [api, saved?.revision]);
  // The shared roster: the line-up to choose from, and each listed player's approved events.
  const [videoRoster, setVideoRoster] = useState<VideoRoster | null>(null);
  const [newLineup, setNewLineup] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    fetch(`${api}/runs/${encodeURIComponent(runId)}/roster`, { headers: authHeaders(), signal: abort.signal })
      .then(response => response.ok ? response.json() : null).then(body => { if (body && Array.isArray(body.players)) setVideoRoster(body as VideoRoster); }).catch(() => undefined);
    return () => abort.abort();
  }, [runId, api, saved?.revision, reviewRevision]);
  const lineupRows = (videoRoster?.players ?? []).filter(row => row.source === "lineup" && row.lineup_id);
  const lineupOptions = (takenBy: string | null) => videoRoster && (["home", "away"] as const).map(side => (
    <optgroup key={side} label={videoRoster.teams[side]}>
      {lineupRows.filter(row => row.team === side).map(row => (
        <option key={row.lineup_id} value={row.lineup_id!} disabled={players.some(other => other.key !== takenBy && other.lineupId === row.lineup_id)}>{row.number ? `${row.number} · ` : ""}{row.name}</option>
      ))}
    </optgroup>
  ));

  const dirty = Boolean(saved) && comparable(players) !== comparable(fromSaved(saved!));
  const owner = (trackId: string) => players.find(player => player.tracks.includes(trackId));
  const current = selected && byId.has(selected) ? selected : identities[0]?.track_id ?? null;
  const player = current ? owner(current) : undefined;
  const trackIds = player ? player.tracks : current ? [current] : [];
  const unnamed = identities.filter(identity => !owner(identity.track_id));
  const change = (next: NamedPlayer[]) => { setPlayers(next); setConfirmed(false); setError(""); };
  const edit = (key: string, patch: Partial<NamedPlayer>) => change(players.map(row => row.key === key ? { ...row, ...patch } : row));
  const tracksOf = (trackId: string) => owner(trackId)?.tracks ?? [trackId];
  const nameOf = (trackId: string) => { const named = owner(trackId); return named ? title(named) : `Unnamed · ${trackId}`; };
  const pickKnown = (name: string) => roster.find(entry => entry.name === name);

  async function save() {
    if (!saved || !dirty || !confirmed || pending || conflict) return;
    if (players.some(row => !row.name.trim())) { setError("Every named player needs a name. Remove the name or type one."); return; }
    if (players.some(row => row.number.trim() && !/^\d{1,2}$/.test(row.number.trim()))) { setError("A shirt number is one or two digits, or left empty."); return; }
    const request = epoch.current; setPending(true); setError("");
    try {
      const response = await fetch(`${api}/runs/${encodeURIComponent(runId)}/players`, { method: "POST", headers: { ...authHeaders(), "Content-Type": "application/json" }, body: JSON.stringify({ revision: saved.revision, confirmed: true, players: players.filter(row => row.tracks.length).map(row => ({ name: row.name.trim(), number: row.number.trim() || null, tracks: row.tracks, ...(row.lineupId ? { lineup_id: row.lineupId } : {}) })) }) });
      const body = await response.json().catch(() => ({})); if (request !== epoch.current) return;
      if (response.status === 409) { setConflict(true); throw new Error("Another session changed these names. Your edits are kept here. Reload the saved names only when you are ready to discard them."); }
      if (!response.ok) throw new Error(typeof body.detail === "string" ? `${body.detail}. Nothing was saved; your edits are kept.` : `Saving failed (${response.status}). Your edits are kept.`);
      setSaved(body); setPlayers(fromSaved(body)); setConfirmed(false);
    } catch (cause) { if (request === epoch.current) setError(String(cause instanceof Error ? cause.message : cause)); }
    finally { if (request === epoch.current) setPending(false); }
  }

  if (!identities.length) return <EmptyState title="No tracked players in this clip">The tracker did not follow any player, so there are no profiles to show.</EmptyState>;

  /** Totals over a set of identities. Each figure counts only the time those identities were tracked. */
  function summarise(ids: string[]) {
    const seen = ids.map(id => measured.get(id)).filter(row => row !== undefined);
    const moved = ids.map(id => motionTracks.get(id)).filter((row): row is MotionPlayer => row !== undefined);
    const measuredSeconds = moved.reduce((sum, row) => sum + row.measured_seconds, 0);
    const distance = moved.reduce((sum, row) => sum + row.distance_m, 0);
    const banded = moved.filter(row => row.speed_band_seconds);
    const ownSpells = spells.filter(spell => ids.includes(spell.track_id)).sort((a, b) => a.start_seconds - b.start_seconds);
    return {
      observedSeconds: seen.length ? seen.reduce((sum, row) => sum + row.observed_seconds, 0) : null,
      nearBall: seen.length ? seen.reduce((sum, row) => sum + row.near_ball_frames, 0) : null,
      moved, measuredSeconds, distance,
      speed: moved.length && measuredSeconds > 0 ? distance / measuredSeconds : null,
      fastest: moved.length ? Math.max(...moved.map(row => row.fastest_hop_m_s)) : null,
      bands: banded.length ? BANDS.map(([key]) => banded.reduce((sum, row) => sum + (row.speed_band_seconds?.[key] ?? 0), 0)) : null,
      ownSpells, spellSeconds: ownSpells.reduce((sum, spell) => sum + spell.observed_seconds, 0),
      events: candidates.filter(event => (event.from_track !== null && ids.includes(event.from_track)) || (event.to_track !== null && ids.includes(event.to_track))).sort((a, b) => a.timestamp_seconds - b.timestamp_seconds),
    };
  }

  const own = summarise(trackIds);
  const { moved, measuredSeconds, distance } = own;
  const cells = motion && moved.length ? Array.from({ length: motion.grid.rows }, () => Array<number>(motion.grid.columns).fill(0)) : null;
  if (cells) moved.forEach(row => row.heat_cells.forEach(([r, c, value]) => { if (cells[r]?.[c] !== undefined) cells[r][c] += value; }));
  const placedSeconds = moved.reduce((sum, row) => sum + row.placed_seconds, 0);
  const kitIds = [...new Set(trackIds.map(id => byId.get(id)?.team).filter((team): team is string => Boolean(team)))];
  const heading = player ? title(player) : `Unnamed · ${current}`;
  const linkedRow = player?.lineupId ? lineupRows.find(row => row.lineup_id === player.lineupId) : undefined;
  const status = pending ? "Working…" : dirty ? "Unsaved names" : saved ? (saved.players.length ? `${saved.players.length} named · saved version ${saved.revision}` : "No names saved") : "Names not loaded";
  const spans = trackIds.flatMap(id => suggested?.appearances?.[id] ?? (byId.get(id) ? [[byId.get(id)!.first_seen_ms / 1000, byId.get(id)!.last_seen_ms / 1000] as [number, number]] : []));
  const share = (seconds: number) => `${Math.max(0, Math.min(100, 100 * seconds / Math.max(duration, .001)))}%`;
  const possible = (suggested?.suggestions ?? []).filter(row => trackIds.includes(row.from_track) !== trackIds.includes(row.to_track));
  // Close fits first and on their own: on labelled test clips they were right about half the time, the rest about one in five.
  // A response from before close fits were marked lists everything, as it did then.
  const graded = possible.some(row => row.close_fit !== undefined);
  const closeFits = graded ? possible.filter(row => row.close_fit) : possible;
  const looseFits = graded ? possible.filter(row => !row.close_fit) : [];
  const listed = looseFor === current ? [...closeFits, ...looseFits] : closeFits;
  const other = compare && byId.has(compare) && !trackIds.includes(compare) ? summarise(tracksOf(compare)) : null;
  const options = <>
    {players.some(row => row.tracks.length) && <optgroup label="Named players">
      {players.filter(row => row.tracks.length).map(row => <option key={row.key} value={row.tracks[0]}>{title(row)} ({row.tracks.length} {row.tracks.length === 1 ? "identity" : "identities"})</option>)}
    </optgroup>}
    <optgroup label="Unnamed tracked identities">
      {unnamed.map(identity => <option key={identity.track_id} value={identity.track_id}>{identity.track_id} · {identity.object_type}{identity.team ? ` · ${kits.label(identity.team)}` : ""} · first seen {secs(identity.first_seen_ms / 1000)}</option>)}
    </optgroup>
  </>;

  const readNumbers = (suggested?.shirt_numbers?.proposals ?? []).filter(row => trackIds.includes(row.track_id));
  const lineupMatches = (suggested?.shirt_numbers?.lineup?.matches ?? []).filter(row => trackIds.includes(row.track_id) && row.status !== "too_few_reads");
  // Puts an identity under a line-up player. It only edits the names on screen: the reviewer still confirms and saves.
  function chooseLineup(trackId: string, who: LineupMatch["candidates"][number]) {
    const holder = players.find(row => row.lineupId === who.lineup_id);
    if (holder) { if (!holder.tracks.includes(trackId)) edit(holder.key, { tracks: [...holder.tracks, trackId] }); return; }
    const mine = owner(trackId);
    if (mine) edit(mine.key, { lineupId: who.lineup_id, name: who.name, number: who.number });
    else change([...players, { key: `new-${nextKey++}`, name: who.name, number: who.number, tracks: [trackId], lineupId: who.lineup_id }]);
  }

  function join(row: Suggestion) {
    const mine = trackIds.includes(row.from_track) ? row.from_track : row.to_track, theirs = mine === row.from_track ? row.to_track : row.from_track;
    const theirOwner = owner(theirs);
    if (player && !theirOwner) edit(player.key, { tracks: [...player.tracks, theirs] });
    else if (!player && theirOwner) edit(theirOwner.key, { tracks: [...theirOwner.tracks, mine] });
  }

  function download() {
    const rows: (string | number | null)[][] = [
      ["metric", "value", "unit", "note"],
      ["player", heading, "", player ? "Name entered by a person" : "Unnamed tracked identity"],
      ["tracked_identities", trackIds.join(" "), "", "Unreviewed tracker identities"],
      ["kit_group", kitIds.map(kit => kits.label(kit)).join(" ") || null, "", ""],
      ["on_screen", own.observedSeconds, "s", "Observed seconds"],
      ["near_ball_frames", own.nearBall, "frames", "Proximity, not possession"],
      ["measured_on_pitch_map", moved.length ? measuredSeconds : null, "s", "Only time tracked in a calibrated frame"],
      ["distance", moved.length ? distance : null, "m", "Estimate on an assumed 105 x 68 m pitch"],
      ["average_speed", own.speed, "m/s", "Estimate"],
      ["fastest_half_second", own.fastest, "m/s", "Estimate"],
      ...BANDS.map(([key, label], index) => [`speed_band_${key}`, own.bands ? own.bands[index] : null, "s", label] as (string | number | null)[]),
      ["ball_near_feet_spells", own.ownSpells.length, "spells", "Estimate, not confirmed touches"],
      ["ball_near_feet", own.spellSeconds, "s", "Estimate"],
      ["event_candidates", own.events.length, "candidates", "Unreviewed"],
    ];
    const cell = (value: string | number | null) => { const text = value == null ? "" : typeof value === "number" ? String(Math.round(value * 100) / 100) : value; const safe = typeof value === "string" && /^(\s*[=+\-@]|[\t\r])/.test(text) ? `'${text}` : text; return `"${safe.replaceAll('"', '""')}"`; };
    const url = URL.createObjectURL(new Blob([rows.map(row => row.map(cell).join(",")).join("\r\n")], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = `${runId}-player-${trackIds[0]}-estimates.csv`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <Panel
      title="Player profile"
      subtitle="Everything measured for one player. A name is something a person typed after watching the footage; the app does not recognise faces or read shirt numbers."
      aside={<span className="muted">{status}</span>}
    >
      <div className="stack" id="player-profile" aria-busy={pending}>
        <div className="toolbar">
          <Field label="Show profile for"><select value={current ?? ""} onChange={event => onSelect(event.target.value)}>{options}</select></Field>
          <Field label="Compare with">
            <select value={other ? compare : ""} onChange={event => setCompare(event.target.value)}><option value="">Nobody</option>{options}</select>
          </Field>
        </div>

        <div className="profile__head">
          <h4 className="profile__name">{heading}</h4>
          <p className="note">
            {kitIds.length ? kitIds.map(kit => <TeamName key={kit} id={kit} />) : <span className="muted">No kit group</span>}
            {" · "}{trackIds.length === 1 ? `Tracked identity ${trackIds[0]}` : `${trackIds.length} tracked identities: ${trackIds.join(", ")}`}
            {kitIds.length > 1 && " · These identities were sorted into different kit groups. Check the name against the footage."}
          </p>
          {linkedRow && videoRoster && (
            <p className="note">
              In the line-up for {videoRoster.teams[linkedRow.team!]}{linkedRow.position ? ` · ${linkedRow.position}` : ""}{linkedRow.minutes != null ? ` · ${num(linkedRow.minutes)} minutes in this video` : ""}.
            </p>
          )}
        </div>

        {linkedRow && (
          <div className="stack stack--sm">
            <h4>Approved events</h4>
            {linkedRow.event_count > 0 ? <>
              <p className="facts">{Object.entries(linkedRow.counts).map(([type, total]) => <span key={type}>{sentence(type)}<b>{total}</b></span>)}</p>
              {linkedRow.per_90 && <p className="note">Per 90 minutes: {Object.entries(linkedRow.per_90).map(([type, value]) => `${type} ${num(value)}`).join(" · ")}.</p>}
            </> : <p className="note">No approved event names this player yet.</p>}
            <p className="note">Recorded and approved by a person under Review, and matched to this player through the line-up. They are counted separately from the automatic estimates on this page.</p>
          </div>
        )}

        {readNumbers.length > 0 && (
          <div className="note note--boxed">
            <p>An experimental text reader saw a number on this player’s shirt. It was trained on general text, not shirts. On ten labelled test clips from another league, 27 of 29 two-digit proposals that could be checked were right. Watch the footage before using it.</p>
            <ul className="itemlist">
              {readNumbers.map(row => (
                <li key={row.track_id}>
                  <span><span className="mono">{row.track_id}</span>: read <strong>{row.proposed_number}</strong> in {Math.round(row.agreement * row.confident_reads)} of {row.confident_reads} confident reads ({row.crops_read} crops tried)</span>
                  <span className="row">
                    <WatchButton at={(byId.get(row.track_id)?.first_seen_ms ?? 0) / 1000} label="Watch" describe={`Watch ${row.track_id} to check the number`} />
                    <button type="button" className="btn btn--secondary btn--sm" disabled={pending || !saved} onClick={() => { if (player) edit(player.key, { number: row.proposed_number }); else setNewNumber(row.proposed_number); }}>Use {row.proposed_number}</button>
                  </span>
                </li>
              ))}
            </ul>
            {!player && <p>“Use” fills in the number below; the player still needs a name before anything is saved.</p>}
          </div>
        )}

        {lineupMatches.length > 0 && videoRoster && (
          <div className="note note--boxed">
            <p>The same reads, checked against the line-up saved under Review. A match says only that the digits are a number on the team sheet. This check has not been scored against labelled footage, so watch the player before using it.</p>
            <ul className="itemlist">
              {lineupMatches.map(row => {
                const who = row.candidates[0], teams = videoRoster.teams;
                return (
                  <li key={row.track_id}>
                    <span>
                      <span className="mono">{row.track_id}</span>: read <strong>{row.number}</strong> in {row.reads} of {row.confident_reads} confident reads.{" "}
                      {row.status === "match" && <>On the line-up, {row.number} is <strong>{who.name}</strong> ({teams[who.team]}).
                        {row.one_digit && " One-digit reads are often wrong, so this is a weak match."}
                        {row.team_basis === "number_worn_on_one_team_only" && " Only one team lists this number; which team this identity is on is not otherwise known."}
                        {row.same_player_tracks?.length ? ` The same player was also read on ${row.same_player_tracks.join(", ")}: possibly one player split by the tracker.` : ""}</>}
                      {row.status === "ambiguous_team" && `Both teams list ${row.number}: ${row.candidates.map(entry => `${entry.name} (${teams[entry.team]})`).join(" and ")}. ${row.kit ? "Name one player from this identity’s kit group first, and the team will be known." : "This identity has no kit group, so its team cannot be worked out from the others."}`}
                      {row.status === "not_on_sheet" && `Nobody on the saved line-up wears ${row.number}${row.team ? ` for ${teams[row.team]}` : ""}. It may be a misread, or the line-up may be incomplete.`}
                      {row.status === "keeper_number_on_outfield_player" && `The line-up gives ${row.number} to a goalkeeper, but this identity is not classed as one. It is probably a misread.`}
                    </span>
                    <span className="row">
                      <WatchButton at={(byId.get(row.track_id)?.first_seen_ms ?? 0) / 1000} label="Watch" describe={`Watch ${row.track_id} to check the number`} />
                      {row.status === "match" && <button type="button" className="btn btn--secondary btn--sm" disabled={pending || !saved} onClick={() => chooseLineup(row.track_id, who)}>This is {who.name}</button>}
                    </span>
                  </li>
                );
              })}
            </ul>
            <p>“This is…” puts the identity under that player in the names below. Nothing is saved until you confirm and save.</p>
          </div>
        )}

        {lineWarning && moved.length > 0 && <p className="note note--boxed" role="note"><strong>Distance, speed and heat map are rough here. </strong>{lineWarning}</p>}
        <div className="tiles tiles--four tiles--compact">
          <StatTile label="On screen" value={own.observedSeconds == null ? DASH : num(own.observedSeconds)} unit="s" evidence={own.observedSeconds == null ? "unknown" : "seen"}
            title="Technical name: observed seconds, summed over the selected identities"
            context={own.nearBall == null ? "Not measured in this analysis" : `Near the ball in ${num(own.nearBall, 0)} analysed frames`} />
          <StatTile label="Distance" value={moved.length ? num(distance, 0) : DASH} unit="m" evidence={moved.length ? "estimated" : "unknown"}
            title="Estimated distance on an assumed 105 x 68 m pitch, summed in half-second hops"
            context={moved.length ? `Over ${secs(measuredSeconds)} measured on the pitch map, not the whole clip` : !motion ? "No pitch map for this clip" : motion.players ? "Tracked too briefly on the pitch map to measure" : "Not measured per player in this analysis"} />
          <StatTile label="Average speed" value={own.speed == null ? DASH : num(own.speed)} unit="m/s" evidence={own.speed == null ? "unknown" : "estimated"}
            title="Estimated distance divided by measured seconds"
            context={own.speed == null ? "Unknown" : `About ${num(own.speed * 3.6)} km/h`} />
          <StatTile label="Fastest half-second" value={own.fastest == null ? DASH : num(own.fastest)} unit="m/s" evidence={own.fastest == null ? "unknown" : "estimated"}
            title="Fastest half-second hop; hops above 11 m/s are rejected as errors"
            context={own.fastest == null ? "Unknown" : `About ${num(own.fastest * 3.6)} km/h`} />
        </div>

        <div className="stack stack--sm">
          <h4>Through the clip</h4>
          <div className="profile__timeline" role="img" aria-label={`${heading} is on screen in ${spans.length} ${spans.length === 1 ? "stretch" : "stretches"} of this ${secs(duration)} clip, with the ball near the feet in ${own.ownSpells.length} ${own.ownSpells.length === 1 ? "spell" : "spells"}. Times are listed below.`}>
            {spans.map(([a, b], index) => <i key={`s${index}`} className="profile__span" style={{ left: share(a), width: share(Math.max(b - a, duration / 400)) }} />)}
            {own.ownSpells.map(spell => <i key={spell.id} className="profile__spell" style={{ left: share(spell.start_seconds), width: share(Math.max(spell.end_seconds - spell.start_seconds, duration / 400)) }} />)}
          </div>
          <p className="note profile__key"><span><i className="profile__span" />On screen</span><span><i className="profile__spell" />Ball near the feet (estimate)</span><span className="mono">0 s to {secs(duration)}</span></p>
        </div>

        {cells && motion ? (
          <div className="profile__map">
            <PitchBoard length={motion.grid.columns * motion.grid.cell_length_m} width={motion.grid.rows * motion.grid.cell_width_m} groups={[]} visibleArea={[]} showIds={false}
              label={`Estimated heat map for ${heading}: where on the pitch this player was placed, over ${num(placedSeconds)} seconds`}
              heat={{ cells, coverage: motion.coverage_seconds ?? [], tone: kits.tone(kitIds[0]), label: heading }} />
            <p className="note">Estimated position over {secs(placedSeconds)} placed on the pitch map. Brighter cells mean more time. Dark cells were never in the camera’s view, so nothing is known there. Which end is which is not established.</p>
          </div>
        ) : (
          <p className="note note--boxed">{!motion ? "No heat map: this clip has no pitch map." : !motion.players ? "No heat map: this clip was analysed before per-player maps existed. It is recalculated the next time the clip is opened from the server." : `No heat map: this player was measured for under ${motion.parameters.min_listed_track_seconds} seconds on the pitch map${(motion.players_listed_total ?? 0) > motion.players.length ? `, or is outside the ${motion.players.length} longest-measured identities kept` : ""}.`}</p>
        )}

        {own.bands && measuredSeconds > 0 && (
          <div className="stack stack--sm">
            <h4>Time at each speed</h4>
            <ul className="profile__bands">
              {BANDS.map(([key, label], index) => (
                <li key={key}>
                  <span>{label}</span>
                  <Meter value={100 * own.bands![index] / measuredSeconds} tone={kits.tone(kitIds[0])} />
                  <span className="mono">{secs(own.bands![index])} · {num(100 * own.bands![index] / measuredSeconds, 0)}%</span>
                </li>
              ))}
            </ul>
            <p className="note">Estimated from half-second steps over the {secs(measuredSeconds)} measured on the pitch map. Half-second averages flatten short bursts, so time in the faster bands is undercounted.</p>
          </div>
        )}

        {other && (
          <div className="stack stack--sm">
            <h4>Compared with {nameOf(compare)}</h4>
            <DataTable label={`${heading} compared with ${nameOf(compare)}`}>
              <thead><tr><th scope="col">Figure</th><th scope="col" className="num">{heading}</th><th scope="col" className="num">{nameOf(compare)}</th></tr></thead>
              <tbody>
                {([
                  ["On screen", secs(own.observedSeconds), secs(other.observedSeconds)],
                  ["Measured on the pitch map", own.moved.length ? secs(own.measuredSeconds) : DASH, other.moved.length ? secs(other.measuredSeconds) : DASH],
                  ["Distance (estimate)", own.moved.length ? `${num(own.distance, 0)} m` : DASH, other.moved.length ? `${num(other.distance, 0)} m` : DASH],
                  ["Average speed (estimate)", own.speed == null ? DASH : `${num(own.speed)} m/s`, other.speed == null ? DASH : `${num(other.speed)} m/s`],
                  ["Fastest half-second (estimate)", own.fastest == null ? DASH : `${num(own.fastest)} m/s`, other.fastest == null ? DASH : `${num(other.fastest)} m/s`],
                  ["Running or faster (estimate)", own.bands ? secs(own.bands[2] + own.bands[3]) : DASH, other.bands ? secs(other.bands[2] + other.bands[3]) : DASH],
                  ["Ball near the feet (estimate)", `${own.ownSpells.length} · ${secs(own.spellSeconds)}`, `${other.ownSpells.length} · ${secs(other.spellSeconds)}`],
                  ["Event candidates (unreviewed)", String(own.events.length), String(other.events.length)],
                ] as [string, string, string][]).map(([label, mine, theirs]) => <tr key={label}><th scope="row">{label}</th><td className="num">{mine}</td><td className="num">{theirs}</td></tr>)}
              </tbody>
            </DataTable>
            <p className="note">Distance depends on how long each player was measured, so compare distances only when the measured times are close. Average speed is fairer across different times.</p>
          </div>
        )}

        <div className="stack stack--sm">
          <h4>Ball near the feet</h4>
          {own.ownSpells.length ? <>
            <p className="note">{own.ownSpells.length} {own.ownSpells.length === 1 ? "spell" : "spells"}, {secs(own.spellSeconds)} in total. A spell is the ball staying close to this player’s feet in the image. It is an estimate, not a confirmed touch or possession.</p>
            <ul className="itemlist">
              {own.ownSpells.slice(0, 12).map(spell => (
                <li key={spell.id}>
                  <span className="mono">{range(spell.start_seconds, spell.end_seconds)}</span>
                  <WatchButton at={spell.start_seconds} label="Watch" describe={`Watch the spell at ${secs(spell.start_seconds)}`} />
                </li>
              ))}
            </ul>
            {own.ownSpells.length > 12 && <p className="note">Showing the first 12 of {own.ownSpells.length}. All spells are listed under Possession.</p>}
          </> : <p className="note">No spell with the ball near this player’s feet met the support rule.</p>}
        </div>

        {own.events.length > 0 && (
          <div className="stack stack--sm">
            <h4>Possible events</h4>
            <p className="note">Possible passes, turnovers, shots, tackles, duels and dribbles this player appears in. These are unreviewed prompts from rules over the tracking, not confirmed events, and they miss most real ones.</p>
            <ul className="itemlist">
              {own.events.slice(0, 12).map(event => {
                const gave = event.from_track !== null && trackIds.includes(event.from_track), partner = gave ? event.to_track : event.from_track;
                const transfer = event.type === "possible_pass" || event.type === "possible_turnover";
                return (
                  <li key={event.id}>
                    <span><span className="mono">{secs(event.timestamp_seconds)}</span> · {candidateName(event.type)} · {transfer && partner ? `${gave ? "to" : "from"} ${nameOf(partner)}` : candidatePlayers(event, nameOf)}</span>
                    <span className="row">
                      <WatchButton at={event.start_seconds} label="Watch" describe={`Watch the candidate at ${secs(event.timestamp_seconds)}`} />
                      {partner && byId.has(partner) && !trackIds.includes(partner) && <button type="button" className="btn btn--secondary btn--sm" onClick={() => onSelect(partner)}>Open {partner}</button>}
                    </span>
                  </li>
                );
              })}
            </ul>
            {own.events.length > 12 && <p className="note">Showing the first 12 of {own.events.length}. All of them are listed under Ball.</p>}
          </div>
        )}

        <div className="stack stack--sm">
          <h4>When this player is on screen</h4>
          <ul className="itemlist">
            {trackIds.map(id => {
              const identity = byId.get(id);
              return identity && (
                <li key={id}>
                  <span><span className="mono">{id}</span> · {range(identity.first_seen_ms / 1000, identity.last_seen_ms / 1000, 1)}{identity.fragments > 1 ? ` · joined from ${identity.fragments} tracker fragments` : ""}</span>
                  <span className="row">
                    <WatchButton at={identity.first_seen_ms / 1000} label="Watch" describe={`Watch ${id} from where it is first seen`} />
                    {player && player.tracks.length > 1 && <button type="button" className="btn btn--secondary btn--sm btn--danger" onClick={() => change(players.map(row => row.key === player.key ? { ...row, tracks: row.tracks.filter(track => track !== id) } : row))}>Not this player</button>}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>

        {suggested && (
          <div className="stack stack--sm">
            <h4>Possibly the same player</h4>
            {possible.length ? <>
              <p className="note">
                The tracker may have split this player. Each line is a question, not a finding: watch the moment, and join the two only if you can see it is the same person. Nothing is joined automatically.
                {graded && " Close fits come first. On ten labelled test clips from another league, about half of the close fits that could be checked were the same player, and about one in five of all suggestions."}
              </p>
              {!listed.length && <p className="note">No close fit for this player.</p>}
              <ul className="itemlist">
                {listed.slice(0, 8).map(row => {
                  const mine = trackIds.includes(row.from_track) ? row.from_track : row.to_track, theirs = mine === row.from_track ? row.to_track : row.from_track;
                  const theirOwner = owner(theirs), canJoin = Boolean(player) !== Boolean(theirOwner);
                  const where = row.basis === "pitch" ? `about ${num(row.distance_m)} m away on the pitch map` : `about ${num(row.distance_heights)} player heights away in the image`;
                  return (
                    <li key={`${row.from_track}-${row.to_track}`}>
                      <span>
                        <span className="mono">{row.from_track}</span> is last seen at {secs(row.from_last_seen_seconds)}; <span className="mono">{row.to_track}</span> appears {secs(row.gap_seconds)} later, {where}.
                        {" "}{row.rivals === 0 ? "No other identity fits this gap." : `${row.rivals} other ${row.rivals === 1 ? "identity fits" : "identities fit"} as well, so check carefully.`}
                        {row.same_kit === null ? " Kit group unknown for one of them." : ""}
                      </span>
                      <span className="row">
                        <WatchButton at={Math.max(0, row.from_last_seen_seconds - 2)} label="Watch" describe={`Watch from just before ${row.from_track} is last seen`} />
                        <button type="button" className="btn btn--secondary btn--sm" onClick={() => onSelect(theirs)}>Open {theirs}</button>
                        <button type="button" className="btn btn--secondary btn--sm" disabled={!canJoin || pending || !saved} title={canJoin ? undefined : player ? "Both are already named. Remove one name first." : "Name one of the two first, then join the other to it."} onClick={() => join(row)}>Same player</button>
                      </span>
                    </li>
                  );
                })}
              </ul>
              {listed.length > 8 && <p className="note">Showing the 8 closest of {listed.length}.</p>}
              {looseFits.length > 0 && (looseFor === current
                ? <button type="button" className="btn btn--secondary btn--sm" onClick={() => setLooseFor(null)}>Hide the {looseFits.length} looser {looseFits.length === 1 ? "suggestion" : "suggestions"}</button>
                : <button type="button" className="btn btn--secondary btn--sm" onClick={() => setLooseFor(current)}>Show {looseFits.length} looser {looseFits.length === 1 ? "suggestion" : "suggestions"} (mostly wrong on test clips)</button>)}
            </> : <p className="note">{suggested.reason ?? "No other identity starts or ends close enough in time and place to be suggested."}</p>}
          </div>
        )}

        <Disclosure variant="panel" summary={player ? "Edit this player’s name" : "Name this player"} meta={dirty ? "Unsaved" : undefined} defaultOpen={dirty || Boolean(newNumber)}>
          <fieldset className="stack profile__form" disabled={pending || !saved}>
            <legend className="sr-only">Name for {heading}</legend>
            <p className="muted">Watch the footage first. Type a name only when you can tell who this is. The tracker often splits one player into several identities, so you can add more identities to the same player.{roster.length ? " Names you saved on other clips are offered as you type; picking one does not match anyone automatically." : ""}</p>
            <datalist id={rosterId}>{[...new Set(roster.map(entry => entry.name))].map(name => <option key={name} value={name} />)}</datalist>
            {!lineupRows.length && <p className="note">No line-up is saved for this video. List it under Review to choose players from it here, and to see their approved events beside these figures.</p>}
            {player ? <>
              {lineupRows.length > 0 && (
                <Field label="Who this is, from the line-up">
                  <select value={player.lineupId ?? ""} onChange={event => { const row = lineupRows.find(entry => entry.lineup_id === event.target.value); edit(player.key, row ? { lineupId: row.lineup_id, name: row.name, number: row.number ?? "" } : { lineupId: null }); }}>
                    <option value="">Not in the line-up (type a name)</option>
                    {lineupOptions(player.key)}
                  </select>
                </Field>
              )}
              {player.lineupId && <p className="note">The name and number come from the line-up. Change them under Review.</p>}
              <div className="formgrid formgrid--two">
                <Field label="Player name"><input maxLength={80} list={rosterId} readOnly={Boolean(player.lineupId)} value={player.name} onChange={event => { const known = pickKnown(event.target.value); edit(player.key, known && !player.number ? { name: event.target.value, number: known.number ?? "" } : { name: event.target.value }); }} /></Field>
                <Field label="Shirt number (optional)"><input inputMode="numeric" maxLength={2} readOnly={Boolean(player.lineupId)} value={player.number} onChange={event => edit(player.key, { number: event.target.value })} /></Field>
              </div>
              {unnamed.length > 0 && (
                <Field label="Add another tracked identity to this player">
                  <select value="" onChange={event => { if (event.target.value) edit(player.key, { tracks: [...player.tracks, event.target.value] }); }}>
                    <option value="">Choose an identity…</option>
                    {unnamed.map(identity => <option key={identity.track_id} value={identity.track_id}>{identity.track_id}{identity.team ? ` · ${kits.label(identity.team)}` : ""} · {range(identity.first_seen_ms / 1000, identity.last_seen_ms / 1000, 1)}</option>)}
                  </select>
                </Field>
              )}
              <div><button type="button" className="btn btn--secondary btn--danger" onClick={() => change(players.filter(row => row.key !== player.key))}>Remove this name</button></div>
            </> : current && <>
              {lineupRows.length > 0 && (
                <Field label="Who this is, from the line-up">
                  <select value={newLineup} onChange={event => { const row = lineupRows.find(entry => entry.lineup_id === event.target.value); setNewLineup(row?.lineup_id ?? ""); if (row) { setNewName(row.name); setNewNumber(row.number ?? ""); } }}>
                    <option value="">Not in the line-up (type a name)</option>
                    {lineupOptions(null)}
                  </select>
                </Field>
              )}
              <div className="formgrid formgrid--two">
                <Field label="Player name"><input maxLength={80} list={rosterId} readOnly={Boolean(newLineup)} value={newName} onChange={event => { const known = pickKnown(event.target.value); setNewName(event.target.value); if (known && !newNumber) setNewNumber(known.number ?? ""); }} /></Field>
                <Field label="Shirt number (optional)"><input inputMode="numeric" maxLength={2} readOnly={Boolean(newLineup)} value={newNumber} onChange={event => setNewNumber(event.target.value)} /></Field>
              </div>
              <div className="row">
                <button type="button" className="btn btn--secondary" disabled={!newName.trim()} onClick={() => { change([...players, { key: `new-${nextKey++}`, name: newName, number: newNumber, tracks: [current], lineupId: newLineup || null }]); setNewName(""); setNewNumber(""); setNewLineup(""); }}>Use this name for {current}</button>
              </div>
              {players.length > 0 && (
                <Field label={`Or add ${current} to a player already named`}>
                  <select value="" onChange={event => { if (event.target.value) edit(event.target.value, { tracks: [...(players.find(row => row.key === event.target.value)?.tracks ?? []), current] }); }}>
                    <option value="">Choose a player…</option>
                    {players.map(row => <option key={row.key} value={row.key}>{title(row)}</option>)}
                  </select>
                </Field>
              )}
            </>}
            {error && <p className="notice notice--error" role="alert"><AlertTriangle size={16} aria-hidden="true" /><span>{error}</span></p>}
            {(conflict || (!saved && !pending)) && <div><button type="button" className="btn btn--secondary" onClick={() => setReload(value => value + 1)}>{conflict ? "Reload saved names · discard edits" : "Retry loading names"}</button></div>}
            {dirty && <>
              <label className="check"><input type="checkbox" checked={confirmed} onChange={event => setConfirmed(event.target.checked)} />I checked these names against this video.</label>
              <div className="row">
                <button type="button" className="btn btn--primary" disabled={!confirmed || conflict} onClick={save}>Save player names</button>
                <button type="button" className="btn btn--quiet" onClick={() => { if (saved) change(fromSaved(saved)); }}>Discard changes</button>
              </div>
              <p className="note">Names are not saved until you press Save. Leaving this clip discards unsaved names.</p>
            </>}
            {saved && !dirty && saved.revision > 0 && <p className="note" role="status">Saved on the server as version {saved.revision}.{saved.overlap_checked === false ? " The check that two identities are never on screen together could not run for this clip." : ""}</p>}
          </fieldset>
        </Disclosure>

        <div><button type="button" className="btn btn--secondary btn--sm" onClick={download}><ArrowDownToLine size={14} aria-hidden="true" /> This player’s figures (CSV)</button></div>

        <Disclosure summary="How this is measured">
          <p>A profile adds up the tracked identities you selected. Identities are joined automatically from tracker fragments and are unreviewed, so one identity can contain two players and one player can be several identities. Naming does not change any automatic figure; it only groups identities. Saving is refused when two identities given to one player are on screen in the same analysed frames.</p>
          <p>Distance, speed, speed bands and the heat map are estimates from automatic pitch calibration on an assumed 105 x 68 m pitch. They count only time tracked in a calibrated frame and are not match totals. On ten labelled test clips from another league, positions were off by a median of about 8 m, distances came out about 1.6 times the labelled ones, and time at fast-running speed was several times too high, so treat them as rough.</p>
          <p>“Possibly the same player” lists identities where one ends and another begins soon after, close enough for one person to have covered the ground, and never two that are on screen together. It uses timing and position only, so team-mates crossing paths can be suggested wrongly.</p>
        </Disclosure>
      </div>
    </Panel>
  );
}
