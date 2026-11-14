import { useState } from "react";
import type { AnalysisResult } from "../main";
import { MeasurementDefinitions, ProvisionalKitPanel } from "../VideoMeasurements";
import type { ClipTabProps } from "../Workspace";
import { DASH, num, pct, secs } from "../format";
import { TeamName } from "../kits";
import { DataTable, Disclosure, EmptyState, Field, Pager, Panel, StatTile, WatchButton } from "../ui";

export function PlayersTab({ result }: ClipTabProps) {
  const summary = result.detection_summary;
  const m = result.metrics.video_measurements;
  const quality = result.metrics.video_insights?.quality;
  const linking = summary?.track_linking;

  return (
    <div className="tabstack">
      <header className="tabhead">
        <div>
          <h2>Players</h2>
          <p>Everything the tracker followed. A track is a moving box the model kept hold of, not a verified player.</p>
        </div>
      </header>

      <div className="tiles">
        <StatTile
          label="Players tracked"
          value={summary ? num(linking?.identities_after ?? summary.unique_tracks, 0) : DASH}
          evidence={summary ? "seen" : "unknown"}
          title={linking ? "Technical name: linked identities" : "Technical name: temporary tracks"}
          context={linking ? `Linked identities, joined from ${num(linking.tracks_before, 0)} tracker tracks (${num(linking.links_accepted, 0)} links)` : "Temporary tracks · IDs can split or change"}
        />
        <StatTile
          label="Players in view, on average"
          value={m?.mean_visible_players == null ? DASH : num(m.mean_visible_players)}
          evidence={m?.mean_visible_players == null ? "unknown" : "seen"}
          title="Technical name: mean visible players. Detections per analysed frame, not unique verified players"
          context={m ? `Most in one frame: ${num(m.max_visible_players, 0)}` : "Not measured in this analysis"}
        />
        <StatTile
          label="Sorted into a kit group"
          value={quality ? num(quality.kit_assignment_coverage_pct) : DASH}
          unit="%"
          evidence={quality?.kit_assignment_coverage_pct == null ? "unknown" : "estimated"}
          title="Technical name: kit assignment coverage"
          context={quality ? `${num(quality.grouped_player_observations, 0)} of ${num(quality.player_observations, 0)} player observations` : "Not measured in this analysis"}
        />
      </div>

      <TrackExplorer result={result} />

      {m && (m.provisional_kit_analytics !== undefined || m.event_candidates !== undefined)
        ? <ProvisionalKitPanel measurements={m} />
        : <EmptyState title="No kit groups for this clip">{m ? "This analysis did not record kit observations." : "Extended camera-image measurements are not present in this older run. Its available detections can still be explored above and downloaded under Files."}</EmptyState>}
    </div>
  );
}

function TrackExplorer({ result }: { result: AnalysisResult }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("observations");
  const [page, setPage] = useState(0);
  const m = result.metrics.video_measurements;
  const measured = new Map(m?.tracks.map(track => [track.track_id, track]) ?? []);
  const linking = result.detection_summary?.track_linking;
  const linked = result.detection_summary?.linked_tracks;
  // Linked identities join tracker fragments offline; measurements for such runs are keyed by the same L… ids.
  const source = linked
    ? linked.map(t => ({ track_id: t.linked_track_id, object_type: t.object_type, observations: t.observations, first_seen_ms: t.first_seen_ms, last_seen_ms: t.last_seen_ms, fragments: t.track_ids.length, team: t.team }))
    : (result.detection_summary?.tracks ?? []).map(t => ({ ...t, fragments: 1, team: null as string | null }));
  const matches = source
    .filter(track => `${track.track_id} ${track.object_type} ${track.team ?? ""}`.toLowerCase().includes(query.trim().toLowerCase()))
    .sort((a, b) => sort === "first" ? a.first_seen_ms - b.first_seen_ms : sort === "near" ? (measured.get(b.track_id)?.near_ball_frames ?? 0) - (measured.get(a.track_id)?.near_ball_frames ?? 0) : b.observations - a.observations);
  const totalPages = Math.max(1, Math.ceil(matches.length / 8));
  const current = Math.min(page, totalPages - 1);

  return (
    <Panel
      title={linked ? "Tracked identities" : "Tracks"}
      subtitle={linked && linking
        ? `${linking.identities_after} identities joined automatically from ${linking.tracks_before} tracker tracks. Links are unreviewed, can join different players, and do not identify verified players.`
        : "Track IDs can split or change. They do not identify verified players."}
    >
      <div className="toolbar">
        <Field label="Search tracks"><input type="search" value={query} onChange={event => { setQuery(event.target.value); setPage(0); }} placeholder="ID, class or kit group" /></Field>
        <Field label="Sort tracks" className="field--fixed">
          <select value={sort} onChange={event => { setSort(event.target.value); setPage(0); }}>
            <option value="observations">Most observations</option>
            <option value="first">First seen</option>
            <option value="near">Near-ball frames</option>
          </select>
        </Field>
      </div>
      {matches.length > 0 ? (
        <DataTable label="Temporary track observations table" stack>
          <thead><tr><th scope="col">Track / class</th><th scope="col" className="num">Observed</th><th scope="col" className="num">Visible</th><th scope="col" className="num">Image movement</th><th scope="col" className="num">Near-ball frames</th><th scope="col">First seen</th></tr></thead>
          <tbody>
            {matches.slice(current * 8, (current + 1) * 8).map(track => {
              const detail = measured.get(track.track_id);
              return (
                <tr key={track.track_id}>
                  <th scope="row">
                    <span className="mono">{track.track_id}</span>
                    <small>{track.object_type}{linked ? ` · ${track.fragments} ${track.fragments === 1 ? "fragment" : "fragments"}` : ""}</small>
                    {track.team && <small><TeamName id={track.team} /></small>}
                  </th>
                  <td className="num" data-label="Observed">{detail ? secs(detail.observed_seconds) : `${track.observations} frames`}</td>
                  <td className="num" data-label="Visible">{detail?.visibility_pct == null ? DASH : pct(detail.visibility_pct)}</td>
                  <td className="num" data-label="Image movement" title="Technical name: image displacement, in pixels">{detail ? `${num(detail.image_displacement_px)} px` : DASH}</td>
                  <td className="num" data-label="Near-ball frames">{num(detail?.near_ball_frames, 0)}</td>
                  <td className="actions"><WatchButton at={track.first_seen_ms / 1000} describe={`Watch track ${track.track_id}`} /></td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      ) : <p className="note" role="status">No tracks match this search.</p>}
      <Pager page={current} pages={totalPages} count={matches.length} noun="matching tracks" onPage={setPage} />
      <Disclosure summary="How this is measured">
        <p>Observed seconds estimate sampled visible intervals, not continuous play. Image movement (displacement) is in pixels and includes camera motion and tracking error; it is not distance run or speed on the pitch. Near-ball frames are proximity observations, not possession or touches. Full detections remain available under Files.</p>
        {m && <MeasurementDefinitions measurements={m} only={["visible_players", "movement", "visibility", "proximity"]} />}
      </Disclosure>
    </Panel>
  );
}
