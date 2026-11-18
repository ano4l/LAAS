import { useMemo, useState } from "react";
import { AlertTriangle, ArrowRight, BarChart3, CircleDot, ClipboardCheck, FolderDown, Play, Shapes, Timer, Users, type LucideIcon } from "lucide-react";
import { PossessionOverview, possessionTakeaway } from "../TemporalPossession";
import { InsightDefinitions } from "../VideoInsights";
import { MeasurementDefinitions } from "../VideoMeasurements";
import type { ClipTabProps, TabId } from "../Workspace";
import { plainBookmark, plainFindings } from "../findings";
import { DASH, num, pct, range, secs } from "../format";
import { useKits } from "../kits";
import { usePlayback } from "../playback";
import { Disclosure, EmptyState, EvidenceLegend, Panel, StatTile, WatchButton } from "../ui";
import { segmentKind } from "../vocabulary";

const FINDING_TAB: Record<string, { tab: TabId; label: string; icon: LucideIcon }> = {
  coverage: { tab: "possession", label: "Possession", icon: Timer },
  possession: { tab: "possession", label: "Possession", icon: Timer },
  limitation: { tab: "possession", label: "Possession", icon: AlertTriangle },
  events: { tab: "ball", label: "Ball", icon: CircleDot },
  shape: { tab: "shape", label: "Shape", icon: Shapes },
};

const EXPLORE: { tab: TabId; title: string; text: string; icon: LucideIcon }[] = [
  { tab: "possession", title: "Possession", text: "Every interval, the split between teams, and how it is measured.", icon: Timer },
  { tab: "ball", title: "Ball", text: "The ball's path through the image and possible events to review.", icon: CircleDot },
  { tab: "shape", title: "Shape", text: "How each group spreads and moves, and where players were seen.", icon: Shapes },
  { tab: "players", title: "Players", text: "Every tracked identity and the provisional kit groups.", icon: Users },
  { tab: "analytics", title: "Analytics", text: "Every requested analytic: figures from approved events, and what the rest still need.", icon: BarChart3 },
  { tab: "review", title: "Review", text: "Add and approve your own event records against the footage.", icon: ClipboardCheck },
  { tab: "files", title: "Files", text: "Annotated video, model data, saved review and exports.", icon: FolderDown },
];

export function SummaryTab({ result, saved, goTab, reviewUnsaved }: ClipTabProps & { reviewUnsaved: boolean }) {
  const kits = useKits();
  const playback = usePlayback();
  const insights = result.metrics.video_insights;
  const m = result.metrics.video_measurements;
  const summary = result.detection_summary;
  const s = insights?.sequences;
  const linking = summary?.track_linking;
  // The staggered reveal plays once, the first time Summary is on screen, not on every return to the tab.
  const [reveal, setReveal] = useState(true);
  const firstSupported = s?.intervals.find(interval => segmentKind(interval.state, interval.team) === "supported");
  const findings = useMemo(() => insights ? plainFindings(insights, kits) : [], [insights, kits]);
  const bookmarks = insights?.bookmarks ?? [];

  return (
    <div className="tabstack">
      <Panel
        level={2}
        title="Who had the ball"
        subtitle={insights ? possessionTakeaway(s) : "Extended video insights are not available in this run. No control or event estimates are assumed for missing data."}
        aside={<button type="button" className="btn btn--secondary btn--sm" onClick={() => goTab("possession")}>Possession details <ArrowRight size={14} aria-hidden="true" /></button>}
      >
        {insights ? <>
          <PossessionOverview sequences={s} />
          <p className="row note">
            <span>When it happened is on the timeline under the video: colour is the team, hatching is assumed, dark is unknown.</span>
            {firstSupported && <WatchButton at={firstSupported.start_seconds} label="Watch the first supported spell" />}
          </p>
        </> : <EmptyState title="No possession estimate for this clip">You can still inspect the available observations in the other tabs and review the footage.</EmptyState>}
      </Panel>

      <div className={`tiles ${reveal ? "tiles--reveal" : ""}`} onAnimationEnd={event => { if (event.target === event.currentTarget.lastElementChild) setReveal(false); }}>
        {s || !insights ? (
          <StatTile label="Possession we can support" value={s ? num(s.coverage_pct) : DASH} unit="%" evidence={s ? "estimated" : "unknown"} title="Technical name: classified known time as a share of non-dead time" context={s ? `${secs(s.known_seconds)} of ${secs(s.classified_non_dead_seconds)} playing time` : "No estimate in this analysis"} onClick={() => goTab("possession")} actionLabel="Possession we can support. Open the Possession tab" />
        ) : (
          <StatTile label="Near-foot control we can support" value={num(insights.control.coverage_pct)} unit="%" evidence="estimated" title="Technical name: strict near-foot control estimate" context={`${secs(insights.control.estimated_control_seconds)} of the full clip · not team possession`} onClick={() => goTab("possession")} actionLabel="Near-foot control. Open the Possession tab" />
        )}
        <StatTile
          label="Including assumed possession between touches"
          value={s?.retention ? num(s.retention.coverage_with_retention_pct) : DASH}
          unit="%"
          evidence={s?.retention ? "assumed" : "unknown"}
          title="Technical name: with possession retained between touches (inferred)"
          context={s?.retention ? (s.retention.retained_seconds > 0 ? `Adds ${secs(s.retention.retained_seconds)} kept between a team's own touches` : "No time could be added by this assumption") : "Not calculated for this analysis"}
        />
        {insights ? (
          <StatTile label="Ball seen" value={num(insights.quality.selected_ball_coverage_pct)} unit="%" evidence="seen" title={`Technical name: eligible selected-ball evidence, confidence ≥ ${insights.parameters.min_ball_confidence ?? .25}`} context={`${secs(insights.quality.selected_ball_seconds)} of the clip${m?.ball_coverage_pct != null ? ` · detected in ${pct(m.ball_coverage_pct)} of frames` : ""}`} onClick={() => goTab("ball")} actionLabel="Ball seen. Open the Ball tab" />
        ) : (
          <StatTile label="Ball detected" value={m?.ball_coverage_pct == null ? DASH : num(m.ball_coverage_pct)} unit="%" evidence={m?.ball_coverage_pct == null ? "unknown" : "seen"} title="Technical name: ball candidate coverage" context="Frames with a ball candidate · not accuracy" onClick={() => goTab("ball")} actionLabel="Ball detected. Open the Ball tab" />
        )}
        <StatTile
          label="Players tracked"
          value={summary ? num(linking?.identities_after ?? summary.unique_tracks, 0) : DASH}
          evidence={summary ? "seen" : "unknown"}
          title={linking ? "Technical name: linked identities" : "Technical name: temporary tracks"}
          context={linking ? `Joined from ${num(linking.tracks_before, 0)} tracker tracks · IDs are not verified players` : "Track IDs can split or change · not verified players"}
          onClick={() => goTab("players")}
          actionLabel="Players tracked. Open the Players tab"
        />
        <StatTile label="Frames analysed" value={num(summary?.analyzed_frames, 0)} evidence={summary ? "seen" : "unknown"} context={m?.analyzed_frame_pct == null ? "Source coverage not recorded" : `${pct(m.analyzed_frame_pct)} of source frames${m.mean_visible_players != null ? ` · ${num(m.mean_visible_players)} players in view on average` : ""}`} />
        <StatTile
          label="Approved human records"
          value={saved ? (saved.summary.approved_events > 0 ? num(saved.summary.approved_events, 0) : "None") : "Loading…"}
          evidence="reviewed"
          context="Saved human annotations only"
          onClick={() => goTab("review")}
          actionLabel="Approved human records. Open the Review tab"
        />
      </div>

      {insights && (
        <Panel title="What this clip can and cannot tell you" subtitle="The analysis's own findings, put in plain words from the same measurements.">
          <ul className="findings">
            {findings.map((finding, index) => {
              const link = FINDING_TAB[finding.kind];
              const Icon = link?.icon ?? AlertTriangle;
              return (
                <li key={`${finding.kind}-${index}`} title={finding.composed ? `Analysis wording: ${finding.original}` : undefined}>
                  <Icon size={16} aria-hidden="true" />
                  <span>{finding.text}</span>
                  {link && <button type="button" className="btn btn--quiet btn--sm" onClick={() => goTab(link.tab)} aria-label={`Open the ${link.label} tab`}>{link.label} <ArrowRight size={13} aria-hidden="true" /></button>}
                </li>
              );
            })}
          </ul>
          {!findings.length && <p className="note">The analysis recorded no findings for this clip.</p>}
          {findings.length > 0 && (
            <Disclosure summary="The analysis's own wording" meta="Unedited">
              <ul>{insights.findings.map((finding, index) => <li key={index}>{finding.text}</li>)}</ul>
              {Boolean(insights.bookmarks?.length) && <ul>{insights.bookmarks!.map(bookmark => <li key={bookmark.id}>{bookmark.label}: {bookmark.detail}</li>)}</ul>}
              <p>These are the sentences the analysis wrote, with its own terms and group ids. The list above states the same measurements using this screen's vocabulary and rounding.</p>
            </Disclosure>
          )}
        </Panel>
      )}

      {insights && bookmarks.length > 0 && (
        <Panel title="Evidence bookmarks" subtitle="Moments the analysis marked as worth watching." aside={<span className="chip chip--plain">{bookmarks.length}</span>}>
          <ul className="rowlist">
            {bookmarks.slice(0, 12).map(bookmark => {
              const plain = plainBookmark(bookmark, insights, kits);
              return (
                <li key={bookmark.id}>
                  <button type="button" className="rowlink" onClick={() => playback.seek(bookmark.start_seconds)} title={plain.composed ? `Analysis wording: ${plain.original}` : undefined}>
                    <Play size={13} fill="currentColor" aria-hidden="true" />
                    <span><strong>{plain.label}</strong><small><span className="mono">{range(bookmark.start_seconds, bookmark.end_seconds, 1)}</span> · {plain.detail}</small></span>
                    <ArrowRight size={15} aria-hidden="true" />
                  </button>
                </li>
              );
            })}
          </ul>
          {bookmarks.length > 12 && <p className="note">Showing the first 12 bookmarks. All bookmarks are included in Insights JSON.</p>}
        </Panel>
      )}

      <div className="note note--boxed note--flood row row--between">
        <span>
          <strong>Human review. </strong>
          {saved
            ? saved.summary.approved_events > 0
              ? `${saved.summary.approved_events} saved approved events. Reviewed coverage: ${secs(saved.summary.reviewed_seconds)} (${pct(saved.summary.reviewed_video_pct)} of clip).`
              : "No saved approved events yet. Passes, shots and possession are not inferred from detection counts."
            : "Loading saved review records…"}
          {reviewUnsaved && " Unsaved annotations and unfinished forms are excluded."}
        </span>
        <button type="button" className="btn btn--secondary btn--sm" onClick={() => goTab("review")}>{saved && !saved.summary.approved_events ? "Start event review" : "Review footage"} <ArrowRight size={14} aria-hidden="true" /></button>
      </div>

      <nav className="explore" aria-label="More about this clip">
        {EXPLORE.map(({ tab, title, text, icon: Icon }) => (
          <button type="button" key={tab} className="explore__link" onClick={() => goTab(tab)}>
            <Icon size={18} aria-hidden="true" />
            <span><strong>{title}</strong><small>{text}</small></span>
            <ArrowRight size={15} aria-hidden="true" />
          </button>
        ))}
      </nav>

      <Panel title="How sure is each number" subtitle="Every figure and chart uses one of five evidence levels. Texture and label, never colour alone.">
        <EvidenceLegend inline />
        {(insights || m) && (
          <Disclosure summary="All definitions and parameters">
            {insights && <>
              <p>
                “Ball seen” (eligible selected-ball evidence) requires one selected observation with detection confidence ≥ {insights.parameters.min_ball_confidence ?? .25}. The broader “ball detected” figure (raw candidate coverage)
                can include ambiguous or lower-confidence observations; the two measures use different eligibility rules.
              </p>
              <InsightDefinitions data={insights} parameters />
              <p>
                Ball path display: {insights.ball.path.length} points from {insights.ball.path_observations} observed positions, display stride {insights.ball.path_display_stride}. Segment IDs preserve gaps and scene boundaries.
                Confidence and support are not measured accuracy.
              </p>
            </>}
            {m && <div><h5>Measurement definitions</h5><MeasurementDefinitions measurements={m} /></div>}
          </Disclosure>
        )}
      </Panel>
    </div>
  );
}
