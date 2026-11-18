import { BallJourney, CandidateQueue, InsightDefinitions } from "../VideoInsights";
import { BallVisibility, MeasurementDefinitions } from "../VideoMeasurements";
import type { ClipTabProps } from "../Workspace";
import { DASH, num, pct, secs } from "../format";
import { Disclosure, EmptyState, StatTile } from "../ui";
import { proposeReviewEvent } from "../ReviewPanel";

export function BallTab({ result, goTab }: ClipTabProps) {
  const insights = result.metrics.video_insights;
  const m = result.metrics.video_measurements;
  const selectedKnown = m && (m.selected_ball_candidate_frames !== undefined || m.selected_ball_candidate_coverage_pct !== undefined);

  return (
    <div className="tabstack">
      <header className="tabhead">
        <div>
          <h2>Ball</h2>
          <p>Where the ball was seen, how often, and the moments worth a second look. Missing positions are never filled in.</p>
        </div>
      </header>

      <div className="tiles tiles--four">
        <StatTile
          label="Ball seen"
          value={insights ? num(insights.quality.selected_ball_coverage_pct) : DASH}
          unit="%"
          evidence={insights ? "seen" : "unknown"}
          title={`Technical name: eligible selected-ball evidence${insights ? `, confidence ≥ ${insights.parameters.min_ball_confidence ?? .25}` : ""}`}
          context={insights ? `${secs(insights.quality.selected_ball_seconds)} of the full clip · confidence ≥ ${insights.parameters.min_ball_confidence ?? .25}` : "Not measured in this analysis"}
        />
        <StatTile
          label="Ball detected"
          value={m?.ball_coverage_pct == null ? DASH : num(m.ball_coverage_pct)}
          unit="%"
          evidence={m?.ball_coverage_pct == null ? "unknown" : "seen"}
          title="Technical name: ball candidate coverage. Observed candidates, not an accuracy score"
          context={m ? `${num(m.ball_detected_frames, 0)} frames · ${num(m.ambiguous_ball_frames, 0)} with more than one candidate` : "Not measured in this analysis"}
        />
        <StatTile
          label="A single ball picked out"
          value={selectedKnown && m.selected_ball_candidate_coverage_pct != null ? num(m.selected_ball_candidate_coverage_pct) : DASH}
          unit="%"
          evidence={selectedKnown && m.selected_ball_candidate_coverage_pct != null ? "estimated" : "unknown"}
          title="Technical name: selected ball candidate coverage. One observed candidate retained after confidence and continuity checks; not a verified detection or an accuracy score"
          context={selectedKnown ? (m.selected_ball_candidate_frames === undefined ? "Observed frame count unknown" : `${num(m.selected_ball_candidate_frames, 0)} frames with one candidate kept after continuity checks`) : "Not measured in this analysis"}
        />
        <StatTile
          label="Longest gap without the ball"
          value={m ? num(m.longest_ball_detection_gap_seconds) : DASH}
          unit="s"
          evidence={m ? "seen" : "unknown"}
          title="Technical name: longest ball detection gap, across analysed intervals"
          context="Longest run of frames with no ball candidate"
        />
      </div>

      {insights
        ? <BallJourney data={insights} />
        : <EmptyState title="No ball path for this clip">Extended video insights are not available in this run, so no ball path or possible events were worked out.</EmptyState>}

      {m
        ? <BallVisibility measurements={m} />
        : <EmptyState title="No frame-by-frame ball measurements">Extended camera-image measurements are not present in this older run. Its available detections can still be downloaded under Files.</EmptyState>}

      {insights && <CandidateQueue data={insights} onDraft={draft => { proposeReviewEvent(result.run_id, draft); goTab("review"); }} />}

      {(insights || m) && (
        <Disclosure variant="panel" summary="How this is measured" meta="Definitions and limits">
          {insights && <>
            <p>
              “Ball seen” (eligible selected-ball evidence) requires one selected observation with detection confidence ≥ {insights.parameters.min_ball_confidence ?? .25}. “Ball detected” (raw candidate coverage) can include ambiguous or
              lower-confidence observations; the measures use different eligibility rules.
            </p>
            <InsightDefinitions data={insights} only={["zones", "events", "actions", "confidence"]} />
            <p>
              Ball path display: {insights.ball.path.length} points from {insights.ball.path_observations} observed positions, display stride {insights.ball.path_display_stride}. Segment IDs preserve gaps and scene boundaries.
              Confidence and support are not measured accuracy.
            </p>
          </>}
          {m && <>
            <p>
              <strong>Ball proximity is not possession or touches. </strong>
              {num(m.ball_proximity_frames, 0)} frames meet the proximity rule ({pct(m.proximity_association_coverage_pct)} of analysed frames). Mean ball confidence: {m.mean_ball_confidence === null ? "Unknown" : pct(m.mean_ball_confidence * 100)}.
              Model confidence is not measured accuracy.
            </p>
            <MeasurementDefinitions measurements={m} only={["selected_ball", "proximity", "confidence", "event_candidates"]} />
          </>}
        </Disclosure>
      )}
    </div>
  );
}
