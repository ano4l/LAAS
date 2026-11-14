import { SequenceContextPanel } from "../SequenceContextPanel";
import { TemporalPossession } from "../TemporalPossession";
import { ControlEstimates, InsightDefinitions } from "../VideoInsights";
import type { ClipTabProps } from "../Workspace";
import { secs } from "../format";
import { Disclosure, EmptyState, Panel } from "../ui";

type Props = ClipTabProps & { api: string; authHeaders: () => Record<string, string>; onRefreshResult: (runId: string) => Promise<void> };

export function PossessionTab({ result, api, authHeaders, onRefreshResult }: Props) {
  const insights = result.metrics.video_insights;
  // Every kit group the run knows about can be given a confirmed team name.
  const kitIds = [...new Set([...(insights?.kits.map(kit => kit.kit) ?? []), ...(result.metrics.video_measurements?.provisional_kit_analytics?.map(kit => kit.kit) ?? [])])];

  return (
    <div className="tabstack">
      <Panel level={2} title="Possession" subtitle="Which team had the ball, for how long, and how sure the model can be. Unknown time is never shared out between teams.">
        {insights
          ? <TemporalPossession sequences={insights.sequences} />
          : <EmptyState title="No possession estimate for this clip">Extended video insights are not available in this run. No control or event estimates are assumed for missing data.</EmptyState>}
      </Panel>

      {insights && (
        <Disclosure variant="panel" summary="Strict near-foot observations" meta={`${secs(insights.control.estimated_control_seconds)} supported · separate rule, not team possession`}>
          <ControlEstimates data={insights} />
          <Disclosure summary="How this is measured">
            <InsightDefinitions data={insights} only={["control", "shares", "confidence"]} />
          </Disclosure>
        </Disclosure>
      )}

      <SequenceContextPanel runId={result.run_id} sourceSha={result.metadata.sha256} duration={result.metadata.duration_seconds} kits={kitIds} api={api} authHeaders={authHeaders} onRefresh={onRefreshResult} />
    </div>
  );
}
