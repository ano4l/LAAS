import { AnalyticsDashboard } from "../AnalyticsDashboard";
import type { ClipTabProps } from "../Workspace";
import { Panel } from "../ui";

/** Every requested analytic in one place: a figure where approved records supply one, a reason where they cannot. */
export function AnalyticsTab({ result, saved, goTab, reviewUnsaved }: ClipTabProps & { reviewUnsaved: boolean }) {
  return (
    <div className="tabstack">
      <header className="tabhead">
        <div>
          <h2>Analytics</h2>
          <p>The football analytics this product was asked for. Each figure is either counted from events a person approved, or the model’s own estimate from the video, and says which.</p>
        </div>
      </header>
      <Panel title="Requested analytics" subtitle="Match, team, opponent, player, scouting, workload and report analytics, with what each one still needs.">
        <AnalyticsDashboard result={result} saved={saved} unsaved={reviewUnsaved} onReview={() => goTab("review")} />
      </Panel>
    </div>
  );
}
