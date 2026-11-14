import { ArrowDownToLine } from "lucide-react";
import { AnalyticsDashboard } from "../AnalyticsDashboard";
import { analyticsAreas } from "../analytics-requirements";
import { VideoInsightsDownloads } from "../VideoInsights";
import type { ClipTabProps } from "../Workspace";
import { Disclosure, Panel } from "../ui";

const REQUIRED_METRICS = analyticsAreas.reduce((total, area) => total + area.metrics.length, 0);

export function FilesTab({ result, saved, goTab, media }: ClipTabProps & { media: (path: string) => string }) {
  const insights = result.metrics.video_insights;
  const files = [
    { label: "Annotated video", description: "Full-duration MP4 with unreviewed model detections.", path: result.files.annotated_video, type: "MP4" },
    { label: "Tracking observations", description: "Timestamped raw model observations and temporary track identities.", path: result.files.tracking_json, type: "JSON" },
    { label: "Measurement data", description: "Computed model measurements with their scope and definitions.", path: result.files.metrics_json, type: "JSON" },
    { label: "Full analysis", description: "Analysis manifest, model measurements, limits and provenance.", path: result.files.manifest_json, type: "JSON" },
  ];

  function downloadReview() {
    if (!saved) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(saved, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${saved.run_id}-saved-review-revision-${saved.revision}.json`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="tabstack">
      <Panel level={2} title="Files" subtitle="Take the evidence with you. Model output and reviewer-approved records are separate files.">
        <ul className="files">
          {files.map(item => (
            <li key={item.label}>
              <span className="files__type">{item.type}</span>
              <div><h4>{item.label}</h4><p>{item.description}</p></div>
              {item.path
                ? <a className="btn btn--secondary btn--sm" href={media(item.path)} target="_blank" rel="noreferrer" aria-label={`Download ${item.label}`}><ArrowDownToLine size={14} aria-hidden="true" /> Download</a>
                : <span className="note">Unavailable in this run</span>}
            </li>
          ))}
          <li>
            <span className="files__type files__type--reviewed">JSON</span>
            <div><h4>Saved human review</h4><p>Last server-saved revision {saved?.revision ?? "—"}. Includes approval states and the saved summary; excludes unsaved edits and unfinished forms.</p></div>
            <button type="button" className="btn btn--secondary btn--sm" onClick={downloadReview} disabled={!saved}><ArrowDownToLine size={14} aria-hidden="true" /> Saved review JSON</button>
          </li>
          {insights && (
            <li>
              <span className="files__type">DATA</span>
              <div><h4>Automatic video insights</h4><p>Observed ball paths, kit shape windows, possession estimates and possible events. Unreviewed, bound to this clip. These are automatic estimates, not approved football records.</p></div>
              <VideoInsightsDownloads insights={insights} runId={result.run_id} />
            </li>
          )}
        </ul>
        <p className="note">Downloads use the existing protected or signed file paths. Missing files are shown as unavailable.</p>
      </Panel>

      <Disclosure variant="panel" summary="Analytics scope" meta={`${analyticsAreas.length} areas · ${REQUIRED_METRICS} required metrics`}>
        <AnalyticsDashboard filename={result.metadata.filename} hasDetections status="complete" onReview={() => goTab("review")} />
      </Disclosure>
    </div>
  );
}
