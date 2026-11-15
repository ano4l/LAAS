import { SoccerTrackShape } from "../SoccerTrackShape";
import { InsightDefinitions, KitEvolution } from "../VideoInsights";
import { ImageOccupancy } from "../VideoMeasurements";
import type { ClipTabProps } from "../Workspace";
import { Disclosure, EmptyState } from "../ui";

export function ShapeTab({ result, media }: ClipTabProps & { media: (path: string) => string }) {
  const insights = result.metrics.video_insights;
  const m = result.metrics.video_measurements;

  return (
    <div className="tabstack">
      <header className="tabhead">
        <div>
          <h2>Shape</h2>
          <p>How the visible players are arranged in the camera image. These are image measurements: not a pitch map, not distances, not a formation.</p>
        </div>
      </header>

      {insights ? <>
        <SoccerTrackShape data={insights.collective_dynamics} imageWidth={insights.source_video?.width} imageHeight={insights.source_video?.height} duration={insights.source_duration_seconds} videoSrc={result.files.annotated_video ? media(result.files.annotated_video) : undefined} />
        <KitEvolution data={insights} />
      </> : <EmptyState title="No shape measurements for this clip">Extended video insights are not available in this run. No player positions or formation are assumed.</EmptyState>}

      {m
        ? <ImageOccupancy measurements={m} />
        : <EmptyState title="No image occupancy for this clip">Extended camera-image measurements are not present in this older run.</EmptyState>}

      {insights && (
        <Disclosure variant="panel" summary="How this is measured" meta="Definitions and limits">
          <InsightDefinitions data={insights} only={["shape", "zones", "confidence"]} />
        </Disclosure>
      )}
    </div>
  );
}
