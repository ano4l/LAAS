import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, CircleDot, ClipboardCheck, FolderDown, LayoutDashboard, Rewind, FastForward, Shapes, Timer, Users } from "lucide-react";
import type { AnalysisResult } from "./main";
import type { Review } from "./ReviewPanel";
import { Timeline } from "./Timeline";
import { clock, loose, num } from "./format";
import { KitProvider, createKitNamer } from "./kits";
import { PlaybackProvider, usePlayback } from "./playback";
import { Disclosure, TabPanel, Tabs, type TabDef } from "./ui";
import { SummaryTab } from "./tabs/SummaryTab";
import { PossessionTab } from "./tabs/PossessionTab";
import { BallTab } from "./tabs/BallTab";
import { ShapeTab } from "./tabs/ShapeTab";
import { PlayersTab } from "./tabs/PlayersTab";
import { ReviewTab } from "./tabs/ReviewTab";
import { AnalyticsTab } from "./tabs/AnalyticsTab";
import { FilesTab } from "./tabs/FilesTab";

export type TabId = "summary" | "possession" | "ball" | "shape" | "players" | "analytics" | "review" | "files";

const TAB_IDS: TabId[] = ["summary", "possession", "ball", "shape", "players", "analytics", "review", "files"];
const readHash = (): TabId => {
  const hash = window.location.hash.slice(1) as TabId;
  return TAB_IDS.includes(hash) ? hash : "summary";
};

type Props = {
  result: AnalysisResult;
  /** False while the library is shown in front of the clip. */
  active: boolean;
  /** Bumped when the job banner asks to open a result; returns to Summary. */
  openRequest: number;
  api: string;
  authHeaders: () => Record<string, string>;
  onRefreshResult: (runId: string) => Promise<void>;
};

export function Workspace(props: Props) {
  const { result } = props;
  const kits = useMemo(() => createKitNamer(result), [result]);
  return (
    <PlaybackProvider duration={result.metadata.duration_seconds} clipKey={result.run_id}>
      <KitProvider value={kits}>
        <ClipWorkspace {...props} />
      </KitProvider>
    </PlaybackProvider>
  );
}

function ClipWorkspace({ result, active, openRequest, api, authHeaders, onRefreshResult }: Props) {
  const playback = usePlayback();
  const [tab, setTabState] = useState<TabId>(readHash);
  const [savedReview, setSavedReview] = useState<Review | null>(null);
  const [reviewUnsaved, setReviewUnsaved] = useState(false);
  const analysis = useRef<HTMLDivElement>(null);
  const saved = savedReview?.run_id === result.run_id ? savedReview : null;
  const summary = result.detection_summary;
  const insights = result.metrics.video_insights;
  const media = useCallback((path: string) => /^https?:\/\//.test(path) ? path : `${api}${path}`, [api]);

  // The open tab lives in the URL hash, so a reload keeps your place.
  const goTab = useCallback((next: TabId) => {
    setTabState(next);
    if (window.location.hash !== `#${next}`) window.history.replaceState(null, "", `#${next}`);
    // Bring the top of the new tab into view if the reader had scrolled past it. The scroll margin in
    // the stylesheet accounts for whatever is stuck above it (app bar, or the video on a narrow screen).
    window.requestAnimationFrame(() => {
      const element = analysis.current;
      if (!element) return;
      const stuckAbove = parseFloat(window.getComputedStyle(element).scrollMarginTop) || 0;
      if (element.getBoundingClientRect().top < stuckAbove - 1) element.scrollIntoView({ block: "start" });
    });
  }, []);

  // On a narrow screen the tab bar sticks directly under the sticky video, so CSS needs the video's height.
  const clip = useRef<HTMLDivElement>(null);
  const player = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const frame = player.current;
    const host = clip.current;
    if (!frame || !host) return;
    const publish = () => host.style.setProperty("--player-h", `${Math.round(frame.getBoundingClientRect().height)}px`);
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onHash = () => setTabState(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const firstOpenRequest = useRef(openRequest);
  useEffect(() => { if (openRequest !== firstOpenRequest.current) goTab("summary"); }, [openRequest, goTab]);
  useEffect(() => { if (!active) playback.pause(); }, [active, playback]);

  const tabs: TabDef<TabId>[] = [
    { id: "summary", label: "Summary", icon: LayoutDashboard },
    { id: "possession", label: "Possession", icon: Timer },
    { id: "ball", label: "Ball", icon: CircleDot, count: insights?.candidates.length ? insights.candidates.length : undefined },
    { id: "shape", label: "Shape", icon: Shapes },
    { id: "players", label: "Players", icon: Users },
    { id: "analytics", label: "Analytics", icon: BarChart3 },
    { id: "review", label: "Review", icon: ClipboardCheck, count: saved?.summary.approved_events ? saved.summary.approved_events : undefined },
    { id: "files", label: "Files", icon: FolderDown },
  ];
  const tabProps = { result, saved, goTab };

  return (
    <div className="clip" ref={clip}>
      <div className="clip__grid">
        <div className="stage">
          <div className="stage__title">
            <h1>{result.metadata.filename}</h1>
            <p>{clock(result.metadata.duration_seconds)} · {result.metadata.width} × {result.metadata.height} · {loose(result.metadata.fps)} fps</p>
          </div>

          <div className="player" ref={player}>
            {result.files.annotated_video
              ? <video key={result.run_id} ref={playback.attach} controls playsInline preload="metadata" src={media(result.files.annotated_video)} aria-label={`Annotated video of ${result.metadata.filename}. Boxes are unreviewed model predictions.`} />
              : <p className="player__missing">An annotated video is not available for this run.</p>}
          </div>

          <Timeline result={result} saved={saved} />

          <div className="transport">
            <button type="button" className="btn btn--secondary btn--sm" onClick={() => playback.nudge(-5)} aria-label="Back 5 seconds"><Rewind size={14} aria-hidden="true" /> −5 s</button>
            <button type="button" className="btn btn--secondary btn--sm" onClick={() => playback.nudge(5)} aria-label="Forward 5 seconds">+5 s <FastForward size={14} aria-hidden="true" /></button>
            <p>Boxes are predictions. Review against the footage.</p>
          </div>

          <Disclosure summary="Source & model details">
            <div className="sourcefacts">
              <p>Source: {result.metadata.width} × {result.metadata.height} · {loose(result.metadata.fps)} fps · {num(result.metadata.frame_count, 0)} frames. Analysis sampling: {loose(summary?.sample_fps)} fps.</p>
              <p>Model: {summary?.model ?? "Unknown"} · Tracker: {summary?.tracker ?? "Not recorded"} · Detector input: {summary?.image_size ?? "Unknown"} px · Device: {summary?.device ?? "Unknown"}.</p>
              <p>{result.report.summary}</p>
              {result.warnings?.length ? <ul>{result.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul> : null}
            </div>
          </Disclosure>
        </div>

        <div className="analysis" ref={analysis}>
          <div className="analysis__tabs">
            <Tabs tabs={tabs} active={tab} onChange={goTab} idBase="clip" label="Clip analysis" />
          </div>
          {/* Keyed by clip: filters, selections and review drafts on screen belong to one clip. */}
          <div className="analysis__body" key={result.run_id}>
            <TabPanel id="summary" idBase="clip" active={tab === "summary"}><SummaryTab {...tabProps} reviewUnsaved={reviewUnsaved} /></TabPanel>
            <TabPanel id="possession" idBase="clip" active={tab === "possession"}><PossessionTab {...tabProps} api={api} authHeaders={authHeaders} onRefreshResult={onRefreshResult} /></TabPanel>
            <TabPanel id="ball" idBase="clip" active={tab === "ball"}><BallTab {...tabProps} /></TabPanel>
            <TabPanel id="shape" idBase="clip" active={tab === "shape"}><ShapeTab {...tabProps} media={media} /></TabPanel>
            <TabPanel id="players" idBase="clip" active={tab === "players"}><PlayersTab {...tabProps} api={api} authHeaders={authHeaders} /></TabPanel>
            <TabPanel id="analytics" idBase="clip" active={tab === "analytics"}><AnalyticsTab {...tabProps} reviewUnsaved={reviewUnsaved} /></TabPanel>
            {/* Review stays mounted from the start: it loads the saved review that Summary and the timeline read. */}
            <TabPanel id="review" idBase="clip" active={tab === "review"}>
              <ReviewTab {...tabProps} api={api} authHeaders={authHeaders} reviewUnsaved={reviewUnsaved} onSaved={setSavedReview} onUnsaved={setReviewUnsaved} />
            </TabPanel>
            <TabPanel id="files" idBase="clip" active={tab === "files"}><FilesTab {...tabProps} media={media} /></TabPanel>
          </div>
        </div>
      </div>
    </div>
  );
}

export type ClipTabProps = { result: AnalysisResult; saved: Review | null; goTab: (tab: TabId) => void };
