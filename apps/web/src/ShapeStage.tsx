import { useEffect, useId, useRef, useState } from "react";
import type { ShapeEllipse } from "./SoccerTrackShape";

export type StageGroup = {
  kit: string;
  label: string;
  tone: "a" | "b" | "x" | null;
  primary: boolean;
  players: { track_id: string; x_pct: number; y_pct: number }[];
  ellipse: ShapeEllipse;
};

type Point = [number, number];

/** Convex hull (monotone chain) of the observed footpoints: the outermost players, joined. Nothing is added. */
function hull(points: Point[]): Point[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length < 3) return sorted;
  const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const build = (list: Point[]) => {
    const out: Point[] = [];
    for (const point of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], point) <= 0) out.pop();
      out.push(point);
    }
    out.pop();
    return out;
  };
  return [...build(sorted), ...build([...sorted].reverse())];
}

/**
 * One observed snapshot of each group's shape, drawn where the players actually were in the camera image.
 * With footage on, the still behind the drawing is the real video frame at the snapshot's own timestamp,
 * so the rings sit under the players they describe. Positions are image coordinates, not pitch coordinates.
 */
export function ShapeStage({ width, height, groups, time, videoSrc, footage, showIds, label }: {
  width: number; height: number; groups: StageGroup[]; time: number; videoSrc?: string; footage: boolean; showIds: boolean; label: string;
}) {
  const id = useId();
  const frame = useRef<HTMLVideoElement>(null);
  const [shown, setShown] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);
  const useFootage = footage && Boolean(videoSrc) && !failed;

  useEffect(() => { setFailed(false); setShown(null); }, [videoSrc]);

  // Seek the still to the snapshot's timestamp. The frame only counts as shown once that seek has landed.
  useEffect(() => {
    const video = frame.current;
    if (!useFootage || !video) return;
    const seek = () => { if (Math.abs(video.currentTime - time) > .001) video.currentTime = time; else setShown(time); };
    if (video.readyState >= 1) seek();
    else { video.addEventListener("loadedmetadata", seek, { once: true }); return () => video.removeEventListener("loadedmetadata", seek); }
  }, [time, useFootage, videoSrc]);

  const ring = (yPct: number) => width * (.0085 + .0075 * Math.min(100, Math.max(0, yPct)) / 100);
  const matched = useFootage && shown !== null && Math.abs(shown - time) < .001;

  return (
    <div className={`shapestage ${useFootage ? "shapestage--footage" : "shapestage--plain"} ${matched ? "is-matched" : ""}`} style={{ aspectRatio: `${width}/${height}` }}>
      {useFootage && (
        <video
          ref={frame}
          className="shapestage__frame"
          src={videoSrc}
          muted
          playsInline
          preload="auto"
          tabIndex={-1}
          aria-hidden="true"
          disablePictureInPicture
          onSeeked={event => setShown(event.currentTarget.currentTime)}
          onLoadedData={event => { if (Math.abs(event.currentTarget.currentTime - time) < .001) setShown(time); }}
          onError={() => setFailed(true)}
        />
      )}
      <svg className="shapestage__overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={label}>
        <defs>
          <clipPath id={`${id}-clip`}><rect width={width} height={height} /></clipPath>
          <filter id={`${id}-glow`} x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation={width * .004} /></filter>
        </defs>
        {!useFootage && (
          <g className="shapestage__grid" aria-hidden="true">
            {[1, 2, 3].map(step => <line key={`v${step}`} x1={width * step / 4} x2={width * step / 4} y1="0" y2={height} />)}
            {[1, 2].map(step => <line key={`h${step}`} x1="0" x2={width} y1={height * step / 3} y2={height * step / 3} />)}
          </g>
        )}
        <g clipPath={`url(#${id}-clip)`}>
          {groups.map(group => {
            const points = group.players.map(player => [player.x_pct * width / 100, player.y_pct * height / 100] as Point);
            const outline = hull(points);
            const path = outline.length ? `M${outline.map(point => `${point[0].toFixed(1)},${point[1].toFixed(1)}`).join("L")}${outline.length > 2 ? "Z" : ""}` : "";
            const cx = group.ellipse.centroid_x_pct * width / 100, cy = group.ellipse.centroid_y_pct * height / 100;
            const arm = width * .009;
            return (
              <g key={group.kit} className={`shapestage__group ${group.tone ? `tone-${group.tone}` : ""} ${group.primary ? "is-primary" : "is-other"}`}>
                {path && <path className="shapestage__hull-glow" d={path} filter={`url(#${id}-glow)`} aria-hidden="true" />}
                {path && <path className="shapestage__hull" d={path} />}
                {group.primary && <ellipse className="shapestage__ellipse" cx={cx} cy={cy} rx={group.ellipse.major_radius_px} ry={group.ellipse.minor_radius_px} transform={`rotate(${group.ellipse.angle_degrees} ${cx} ${cy})`} />}
                {group.players.map(player => {
                  const r = ring(player.y_pct);
                  return (
                    <g key={player.track_id} className="shapestage__player" style={{ transform: `translate(${player.x_pct * width / 100}px, ${player.y_pct * height / 100}px)` }}>
                      <title>{`${group.label} · track ${player.track_id} · x ${player.x_pct.toFixed(1)}% · y ${player.y_pct.toFixed(1)}%`}</title>
                      <ellipse className="shapestage__ring" rx={r} ry={r * .42} />
                      <circle className="shapestage__dot" r={r * .26} />
                      {showIds && <text className="shapestage__id" x={r * 1.15} y={-r * .6} fontSize={width * .014}>{player.track_id}</text>}
                    </g>
                  );
                })}
                {group.primary && (
                  <g className="shapestage__centre" style={{ transform: `translate(${cx}px, ${cy}px)` }}>
                    <path d={`M${-arm},0H${arm}M0,${-arm}V${arm}`} />
                  </g>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      {useFootage && !matched && <span className="shapestage__status" role="status">Loading the frame…</span>}
    </div>
  );
}
