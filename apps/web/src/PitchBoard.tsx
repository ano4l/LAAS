import { useId } from "react";
import { hull } from "./ShapeStage";

export type BoardGroup = {
  kit: string;
  label: string;
  tone: "a" | "b" | "x" | null;
  primary: boolean;
  players: { track_id: string; x_m: number; y_m: number }[];
};

/** Whole-clip presence of one group, in grid cells over the pitch. `coverage` is how long the camera saw each cell. */
export type BoardHeat = { cells: number[][]; coverage: number[][]; tone: "a" | "b" | "x" | null; label: string };

const PAD = 4.5; // metres drawn around the pitch; estimated positions up to 3 m outside the outline are still shown

/**
 * Top-down view of one snapshot. Player positions are estimates: the camera frame is mapped onto an assumed
 * 105 x 68 m pitch from landmarks a model found in that frame. The part of the pitch the camera did not
 * cover is dimmed, because nothing is known about players there.
 */
export function PitchBoard({ length, width, groups, visibleArea, showIds, label, heat }: {
  length: number; width: number; groups: BoardGroup[]; visibleArea: [number, number][]; showIds: boolean; label: string; heat?: BoardHeat;
}) {
  const heatRows = heat?.cells.length ?? 0, heatColumns = heat?.cells[0]?.length ?? 0;
  const cellL = heatColumns ? length / heatColumns : 0, cellW = heatRows ? width / heatRows : 0;
  const heatMax = heat ? Math.max(0, ...heat.cells.flat()) : 0;
  const id = useId();
  const midX = length / 2, midY = width / 2;
  const areaW = 40.32, areaL = 16.5, goalW = 18.32, goalL = 5.5, radius = 9.15, spot = 11, goalMouth = 7.32;
  // The penalty arc is the part of the 9.15 m circle around the spot that lies outside the penalty area.
  const arcHalf = Math.sqrt(radius ** 2 - (areaL - spot) ** 2);
  const stripes = Array.from({ length: 20 }, (_, index) => index).filter(index => index % 2 === 0);
  const seen = visibleArea.map(point => point.join(",")).join(" ");
  const full = `M${-PAD},${-PAD}H${length + PAD}V${width + PAD}H${-PAD}Z`;

  return (
    <div className="pitchboard" style={{ aspectRatio: `${length + 2 * PAD}/${width + 2 * PAD}` }}>
      <svg viewBox={`${-PAD} ${-PAD} ${length + 2 * PAD} ${width + 2 * PAD}`} role="img" aria-label={label}>
        <rect className="pitchboard__surround" x={-PAD} y={-PAD} width={length + 2 * PAD} height={width + 2 * PAD} />
        <rect className="pitchboard__grass" width={length} height={width} />
        <g className="pitchboard__stripes" aria-hidden="true">
          {stripes.map(index => <rect key={index} x={index * length / 20} width={length / 20} height={width} />)}
        </g>
        {heat && heatRows > 0 && heatColumns > 0 && (
          <g className={`pitchboard__heat ${heat.tone ? `tone-${heat.tone}` : ""}`}>
            {heat.cells.flatMap((row, r) => row.map((value, c) => {
              const seen = heat.coverage[r]?.[c] ?? 0;
              if (seen > 0 && value <= 0) return null;
              return (
                <rect key={`${r}-${c}`} className={seen > 0 ? "pitchboard__cell" : "pitchboard__cell pitchboard__cell--unseen"}
                  x={c * cellL} y={r * cellW} width={cellL} height={cellW}
                  style={seen > 0 && heatMax > 0 ? { fillOpacity: .12 + .78 * Math.sqrt(value / heatMax) } : undefined}>
                  <title>{seen > 0 ? `${heat.label}: about ${value.toFixed(1)} player-seconds here (estimate); in the camera’s view for ${seen.toFixed(1)} s` : "Never in the camera’s view: unknown"}</title>
                </rect>
              );
            }))}
          </g>
        )}
        <g className="pitchboard__lines" aria-hidden="true">
          <rect width={length} height={width} />
          <line x1={midX} x2={midX} y1="0" y2={width} />
          <circle cx={midX} cy={midY} r={radius} />
          <rect y={midY - areaW / 2} width={areaL} height={areaW} />
          <rect x={length - areaL} y={midY - areaW / 2} width={areaL} height={areaW} />
          <rect y={midY - goalW / 2} width={goalL} height={goalW} />
          <rect x={length - goalL} y={midY - goalW / 2} width={goalL} height={goalW} />
          <path d={`M${areaL},${midY - arcHalf}A${radius},${radius} 0 0 1 ${areaL},${midY + arcHalf}`} />
          <path d={`M${length - areaL},${midY - arcHalf}A${radius},${radius} 0 0 0 ${length - areaL},${midY + arcHalf}`} />
          <path d={`M1,0A1,1 0 0 1 0,1M${length - 1},0A1,1 0 0 0 ${length},1M0,${width - 1}A1,1 0 0 1 1,${width}M${length},${width - 1}A1,1 0 0 0 ${length - 1},${width}`} />
          <rect className="pitchboard__goal" x="-2" y={midY - goalMouth / 2} width="2" height={goalMouth} />
          <rect className="pitchboard__goal" x={length} y={midY - goalMouth / 2} width="2" height={goalMouth} />
        </g>
        <g className="pitchboard__spots" aria-hidden="true">
          <circle cx={midX} cy={midY} r=".45" /><circle cx={spot} cy={midY} r=".45" /><circle cx={length - spot} cy={midY} r=".45" />
        </g>
        {visibleArea.length >= 3 && <>
          <path className="pitchboard__unseen" fillRule="evenodd" d={`${full}M${visibleArea.map(point => point.join(",")).join("L")}Z`} />
          <polygon className="pitchboard__seen" points={seen} />
        </>}
        <defs><filter id={`${id}-glow`} x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation=".7" /></filter></defs>
        {groups.map(group => {
          const outline = hull(group.players.map(player => [player.x_m, player.y_m] as [number, number]));
          const path = outline.length ? `M${outline.map(point => `${point[0].toFixed(2)},${point[1].toFixed(2)}`).join("L")}${outline.length > 2 ? "Z" : ""}` : "";
          return (
            <g key={group.kit} className={`pitchboard__group ${group.tone ? `tone-${group.tone}` : ""} ${group.primary ? "is-primary" : "is-other"}`}>
              {path && group.primary && <path className="pitchboard__hull-glow" d={path} filter={`url(#${id}-glow)`} aria-hidden="true" />}
              {path && <path className="pitchboard__hull" d={path} />}
              {group.players.map(player => (
                <g key={player.track_id} className="pitchboard__player" style={{ transform: `translate(${player.x_m}px, ${player.y_m}px)` }}>
                  <title>{`${group.label} · track ${player.track_id} · about ${player.x_m.toFixed(1)} m along, ${player.y_m.toFixed(1)} m across (estimate)`}</title>
                  <circle className="pitchboard__dot" r="1.25" />
                  {showIds && <text className="pitchboard__id" x="1.8" y="-1.4" fontSize="2.3">{player.track_id}</text>}
                </g>
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
