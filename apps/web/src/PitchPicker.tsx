import type { KeyboardEvent, MouseEvent, ReactNode } from "react";

/** A point on the drawn pitch, in metres from its top-left corner. The pitch is assumed to be 105 x 68 m. */
export type PitchPoint = { x: number; y: number };
export type PitchMark = { key: string; at: PitchPoint; to?: PitchPoint | null; tone?: "home" | "away" | null; filled?: boolean; title?: string };

export const PITCH_LENGTH = 105, PITCH_WIDTH = 68;
const PAD = 2;

/**
 * A top-down pitch for placing and showing events. With `onPick` it is clickable and reports the point in
 * metres; without it, it only shows `marks`. Positions are a person's judgement by eye, never a measurement.
 */
export function PitchPicker({ marks, onPick, label, leftLabel, rightLabel, overlay }: {
  marks: PitchMark[]; onPick?: (point: PitchPoint) => void; label: string; leftLabel?: string; rightLabel?: string;
  /** Extra drawing in pitch metres, above the markings and below the marks. */
  overlay?: ReactNode;
}) {
  const midX = PITCH_LENGTH / 2, midY = PITCH_WIDTH / 2;
  const areaW = 40.32, areaL = 16.5, goalW = 18.32, goalL = 5.5, radius = 9.15, spot = 11, mouth = 7.32;
  const arcHalf = Math.sqrt(radius ** 2 - (areaL - spot) ** 2);

  function pick(event: MouseEvent<SVGSVGElement>) {
    if (!onPick) return;
    const svg = event.currentTarget, matrix = svg.getScreenCTM();
    if (!matrix) return;
    const local = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    onPick({ x: Math.round(Math.max(0, Math.min(PITCH_LENGTH, local.x)) * 10) / 10, y: Math.round(Math.max(0, Math.min(PITCH_WIDTH, local.y)) * 10) / 10 });
  }
  // The keyboard route is the typed coordinates beside the pitch; Enter here only explains that.
  function key(event: KeyboardEvent<SVGSVGElement>) { if (onPick && (event.key === "Enter" || event.key === " ")) event.preventDefault(); }

  return (
    <div className={`pitchpick ${onPick ? "pitchpick--live" : ""}`}>
      <svg viewBox={`${-PAD} ${-PAD} ${PITCH_LENGTH + 2 * PAD} ${PITCH_WIDTH + 2 * PAD}`} role="img" aria-label={label} onClick={pick} onKeyDown={key}>
        <rect className="pitchpick__grass" x={0} y={0} width={PITCH_LENGTH} height={PITCH_WIDTH} />
        <g className="pitchpick__lines">
          <rect x={0} y={0} width={PITCH_LENGTH} height={PITCH_WIDTH} />
          <line x1={midX} y1={0} x2={midX} y2={PITCH_WIDTH} />
          <circle cx={midX} cy={midY} r={radius} />
          {[0, 1].map(side => {
            const goalLine = side ? PITCH_LENGTH : 0, inward = side ? -1 : 1;
            return (
              <g key={side}>
                <rect x={side ? PITCH_LENGTH - areaL : 0} y={midY - areaW / 2} width={areaL} height={areaW} />
                <rect x={side ? PITCH_LENGTH - goalL : 0} y={midY - goalW / 2} width={goalL} height={goalW} />
                <path d={`M${goalLine + inward * areaL},${midY - arcHalf}A${radius},${radius} 0 0 ${side ? 0 : 1} ${goalLine + inward * areaL},${midY + arcHalf}`} />
                <circle className="pitchpick__spot" cx={goalLine + inward * spot} cy={midY} r={0.4} />
                <rect className="pitchpick__goal" x={side ? PITCH_LENGTH : -1.2} y={midY - mouth / 2} width={1.2} height={mouth} />
              </g>
            );
          })}
        </g>
        {overlay}
        {marks.map(mark => (
          <g key={mark.key} className={`pitchpick__mark ${mark.tone ? `pitchpick__mark--${mark.tone}` : ""}`}>
            {mark.title && <title>{mark.title}</title>}
            {mark.to && <line x1={mark.at.x} y1={mark.at.y} x2={mark.to.x} y2={mark.to.y} />}
            {mark.to && <rect x={mark.to.x - 0.9} y={mark.to.y - 0.9} width={1.8} height={1.8} transform={`rotate(45 ${mark.to.x} ${mark.to.y})`} />}
            <circle cx={mark.at.x} cy={mark.at.y} r={1.3} className={mark.filled ? "is-filled" : ""} />
          </g>
        ))}
      </svg>
      {(leftLabel || rightLabel) && <p className="pitchpick__ends"><span>{leftLabel}</span><span>{rightLabel}</span></p>}
    </div>
  );
}
