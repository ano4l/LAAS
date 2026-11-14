import type { SequenceState } from "./TemporalPossession";
import type { Fill } from "./ui";

/**
 * Plain-language names for the possession states, with the pipeline's own term kept for tooltips
 * and the "How this is measured" disclosures so nothing is lost in translation.
 */
export const STATE_LABEL: Record<SequenceState, string> = {
  controlled: "On the ball",
  in_transit: "Ball travelling",
  control_gap: "Brief gap",
  team_retained: "Kept between touches (assumed)",
  contested: "Contested",
  unknown: "Unknown",
  dead_ball: "Stoppage (confirmed)",
};

export const STATE_TECHNICAL: Record<SequenceState, string> = {
  controlled: "Controlled estimate",
  in_transit: "Ball in transit",
  control_gap: "Bounded control gap",
  team_retained: "Retained between touches (inferred)",
  contested: "Contested",
  unknown: "Unknown / unassigned",
  dead_ball: "Confirmed stoppage",
};

export type SegmentKind = "supported" | "assumed" | "contested" | "stoppage" | "unknown";

/** How an interval is drawn. A state with no team cannot be credited to anyone, so it reads as unknown. */
export function segmentKind(state: SequenceState, team: string | null): SegmentKind {
  if (state === "dead_ball") return "stoppage";
  if (state === "contested") return "contested";
  if (state === "unknown" || !team) return "unknown";
  if (state === "team_retained") return "assumed";
  return "supported";
}

export const KIND_FILL: Record<SegmentKind, Fill> = {
  supported: "solid",
  assumed: "hatch",
  contested: "contested",
  stoppage: "stoppage",
  unknown: "unknown",
};
