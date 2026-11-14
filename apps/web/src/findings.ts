import type { VideoInsights } from "./VideoInsights";
import { clock, pct, secs } from "./format";
import type { KitNamer } from "./kits";

/**
 * The analysis writes its findings and bookmarks as finished sentences in pipeline vocabulary
 * ("Sustained control supports 1.60s of 30.00s"). Those sentences are never edited here. For each kind
 * whose numbers also exist as structured fields, a plain-language sentence is composed from the fields,
 * so it uses the same figures and rounding as the rest of the screen. The original wording is kept
 * alongside for a tooltip and the "analysis wording" disclosure. An unrecognised kind is passed through
 * unchanged.
 */
export type Plain = { text: string; original: string; composed: boolean };

/** Reading order: the possession estimate first, then the stricter check it could be confused with. */
const ORDER = ["possession", "coverage", "limitation", "events", "shape"];

type Finding = VideoInsights["findings"][number];
type Bookmark = NonNullable<VideoInsights["bookmarks"]>[number];

/** The analysis writes one shape finding per kit group with at least two supported windows, in kit order. */
function shapeSubjects(insights: VideoInsights) {
  return insights.kits
    .map(kit => ({ kit, supported: kit.shape_timeline.filter(window => window.supported_seconds >= 0.5) }))
    .filter(entry => entry.supported.length >= 2);
}

export function plainFindings(insights: VideoInsights, kits: KitNamer): (Plain & { kind: string })[] {
  const s = insights.sequences;
  const control = insights.control;
  const subjects = shapeSubjects(insights);
  const shapeCount = insights.findings.filter(finding => finding.kind === "shape").length;
  let shapeIndex = 0;

  const compose = (finding: Finding): string | null => {
    switch (finding.kind) {
      case "possession": {
        if (!s) return null;
        const contested = s.contested_seconds > 0 ? ` and ${secs(s.contested_seconds)} is contested` : "";
        return s.known_seconds > 0
          ? `Possession we can support: ${secs(s.known_seconds)} of ${secs(s.classified_non_dead_seconds)} playing time (${pct(s.coverage_pct)}). ${secs(s.unknown_seconds)} is unknown${contested}. The split between the teams covers the supported time only.`
          : `No possession could be supported in this clip: ${secs(s.unknown_seconds)} of playing time is unknown${contested}. That is unknown, not an even split.`;
      }
      case "coverage":
        return `A stricter check, the ball within reach of one player's feet for several frames in a row, supports ${secs(control.estimated_control_seconds)} of the ${secs(insights.source_duration_seconds)} clip. `
          + (s ? "It is a separate measure from the possession estimate, not a second possession figure." : "It is a separate measure and is not team possession.");
      case "limitation":
        return control.grouped_coverage_pct == null
          ? null
          : `Under that stricter check, only ${pct(control.grouped_coverage_pct)} of the clip can be credited to a kit group. That is too little for its kit shares to describe match possession.`;
      case "events": {
        const passes = insights.candidates.filter(candidate => candidate.type === "possible_pass").length;
        const turnovers = insights.candidates.filter(candidate => candidate.type === "possible_turnover").length;
        return passes + turnovers === 0
          ? "No possible passes or turnovers have enough supporting evidence. This does not establish a pass or turnover total."
          : `${passes} possible ${passes === 1 ? "pass" : "passes"} and ${turnovers} possible ${turnovers === 1 ? "turnover" : "turnovers"} are waiting for review. None is counted until a person approves it.`;
      }
      case "shape": {
        // Only trust the pairing when the counts line up; otherwise show the analysis's own sentence.
        const subject = shapeCount === subjects.length ? subjects[shapeIndex++] : undefined;
        const first = subject?.supported[0];
        const last = subject?.supported[subject.supported.length - 1];
        if (!subject || !first || !last || first.width_pct == null || last.width_pct == null) return null;
        return `${kits.label(subject.kit.kit)} was spread across ${pct(first.width_pct)} of the image width in the window starting at ${clock(first.start_seconds)}, and ${pct(last.width_pct)} in the window starting at ${clock(last.start_seconds)}. Camera movement and perspective affect this comparison.`;
      }
      default:
        return null;
    }
  };

  const composed = insights.findings.map((finding, index) => {
    const text = compose(finding);
    return { kind: finding.kind, text: text ?? finding.text, original: finding.text, composed: text !== null, index };
  });
  const rank = (kind: string) => { const position = ORDER.indexOf(kind); return position < 0 ? ORDER.length : position; };
  return composed.sort((a, b) => rank(a.kind) - rank(b.kind) || a.index - b.index);
}

export function plainBookmark(bookmark: Bookmark, insights: VideoInsights, kits: KitNamer): { label: string; detail: string; original: string; composed: boolean } {
  const original = `${bookmark.label}. ${bookmark.detail}`;
  const sameWindow = (window: { start_seconds: number; end_seconds: number }) => window.start_seconds === bookmark.start_seconds && window.end_seconds === bookmark.end_seconds;

  if (bookmark.type === "ball_visibility") {
    const slice = insights.timeline.find(sameWindow);
    if (slice) return { label: "Clearest view of the ball", detail: `Ball seen for ${secs(slice.ball_observed_seconds)} of this window (${pct(slice.ball_coverage_pct)}).`, original, composed: true };
  }
  if (bookmark.type === "kit_shape") {
    const kit = insights.kits.find(item => bookmark.id === `width-${item.kit}`);
    const window = kit?.shape_timeline.find(sameWindow);
    if (kit && window && window.width_pct != null) {
      return { label: `${kits.label(kit.kit)} at its widest`, detail: `Spread across ${pct(window.width_pct)} of the image width, with at least three of the group in view for ${secs(window.supported_seconds)}.`, original, composed: true };
    }
  }
  return { label: bookmark.label, detail: bookmark.detail, original, composed: false };
}
