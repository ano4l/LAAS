import { createContext, useContext } from "react";
import type { AnalysisResult } from "./main";

/**
 * Kit groups are anonymous jersey-colour clusters such as "scene-0/kit-A". They are shown as "Kit A"
 * and never called Home/Away. A real team name appears only when the user has confirmed a mapping
 * in the Team & stoppage context panel (sequences.mapping_status === "user_confirmed").
 */
export type Tone = "a" | "b" | "x";

export type KitNamer = {
  multiScene: boolean;
  confirmed: boolean;
  /** Reader-facing name for a kit id or a team string from the data. */
  label: (id: string | null | undefined, fallback?: string) => string;
  /** Which of the two kit colours to use; null when the id is not a known group. */
  tone: (id: string | null | undefined) => Tone | null;
  /** The anonymous name ("Kit A") even when a team name has been confirmed. */
  anonymous: (id: string) => string;
};

const KIT = /^(?:scene-(\d+)\/)?kit-([A-Za-z0-9]+)$/;

export function createKitNamer(result: AnalysisResult | null): KitNamer {
  const insights = result?.metrics.video_insights;
  const sequences = insights?.sequences;
  const ids = new Set<string>();
  const add = (id: string | null | undefined) => { if (id) ids.add(id); };

  insights?.kits.forEach(kit => add(kit.kit));
  insights?.control.by_kit.forEach(kit => add(kit.kit));
  insights?.collective_dynamics?.teams.forEach(team => add(team.kit));
  result?.metrics.video_measurements?.provisional_kit_analytics?.forEach(kit => add(kit.kit));
  sequences?.intervals.forEach(interval => add(interval.kit));
  sequences?.anchors.forEach(anchor => add(anchor.kit));
  result?.detection_summary?.linked_tracks?.forEach(track => add(track.team));

  const scenes = new Set<string>();
  const letters = new Set<string>();
  ids.forEach(id => {
    const match = KIT.exec(id);
    if (!match) return;
    if (match[1] !== undefined) scenes.add(match[1]);
    letters.add(match[2].toUpperCase());
  });
  const order = [...letters].sort();
  const multiScene = scenes.size > 1;
  const confirmed = sequences?.mapping_status === "user_confirmed";

  // A confirmed mapping shows up in the data as a team string that differs from its kit id.
  const kitToTeam = new Map<string, string>();
  const teamToKit = new Map<string, string>();
  if (confirmed) {
    const pair = (team: string | null, kit: string | null) => {
      if (!team || !kit || team === kit) return;
      kitToTeam.set(kit, team);
      teamToKit.set(team, kit);
    };
    sequences?.intervals.forEach(interval => pair(interval.team, interval.kit));
    sequences?.anchors.forEach(anchor => pair(anchor.team, anchor.kit));
  }

  const anonymous: KitNamer["anonymous"] = id => {
    const match = KIT.exec(id);
    if (!match) return id;
    const scene = multiScene && match[1] !== undefined ? ` · scene ${Number(match[1]) + 1}` : "";
    return `Kit ${match[2].toUpperCase()}${scene}`;
  };

  const label: KitNamer["label"] = (id, fallback = "Unknown") => {
    if (!id) return fallback;
    return kitToTeam.get(id) ?? anonymous(id);
  };

  const tone: KitNamer["tone"] = id => {
    if (!id) return null;
    const match = KIT.exec(id) ?? KIT.exec(teamToKit.get(id) ?? "");
    if (!match) return null;
    const index = order.indexOf(match[2].toUpperCase());
    return index === 0 ? "a" : index === 1 ? "b" : "x";
  };

  return { multiScene, confirmed, label, anonymous, tone };
}

const KitContext = createContext<KitNamer>(createKitNamer(null));
export const KitProvider = KitContext.Provider;
export const useKits = () => useContext(KitContext);

/** A kit group or confirmed team, with its colour chip. The raw id stays available as a tooltip. */
export function TeamName({ id, fallback = "Unknown" }: { id: string | null | undefined; fallback?: string }) {
  const kits = useKits();
  if (!id) return <span className="muted">{fallback}</span>;
  const tone = kits.tone(id);
  const name = kits.label(id);
  return (
    <span className={`team ${tone ? `tone-${tone}` : ""}`} title={name === id ? undefined : `Group id in the data: ${id}`}>
      {tone && <i className="team__dot" aria-hidden="true" />}
      {name}
    </span>
  );
}
