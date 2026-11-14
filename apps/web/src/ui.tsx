import { useEffect, useId, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";
import { Check, ChevronRight, Play, type LucideIcon } from "lucide-react";
import { DASH, clockTenths } from "./format";
import type { Tone } from "./kits";
import { usePlayback } from "./playback";

/* ---------- Evidence ---------- */

export type Evidence = "seen" | "estimated" | "assumed" | "unknown" | "reviewed";

export const EVIDENCE: Record<Evidence, { label: string; meaning: string }> = {
  seen: { label: "Seen", meaning: "Counted directly from the model's detections. Detections can still be wrong." },
  estimated: { label: "Estimated", meaning: "A model estimate that passed its evidence rules. Not checked by a person." },
  assumed: { label: "Assumed", meaning: "An inference that could be wrong, such as a team keeping the ball between two of its own touches." },
  unknown: { label: "Unknown", meaning: "No estimate. Unknown time is never shared out between teams." },
  reviewed: { label: "Reviewed", meaning: "A person checked it against the footage and approved it." },
};

export function EvidenceChip({ level, title }: { level: Evidence; title?: string }) {
  const { label, meaning } = EVIDENCE[level];
  return (
    <span className={`chip chip--${level}`} title={title ?? meaning}>
      {level === "reviewed" ? <Check size={12} strokeWidth={3} aria-hidden="true" /> : <i className="chip__swatch" aria-hidden="true" />}
      {level === "estimated" ? `~ ${label}` : label}
    </span>
  );
}

/** The one legend for the evidence system. Shown in the app bar popover and at the foot of Summary. */
export function EvidenceLegend({ inline = false }: { inline?: boolean }) {
  return (
    <ul className={`legend ${inline ? "legend--inline" : ""}`}>
      {(Object.keys(EVIDENCE) as Evidence[]).map(level => (
        <li key={level}>
          <span><EvidenceChip level={level} /></span>
          <span>{EVIDENCE[level].meaning}</span>
        </li>
      ))}
    </ul>
  );
}

/* ---------- Stat tile ---------- */

type StatTileProps = {
  label: string;
  /** The headline value. Pass DASH or "Unknown" when the data has no value; never a made-up zero. */
  value: ReactNode;
  unit?: string;
  context?: ReactNode;
  evidence?: Evidence;
  /** Exact technical definition, kept as a tooltip. */
  title?: string;
  /** A chip of your own in place of an evidence level, for a state outside the five levels. */
  chip?: ReactNode;
  onClick?: () => void;
  actionLabel?: string;
};

export function StatTile({ label, value, unit, context, evidence, chip, title, onClick, actionLabel }: StatTileProps) {
  const missing = value === DASH || value === "Unknown" || value == null;
  const wordy = typeof value === "string" && /[A-Za-z]{3}/.test(value);
  const body = (
    <>
      <span className="tile__top">
        <span className="tile__label">{label}</span>
        {chip ?? (evidence && <EvidenceChip level={evidence} />)}
      </span>
      <strong className={`tile__value ${wordy ? "tile__value--text" : ""} ${missing ? "tile__value--unknown" : ""}`}>
        {value ?? DASH}
        {unit && !missing && <small>{unit}</small>}
      </strong>
      {context && <span className="tile__context">{context}</span>}
    </>
  );
  if (onClick) return <button type="button" className="tile tile--link" title={title} onClick={onClick} aria-label={actionLabel}>{body}</button>;
  return <div className="tile" title={title}>{body}</div>;
}

/* ---------- Panel ---------- */

export function Panel({ title, subtitle, aside, children, className = "", level = 3 }: { title: ReactNode; subtitle?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string; level?: 2 | 3 }) {
  const id = useId();
  const Heading = level === 2 ? "h2" : "h3";
  return (
    <section className={`panel ${className}`} aria-labelledby={id}>
      <header className="panel__head">
        <div>
          <Heading className="panel__title" id={id}>{title}</Heading>
          {subtitle && <p className="panel__sub">{subtitle}</p>}
        </div>
        {aside && <div className="panel__aside">{aside}</div>}
      </header>
      <div className="panel__body">{children}</div>
    </section>
  );
}

/* ---------- Disclosure ---------- */

export function Disclosure({ summary, meta, children, variant, defaultOpen = false, className = "" }: { summary: ReactNode; meta?: ReactNode; children: ReactNode; variant?: "panel"; defaultOpen?: boolean; className?: string }) {
  return (
    <details className={`disclosure ${variant ? `disclosure--${variant}` : ""} ${className}`} open={defaultOpen || undefined}>
      <summary>
        <ChevronRight size={16} aria-hidden="true" />
        <span className="disclosure__label">{summary}</span>
        {meta && <span className="disclosure__meta">{meta}</span>}
      </summary>
      <div className="disclosure__body">{children}</div>
    </details>
  );
}

/* ---------- Tabs ---------- */

export type TabDef<T extends string> = { id: T; label: string; icon?: LucideIcon; count?: ReactNode };

export function Tabs<T extends string>({ tabs, active, onChange, idBase, label }: { tabs: TabDef<T>[]; active: T; onChange: (id: T) => void; idBase: string; label: string }) {
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    list.current?.querySelector<HTMLElement>('[aria-selected="true"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [active]);

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const index = tabs.findIndex(tab => tab.id === active);
    const next = event.key === "ArrowRight" ? (index + 1) % tabs.length
      : event.key === "ArrowLeft" ? (index - 1 + tabs.length) % tabs.length
      : event.key === "Home" ? 0
      : event.key === "End" ? tabs.length - 1
      : -1;
    if (next < 0) return;
    event.preventDefault();
    onChange(tabs[next].id);
    window.requestAnimationFrame(() => document.getElementById(`${idBase}-tab-${tabs[next].id}`)?.focus());
  }

  return (
    <div className="tabs" role="tablist" aria-label={label} ref={list} onKeyDown={onKeyDown}>
      {tabs.map(({ id, label: text, icon: Icon, count }) => (
        <button
          key={id}
          type="button"
          role="tab"
          id={`${idBase}-tab-${id}`}
          className="tab"
          aria-selected={active === id}
          aria-controls={`${idBase}-panel-${id}`}
          tabIndex={active === id ? 0 : -1}
          onClick={() => onChange(id)}
        >
          {Icon && <Icon size={16} aria-hidden="true" />}
          {text}
          {count != null && <span className="tab__count">{count}</span>}
        </button>
      ))}
    </div>
  );
}

/** Panels stay mounted so filters, drafts and scroll positions survive a tab switch. */
export function TabPanel({ id, idBase, active, children }: { id: string; idBase: string; active: boolean; children: ReactNode }) {
  return (
    <div className="tabpanel" role="tabpanel" id={`${idBase}-panel-${id}`} aria-labelledby={`${idBase}-tab-${id}`} hidden={!active} tabIndex={-1}>
      {children}
    </div>
  );
}

/* ---------- Bars ---------- */

export type Fill = "solid" | "hatch" | "unknown" | "contested" | "stoppage" | "ball";
export type BarSegment = { key: string; value: number; fill: Fill; tone?: Tone | null; title?: string };

/** Segments are drawn against `total`; whatever is left over stays as the bar's unknown ground. */
export function StackedBar({ segments, total, label, size }: { segments: BarSegment[]; total: number; label: string; size?: "thin" | "tall" }) {
  return (
    <div className={`bar ${size ? `bar--${size}` : ""}`} role="img" aria-label={label}>
      {total > 0 && segments.filter(segment => segment.value > 0).map(segment => (
        <span
          key={segment.key}
          className={`bar__seg fill-${segment.fill} ${segment.tone ? `tone-${segment.tone}` : ""}`}
          style={{ width: `${Math.min(100, (100 * segment.value) / total)}%` }}
          title={segment.title}
        />
      ))}
    </div>
  );
}

/** A single share on a 0–100 scale. A null value leaves the bar empty rather than drawing zero. */
export function Meter({ value, tone, fill = "solid" }: { value: number | null | undefined; tone?: Tone | null; fill?: Fill }) {
  return (
    <div className="bar bar--thin" aria-hidden="true">
      {value != null && <span className={`bar__seg fill-${fill} ${tone ? `tone-${tone}` : ""}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />}
    </div>
  );
}

export function Swatch({ fill, tone }: { fill: Fill; tone?: Tone | null }) {
  return <i className={`swatch fill-${fill} ${tone ? `tone-${tone}` : ""}`} aria-hidden="true" />;
}

/* ---------- Empty state, pager, table ---------- */

export function EmptyState({ icon: Icon, title, children, action }: { icon?: LucideIcon; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty" role="status">
      <p className="empty__title">{Icon && <Icon size={17} aria-hidden="true" />}{title}</p>
      {children && <div>{children}</div>}
      {action}
    </div>
  );
}

export function Pager({ page, pages, count, noun, onPage }: { page: number; pages: number; count: number; noun: string; onPage: (page: number) => void }) {
  return (
    <div className="pager">
      <span>{count.toLocaleString()} {noun} · page {page + 1} of {pages}</span>
      <span className="pager__buttons">
        <button type="button" className="btn btn--secondary btn--sm" disabled={page === 0} onClick={() => onPage(page - 1)} aria-label={`Previous ${noun}`}>Previous</button>
        <button type="button" className="btn btn--secondary btn--sm" disabled={page >= pages - 1} onClick={() => onPage(page + 1)} aria-label={`Next ${noun}`}>Next</button>
      </span>
    </div>
  );
}

/**
 * A table inside a keyboard-reachable scroll region. With `stack`, each row becomes a small card on a
 * phone (cells need a data-label), so no column is pushed off screen.
 */
export function DataTable({ label, children, tall = false, stack = false }: { label: string; children: ReactNode; tall?: boolean; stack?: boolean }) {
  return (
    <div className={`table-wrap ${tall ? "table-wrap--tall" : ""}`} tabIndex={0} role="region" aria-label={label}>
      <table className={`table ${stack ? "table--stack" : ""}`}>{children}</table>
    </div>
  );
}

export function Field({ label, children, className = "" }: { label: ReactNode; children: ReactNode; className?: string }) {
  return <label className={`field ${className}`}>{label}{children}</label>;
}

/* ---------- Popover ---------- */

type PopoverProps = {
  /** Contents of the trigger button. */
  trigger: ReactNode;
  triggerClassName?: string;
  triggerLabel?: string;
  panelLabel: string;
  align?: "start" | "end";
  padded?: boolean;
  className?: string;
  children: ReactNode | ((close: () => void) => ReactNode);
};

export function Popover({ trigger, triggerClassName = "btn btn--secondary", triggerLabel, panelLabel, align = "start", padded = false, className = "", children }: PopoverProps) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const id = useId();

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      const target = event.target as Node;
      // A press on something React has just removed (a list item that re-rendered) is not an outside press.
      if (target.isConnected && !root.current?.contains(target)) setOpen(false);
    };
    const onKey = (event: globalThis.KeyboardEvent) => { if (event.key === "Escape") { setOpen(false); button.current?.focus(); } };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("pointerdown", onPointer); document.removeEventListener("keydown", onKey); };
  }, [open]);

  const close = () => { setOpen(false); button.current?.focus(); };

  return (
    <div className={`popover ${className}`} ref={root}>
      <button ref={button} type="button" className={triggerClassName} aria-haspopup="dialog" aria-expanded={open} aria-controls={open ? id : undefined} aria-label={triggerLabel} onClick={() => setOpen(value => !value)}>
        {trigger}
      </button>
      {open && (
        <div id={id} className={`popover__panel popover__panel--${align} ${padded ? "popover__panel--pad" : ""}`} role="dialog" aria-label={panelLabel}>
          {typeof children === "function" ? children(close) : children}
        </div>
      )}
    </div>
  );
}

/* ---------- Watch ---------- */

/** Plays the shared video from a moment. It never changes tab or scrolls the page away. */
export function WatchButton({ at, label, describe }: { at: number; label?: string; describe?: string }) {
  const playback = usePlayback();
  const stamp = clockTenths(at);
  return (
    <button type="button" className={`watch ${label ? "watch--text" : ""}`} onClick={() => playback.seek(at)} aria-label={`${describe ?? label ?? "Watch"} from ${stamp}`} title={`Play the video from ${stamp}`}>
      <Play size={11} fill="currentColor" aria-hidden="true" />
      {label ?? stamp}
    </button>
  );
}

/** Inline CSS custom properties, typed. Used to pass data values to CSS (for example a heat level). */
export const cssVars = (vars: Record<string, string | number>) => vars as CSSProperties;
