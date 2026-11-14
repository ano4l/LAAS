import { useMemo, useState } from "react";
import { Check, ChevronsUpDown, Film, LibraryBig, Search } from "lucide-react";
import type { LibraryRun } from "./main";
import { clock } from "./format";
import { Popover } from "./ui";

/** Model analysis of real footage, as opposed to the synthetic interface demo. */
export const isModelRun = (run: LibraryRun) => run.engine === "yolo_bytetrack" || run.metrics.data_quality?.mode === "yolo_bytetrack";

export const matchesClip = (run: LibraryRun, query: string) => `${run.metadata.filename} ${run.run_id}`.toLowerCase().includes(query.trim().toLowerCase());

export const shortDate = (iso: string) => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
};

type Props = {
  currentName: string | null;
  currentRunId?: string;
  runs: LibraryRun[];
  loading: boolean;
  showSynthetic: boolean;
  onShowSynthetic: (show: boolean) => void;
  onOpen: (runId: string) => void;
  onLibrary: () => void;
};

/** The current clip's name, as a button that opens a searchable list of saved clips. */
export function ClipSwitcher(props: Props) {
  const { currentName, loading } = props;
  return (
    <Popover
      className="switcher"
      triggerClassName="switcher__trigger"
      triggerLabel={currentName ? `Current clip: ${currentName}. Switch clip` : "Choose a clip"}
      panelLabel="Switch clip"
      trigger={<><Film size={16} aria-hidden="true" /><span className="switcher__name">{currentName ?? (loading ? "Loading clips…" : "Choose a clip")}</span><ChevronsUpDown size={15} aria-hidden="true" /></>}
    >
      {close => <SwitcherPanel {...props} close={close} />}
    </Popover>
  );
}

/** Mounted only while the popover is open, so the search starts empty every time it opens. */
function SwitcherPanel({ currentRunId, runs, loading, showSynthetic, onShowSynthetic, onOpen, onLibrary, close }: Props & { close: () => void }) {
  const [query, setQuery] = useState("");
  const real = useMemo(() => runs.filter(isModelRun), [runs]);
  const synthetic = useMemo(() => runs.filter(run => !isModelRun(run)), [runs]);
  const realMatches = real.filter(run => matchesClip(run, query));
  const syntheticMatches = showSynthetic ? synthetic.filter(run => matchesClip(run, query)) : [];
  const first = realMatches[0] ?? syntheticMatches[0];
  const open = (runId: string) => { close(); onOpen(runId); };

  const item = (run: LibraryRun) => (
    <li key={run.run_id}>
      <button type="button" className="switcher__item" aria-current={run.run_id === currentRunId ? "true" : undefined} onClick={() => open(run.run_id)}>
        <strong>{run.metadata.filename}</strong>
        <small>{clock(run.metadata.duration_seconds)} · {run.metadata.width} × {run.metadata.height} · {shortDate(run.created_at)}</small>
        {run.run_id === currentRunId && <Check size={16} aria-label="Open now" />}
      </button>
    </li>
  );

  return (
    <div className="switcher__panel">
      <div className="switcher__search">
        <label className="searchbox">
          <span className="sr-only">Search saved clips</span>
          <Search size={15} aria-hidden="true" />
          <input
            className="input"
            type="search"
            autoFocus
            value={query}
            placeholder="Search clip names"
            onChange={event => setQuery(event.target.value)}
            onKeyDown={event => {
              if (event.key !== "Enter" || !first) return;
              // Closing moves focus to the trigger button while Enter is still down. Without this, the
              // browser's keypress then "clicks" that button and the popover reopens behind the reader's
              // back, so their next click on the trigger closes it instead of opening it.
              event.preventDefault();
              open(first.run_id);
            }}
          />
        </label>
      </div>
      <div className="switcher__scroll">
        {loading && !runs.length && <p className="switcher__empty" role="status">Loading clips…</p>}
        {!loading && !realMatches.length && !syntheticMatches.length && (
          <p className="switcher__empty" role="status">{runs.length ? "No clips match this search." : "No saved clips yet. Upload a video to start."}</p>
        )}
        {realMatches.length > 0 && <ul className="switcher__list" aria-label="Analysed clips">{realMatches.map(item)}</ul>}
        {syntheticMatches.length > 0 && <>
          <p className="switcher__group">Synthetic demos · interface examples, not real analysis</p>
          <ul className="switcher__list" aria-label="Synthetic demos">{syntheticMatches.map(item)}</ul>
        </>}
      </div>
      <div className="switcher__foot">
        <label className="check">
          <input type="checkbox" checked={showSynthetic} onChange={event => onShowSynthetic(event.target.checked)} />
          Show synthetic demos{synthetic.length ? ` (${synthetic.length})` : ""}
        </label>
        <button type="button" className="btn btn--quiet btn--sm" onClick={() => { close(); onLibrary(); }}><LibraryBig size={14} aria-hidden="true" /> Open library</button>
      </div>
    </div>
  );
}
