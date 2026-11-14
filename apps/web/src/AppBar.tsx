import type { ChangeEvent, RefObject } from "react";
import { Info, LibraryBig, Upload } from "lucide-react";
import type { LibraryRun } from "./main";
import { ClipSwitcher } from "./ClipSwitcher";
import { EvidenceLegend, Popover } from "./ui";

type Props = {
  clipName: string | null;
  currentRunId?: string;
  /** True while the open clip is the synthetic interface demo rather than model output. */
  synthetic: boolean;
  runs: LibraryRun[];
  libraryLoading: boolean;
  /** Hides clip switching and the sample in the static demo build, which has no API. */
  demoMode: boolean;
  showSynthetic: boolean;
  onShowSynthetic: (show: boolean) => void;
  onOpen: (runId: string) => void;
  onLibrary: () => void;
  onSample: () => void;
  sampleDisabled: boolean;
  fileRef: RefObject<HTMLInputElement | null>;
  onFile: (file: File) => void;
  uploadDisabled: boolean;
  uploadTitle: string;
  uploadPct: number | null;
  /** Shown while an upload or analysis is in progress. */
  jobStatus: string | null;
  onOpenJob?: () => void;
};

export function AppBar(props: Props) {
  const { clipName, currentRunId, synthetic, runs, libraryLoading, demoMode, showSynthetic, onShowSynthetic, onOpen, onLibrary, onSample, sampleDisabled, fileRef, onFile, uploadDisabled, uploadTitle, uploadPct, jobStatus, onOpenJob } = props;
  const onChange = (event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (file) onFile(file); };

  return (
    <header className="appbar">
      <div className="appbar__inner">
        <button type="button" className="appbar__brand" onClick={onLibrary} aria-label="LAAS, open the clip library" title="Clip library">LAAS</button>
        {demoMode
          ? clipName && <span className="switcher__trigger switcher" title={clipName}><span className="switcher__name">{clipName}</span></span>
          : <ClipSwitcher currentName={clipName} currentRunId={currentRunId} runs={runs} loading={libraryLoading} showSynthetic={showSynthetic} onShowSynthetic={onShowSynthetic} onOpen={onOpen} onLibrary={onLibrary} />}

        <span className="appbar__spacer" />

        <div className="appbar__group appbar__group--meta">
          <div className="appbar__group">
            <Popover
              align="end"
              padded
              panelLabel="How to read this"
              triggerClassName="appbar__badge"
              triggerLabel={`${synthetic ? "Synthetic demo data" : "Model output, unreviewed"}. How to read this`}
              trigger={<><i aria-hidden="true" />{synthetic ? "Synthetic demo · not real analysis" : "Model output · unreviewed"}<Info size={14} aria-hidden="true" /></>}
            >
              <div className="howto">
                <h2>How to read this</h2>
                <p>
                  {synthetic
                    ? "This clip is a synthetic demo made for exploring the interface. None of its numbers come from real footage."
                    : "Everything here is model output that nobody has checked, unless it is marked Reviewed. Each number carries one of five evidence levels; charts use the same fills."}
                </p>
                <EvidenceLegend />
                <hr />
                <p>Kit groups are anonymous jersey-colour clusters, shown as Kit A and Kit B. A team name appears only after you confirm it under Possession → Team &amp; stoppage context.</p>
                <p>Records a person has approved live under Review and are never mixed into model estimates.</p>
              </div>
            </Popover>
            {jobStatus && (
              <button type="button" className="jobchip" onClick={onOpenJob} disabled={!onOpenJob} title={onOpenJob ? "Open the job that is processing" : undefined}>
                <span className="spinner" aria-hidden="true" /><span role="status">{jobStatus}</span>
              </button>
            )}
          </div>
          <div className="appbar__group">
            {!demoMode && <button type="button" className="btn btn--quiet btn--collapse" onClick={onLibrary} aria-label="Clip library"><LibraryBig size={16} aria-hidden="true" /><span>Library</span></button>}
            {!demoMode && <button type="button" className="btn btn--secondary" onClick={onSample} disabled={sampleDisabled}>Open sample</button>}
          </div>
        </div>

        <span className="appbar__divider" aria-hidden="true" />
        <input ref={fileRef} hidden type="file" aria-label="Upload football video" accept=".mp4,.mov,.avi,.mkv,.webm,.m4v" onChange={onChange} />
        <button type="button" className="btn btn--primary btn--collapse" onClick={() => fileRef.current?.click()} disabled={uploadDisabled} title={uploadTitle} aria-label={uploadPct !== null ? `Uploading ${uploadPct}%` : "Upload video"}>
          <Upload size={16} aria-hidden="true" /><span>{uploadPct !== null ? `Uploading ${uploadPct}%` : "Upload video"}</span>
        </button>
      </div>
    </header>
  );
}
