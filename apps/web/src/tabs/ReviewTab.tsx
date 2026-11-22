import { ApprovedSummary, ReviewPanel, type Review } from "../ReviewPanel";
import type { ClipTabProps } from "../Workspace";

type Props = ClipTabProps & {
  api: string;
  authHeaders: () => Record<string, string>;
  reviewUnsaved: boolean;
  onSaved: (saved: Review | null) => void;
  onUnsaved: (unsaved: boolean) => void;
};

/** Human annotation. Reviewed records live here and are never blended into the model's estimates. */
export function ReviewTab({ result, saved, api, authHeaders, reviewUnsaved, onSaved, onUnsaved }: Props) {
  return (
    <div className="tabstack">
      <ReviewPanel runId={result.run_id} duration={result.metadata.duration_seconds} api={api} authHeaders={authHeaders} onSaved={onSaved} onUnsaved={onUnsaved} hasTracking={result.engine !== "review_only"} />
      {saved
        ? <ApprovedSummary saved={saved} unsaved={reviewUnsaved} />
        : <p className="notice" role="status"><span className="spinner" aria-hidden="true" /><span>Loading saved review records…</span></p>}
    </div>
  );
}
