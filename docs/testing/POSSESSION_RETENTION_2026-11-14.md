# Team possession retained between touches, 14 November 2026

## What changed

- `workers/video/sequences.py` (`temporal-possession-2`): a new inferred state, `team_retained`. When two consecutive controlled touches belong to the same known team, are at most 5 seconds apart, sit in one scene and no confirmed stoppage overlaps them, the unknown time between them is attributed to that team. A touch by a team that is unknown, or by the other team, ends the chain.
- Retention is reported in its own `retention` block (seconds, touch pairs, coverage and team shares with retention). The existing fields keep their meaning: `known_seconds`, `coverage_pct` and each team's `estimated_seconds` exclude retained time, and `unknown_seconds` still includes it.
- `workers/video/insights.py` (`video-insights-4`): a ball detection below 0.25 confidence is now used when the offline ball path placed it between confident detections. Older runs are recalculated the next time they are opened.
- Dashboard: the possession panel shows retained time as a separate, labelled block and as its own colour in the state strip.

## Results on the five clips

Replayed through the current pipeline from stored detections (CPU). "Supported" is the existing definition: controlled touches plus ball flight and short gaps backed by ball evidence.

| Clip | Supported before | Supported now | With retention | Touch pairs bridged |
|---|---|---|---|---|
| dfl_test_1 | 2.8% | 10.0% | 26.9% | 5 |
| dfl_test_10 | 8.3% | 9.5% | 22.0% | 4 |
| dfl_test_11 | 12.0% | 13.5% | 46.8% | 12 |
| untitled_design | 5.5% | 9.6% | 62.1% | 12 |
| input_vid | 0.7% | 0.7% | 2.0% | 1 |
| All five | 5.8% | 8.5% | 28.8% | 34 |

The rise in supported time comes from using path-supported weak ball detections. "Before" already includes track linking and the offline ball path.

Among consecutive touches by known teams no more than 5 seconds apart, 47 were followed by the same team and 4 by the other team.

## Limitations

- No human possession labels exist for these clips, so none of these figures is an accuracy measurement. Coverage says how much time receives an estimate, not whether the estimate is right.
- Retention is an assumption. If the model misses an opponent's touch between two touches by one team, the retained time is wrong. The 5-second cap is a judgement, not a tuned value.
- It inherits every upstream error: a wrong kit group, a wrong linked identity or a wrong ball pick produces a wrong touch, and retention stretches that error over more time.
- Three clips show every touch by one kit group. That may be real in 30 seconds of play, or a sign that one group's touches are being missed; it has not been checked.
- `input_vid` has only three touches in 30 seconds, so almost nothing can be bridged.
- More than 70% of clip time is still unassigned overall.
- Checked in the browser on one local run (`dfl_test_10`, an older run recalculated on open). Not run on GPU or deployed.
