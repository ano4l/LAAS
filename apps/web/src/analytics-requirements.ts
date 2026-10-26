// Product requirements extracted from the two user-supplied Word documents.
// These definitions describe required analytics; they are not calculated match data.
export type RequiredMetric = { id: string; label: string; needs: string };
export type AnalyticsArea = { id: string; title: string; description: string; source: string; metrics: RequiredMetric[] };
const events = "Needs approved events and verified teams/players";
const history = "Needs validated match and season statistics";

export const analyticsAreas: AnalyticsArea[] = [
  { id: "match", title: "Match analysis", description: "Review home and away performance, key contributors and match problems with video evidence.", source: "Analytics Requirements: Match analysis; Football analytic AI §14", metrics: [
    { id: "passes", label: "Passes and completion", needs: events },
    { id: "tackles", label: "Tackles and outcomes", needs: events },
    { id: "interceptions", label: "Interceptions", needs: events },
    { id: "dribbles", label: "Dribbles and success rate", needs: events },
    { id: "shots-goals", label: "Shots and goals", needs: events },
    { id: "duels", label: "Duels and outcomes", needs: events },
    { id: "possession", label: "Possession and turnovers", needs: "Needs reviewed possession sequences and ball control" },
    { id: "home-away", label: "Home and away comparison", needs: events },
    { id: "key-players", label: "Key players and in-game problems", needs: "Needs verified identities, match statistics and supporting clips" },
  ] },
  { id: "team", title: "Team performance", description: "Compare current and previous seasons, identify strengths and highlight problems.", source: "Analytics Requirements: Team performance; Football analytic AI §14", metrics: [
    { id: "team-season", label: "Match and seasonal statistics", needs: history },
    { id: "season-trends", label: "Current vs previous season trends", needs: history },
    { id: "attacking-defensive", label: "Attacking and defensive performance", needs: history },
    { id: "team-comparison", label: "Team comparisons", needs: "Needs compatible competition and season data" },
    { id: "team-xg", label: "Expected goals (xG)", needs: "Needs a validated xG model/provider and shot data" },
    { id: "team-alerts", label: "Strengths and problem indicators", needs: "Needs historical baselines and defined metrics" },
  ] },
  { id: "opponents", title: "Opponent analysis", description: "Compare opponents in the competition and expose attacking threats and defensive weaknesses.", source: "Analytics Requirements: Opponent analysis; Football analytic AI §14", metrics: [
    { id: "opponent-season", label: "Opponent match and seasonal trends", needs: history },
    { id: "opponent-threats", label: "Attacking threats", needs: "Needs opponent events, shot data and verified statistics" },
    { id: "opponent-weaknesses", label: "Defensive weaknesses and possession turnovers", needs: events },
    { id: "opponent-xg", label: "Opponent xG", needs: "Needs a validated xG model/provider and opponent shot data" },
    { id: "opponent-compare", label: "Opponent vs own team", needs: "Needs comparable team data with competition/season context" },
  ] },
  { id: "players", title: "Players & development", description: "Track individual performance, rank squad members fairly and support development decisions with evidence.", source: "Analytics Requirements: Player development; Football analytic AI §§12,14", metrics: [
    { id: "player-statistics", label: "Player match and seasonal performance", needs: "Needs verified roster identities and approved player events" },
    { id: "role-rankings", label: "Squad and position-specific rankings", needs: "Needs compatible roles, adequate minutes and defined ranking weights" },
    { id: "player-trends", label: "Performance trends and role comparisons", needs: history },
    { id: "development", label: "Development priorities", needs: "Needs reviewed performance evidence and role-specific baselines" },
    { id: "squad-retention", label: "Squad retention recommendations", needs: "Needs statistical evidence and analyst review; not an automated decision" },
    { id: "movement", label: "Trajectories, distance and speed", needs: "Needs pitch/camera calibration and reliable player identity" },
    { id: "heatmaps", label: "Heatmaps, running patterns and team shape", needs: "Needs calibrated pitch coordinates and validated tracking" },
  ] },
  { id: "scouting", title: "Scouting", description: "Rank and shortlist players across competitions, with adjustable criteria and explainable recommendations.", source: "Analytics Requirements: Scouting; Football analytic AI §§13–15", metrics: [
    { id: "competition-ranking", label: "Competition and position rankings", needs: "Needs licensed/imported player statistics and consistent role definitions" },
    { id: "progressive-passes", label: "Progressive passes and selected metrics", needs: "Needs approved events with pitch coordinates and metric definitions" },
    { id: "ranking-weights", label: "Configurable ranking weights", needs: "Needs comparable player data and validated ranking calculations" },
    { id: "shortlists", label: "Player shortlists and video evidence", needs: "Needs verified player profiles and linked match clips" },
    { id: "recruitment-fit", label: "Recruitment suitability recommendations", needs: "Needs team requirements, statistical evidence and analyst review" },
  ] },
  { id: "workload", title: "Workload & injury indicators", description: "Combine player performance and GPS loads to flag changes for qualified staff to review.", source: "Analytics Requirements: Injury prevention; Football analytic AI §12", metrics: [
    { id: "gps-load", label: "GPS external load and distance per minute", needs: "Needs measured GPS imports, player matching and preserved units" },
    { id: "intensity", label: "Movement intensity and running load", needs: "Needs measured loads or validated calibrated video estimates" },
    { id: "load-history", label: "Historical workload changes", needs: "Needs longitudinal player load and match participation data" },
    { id: "injury-indicators", label: "Injury-risk indicators", needs: "Needs validated load history and performance data; staff review, not diagnosis" },
    { id: "workload-alerts", label: "Configurable workload alerts", needs: "Needs defined thresholds, authorised recipients and notification delivery" },
  ] },
  { id: "reports", title: "Reports & exports", description: "Build reusable reports with selected metrics, comparisons and supporting video evidence.", source: "Football analytic AI §§16–17; Analytics Requirements: Structured JSON", metrics: [
    { id: "report-builder", label: "Custom report sections, charts and tables", needs: "Needs validated analytics and report composition" },
    { id: "report-templates", label: "Saved report templates", needs: "Needs persistent report configuration" },
    { id: "report-exports", label: "PDF, CSV and Excel reports", needs: "Needs approved statistics and export generation" },
    { id: "approved-json", label: "Approved event JSON", needs: "Needs event editing/approval and a versioned central event store" },
    { id: "notifications", label: "Completion, review and workload notifications", needs: "Needs delivery preferences and authorised in-app/email recipients" },
  ] },
];

export const requiredFilters = ["Competition", "Season", "Team", "Player", "Position", "Date range", "Match"];
