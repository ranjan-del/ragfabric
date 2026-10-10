// The evaluation API's shapes. PROVISIONAL.
//
// Phase 8 (issue #9) adds GET /api/eval/runs, /api/eval/runs/{id} and
// /api/eval/dashboard. It was not merged when Phase 9 was built, so these
// shapes are typed by hand from Phase 8's design (decision D20 of the Phase 9
// design) and every field the design does not pin is optional. When Phase 8
// merges, its routes appear in openapi.json; these interfaces are then
// replaced by aliases of the generated types and the pages re-verified.

/** Mean scores for one target, overall or for one question category. Null means not measured. */
export interface EvalMetrics {
  precision?: number | null;
  recall?: number | null;
  hit_rate?: number | null;
  mrr?: number | null;
  correctness?: number | null;
  faithfulness?: number | null;
  context_relevance?: number | null;
  citation_correctness?: number | null;
}

export interface EvalSummary {
  /** Set when the target could not run in this deployment, with the reason. */
  skipped?: string | null;
  judge?: string | null;
  judge_model?: string | null;
  prompt_version?: string | null;
  top_k?: number | null;
  questions?: number | null;
  errors?: number | null;
  means?: EvalMetrics;
  per_category?: Record<string, EvalMetrics>;
  latency_ms?: { p50?: number | null; p95?: number | null };
  estimated_cost_usd?: number | null;
  [extra: string]: unknown;
}

/** One evaluation_runs row: one target (strategy) in one batch. */
export interface EvalRun {
  id: number;
  /** The batch label; a batch is every run sharing it (Phase 8, D8). */
  name: string;
  commit_sha?: string;
  strategy: string;
  llm_model?: string;
  embedding_model?: string;
  judge_model?: string | null;
  question_set?: string;
  started_at: string;
  finished_at?: string | null;
  summary: EvalSummary;
}

/** One evaluation_results row. */
export interface EvalResult {
  id: number;
  question_id: string;
  question_type: string;
  difficulty?: string;
  question: string;
  expected_answer?: string;
  answer?: string | null;
  precision?: number | null;
  recall?: number | null;
  hit?: boolean | null;
  reciprocal_rank?: number | null;
  correctness?: number | null;
  faithfulness?: number | null;
  context_relevance?: number | null;
  citation_correct?: boolean | null;
  latency_ms?: number | null;
  estimated_cost_usd?: number | null;
  details?: Record<string, unknown>;
}

export interface EvalRunDetail extends EvalRun {
  results: EvalResult[];
}

export interface LatencyPercentiles {
  strategy: string;
  p50_ms: number | null;
  p95_ms: number | null;
  runs: number;
}

export interface CostPerDay {
  day: string;
  estimated_cost_usd: number | null;
  runs: number;
  /** Runs whose model is not in pricing.yaml: counted, never priced as zero. */
  unpriced_runs: number;
}

export interface CallsPerStrategy {
  strategy: string;
  llm_calls: number;
  retrieval_calls: number;
  runs: number;
}

export interface FallbackRate {
  /** Share of auto runs where the routed strategy found nothing and traditional answered. */
  rate: number | null;
  fallbacks: number;
  auto_runs: number;
}

export interface QualityPoint {
  batch: string;
  started_at: string;
  correctness: number | null;
}

export interface QualityTrend {
  strategy: string;
  points: QualityPoint[];
}

export interface EvalDashboard {
  days: number;
  latency: LatencyPercentiles[];
  cost_per_day: CostPerDay[];
  calls_per_strategy: CallsPerStrategy[];
  fallback_rate: FallbackRate;
  quality_trend: QualityTrend[];
}
