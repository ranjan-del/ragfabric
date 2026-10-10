// The evaluation API's shapes. PROVISIONAL.
//
// Phase 8 (issue #9) adds GET /api/eval/runs, /api/eval/runs/{id} and
// /api/eval/dashboard. It was not merged when Phase 9 was built, so these
// shapes are typed by hand (decision D20 of the Phase 9 design): the run,
// result and summary shapes from Phase 8's store and runner as they stood on
// its branch on 2026-10-10 (store.run_dict, store.result_dict,
// runner.summarise), the dashboard from its design alone. When Phase 8
// merges, its routes appear in openapi.json; these interfaces are then
// replaced by aliases of the generated types and the pages re-verified.

/**
 * Mean scores for one target, overall or for one question category.
 * Null means nothing was measured, never zero. `hit` and `citation_correct`
 * are booleans per question, so their means are rates.
 */
export interface EvalMetrics {
  precision?: number | null;
  recall?: number | null;
  hit?: number | null;
  reciprocal_rank?: number | null;
  correctness?: number | null;
  faithfulness?: number | null;
  context_relevance?: number | null;
  citation_correct?: number | null;
}

export interface EvalMeta {
  judge_kind?: string | null;
  judge_model?: string | null;
  judge_prompt_version?: string | null;
  question_set?: string | null;
  top_k?: number | null;
  [extra: string]: unknown;
}

/** `evaluation_runs.summary`, as Phase 8's runner writes it. */
export interface EvalSummary {
  meta?: EvalMeta;
  status?: 'running' | 'finished' | 'skipped' | string;
  /** The reason, when the target could not run in this deployment. */
  skipped?: string | null;
  questions?: number | null;
  errors?: number | null;
  metrics?: EvalMetrics;
  by_category?: Record<string, EvalMetrics & { questions?: number }>;
  latency_ms?: { p50?: number | null; p95?: number | null };
  llm_calls?: number | null;
  retrieval_calls?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  judge_calls?: number | null;
  estimated_cost_usd?: number | null;
  /** Questions whose model is not in pricing.yaml: counted, never priced as zero. */
  cost_unknown?: number | null;
  fallbacks?: number | null;
  strategies_used?: Record<string, number>;
  [extra: string]: unknown;
}

/** One stored evaluation run: one target in one batch. */
export interface EvalRun {
  id: number;
  /** The batch label; a batch is every run sharing it (Phase 8, D8). */
  batch: string;
  /** The target: a strategy, `auto`, or a variant such as `traditional+rerank=llm`. */
  target: string;
  commit?: string;
  llm_model?: string;
  embedding_model?: string;
  judge?: string | null;
  question_set?: string;
  started_at: string | null;
  finished_at?: string | null;
  summary: EvalSummary;
}

/** One question's row in a run. */
export interface EvalResult {
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

// The dashboard's shape is not yet fixed by Phase 8 (its Task 11); the
// interfaces below follow its design's description and are the least certain
// part of this file.

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
