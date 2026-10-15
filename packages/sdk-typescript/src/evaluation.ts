// The evaluation API's shapes. PROVISIONAL.
//
// Phase 8 (issue #9) adds GET /api/eval/runs, /api/eval/runs/{id} and
// /api/eval/dashboard. It was not merged when Phase 9 was built, so these
// shapes are typed by hand (decision D20 of the Phase 9 design) from the
// response shapes Phase 8 documents in its design ("Response shapes (as built
// in Task 11)") and from its store, runner and route on its branch on
// 2026-10-10. Its routes return plain dicts with no response model, so even
// after it merges the specification types them as unknown and these stay
// hand typed until those routes declare response models. When Phase 8
// merges, the pages are re-verified against the live routes.

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

/** Latency percentiles of live runs for one strategy (nearest rank). */
export interface LatencyPercentiles {
  p50: number | null;
  p95: number | null;
  runs: number;
}

export interface StrategyCalls {
  runs: number;
  llm_calls: number | null;
  retrieval_calls: number | null;
}

export interface CostPerDay {
  date: string;
  runs: number;
  /** Null for a day whose runs were all unpriced; never priced as zero. */
  estimated_cost_usd: number | null;
  unpriced_runs: number;
}

export interface FallbackRate {
  runs: number;
  fallbacks: number;
  rate: number | null;
}

/** One finished evaluation run in the quality trend; the list is oldest first. */
export interface QualityPoint {
  run_id: number;
  batch: string;
  target: string;
  started_at: string | null;
  judge: string | null;
  hit: number | null;
  reciprocal_rank: number | null;
  correctness: number | null;
  citation_correct: number | null;
}

/** GET /api/eval/dashboard?days=N. Live system numbers come from retrieval_runs, quality from evaluation runs. */
export interface EvalDashboard {
  window_days: number;
  since: string;
  runs: number;
  latency_ms: Record<string, LatencyPercentiles>;
  calls_per_strategy: Record<string, StrategyCalls>;
  cost_per_day: CostPerDay[];
  fallback_rate: FallbackRate;
  quality_trend: QualityPoint[];
}
