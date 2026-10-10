import { EvalRun, STRATEGIES } from '@ragfabric/sdk';

import { Benchmark } from '../compare/compare-state';

/**
 * The benchmark score Compare shows: each strategy's mean correctness in the
 * most recent evaluation batch (a batch is every run sharing a name, Phase 8
 * D8). Rerank variants and `auto` are left out, since Compare's columns are
 * the four plain strategies; a skipped or unscored target is null, never 0.
 */
export function latestBenchmark(runs: readonly EvalRun[]): Benchmark | null {
  if (runs.length === 0) {
    return null;
  }
  const latest = [...runs].sort((a, b) => b.started_at.localeCompare(a.started_at))[0] as EvalRun;
  const scores: Partial<Record<string, number | null>> = {};
  for (const run of runs) {
    if (run.name !== latest.name || !(STRATEGIES as readonly string[]).includes(run.strategy)) {
      continue;
    }
    scores[run.strategy] = run.summary.skipped ? null : (run.summary.means?.correctness ?? null);
  }
  return { batch: latest.name, scores };
}
