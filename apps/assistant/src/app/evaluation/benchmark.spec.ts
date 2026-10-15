// The latest batch's benchmark score per strategy, for Compare (decision D16).

import { EvalRun } from '@ragfabric/sdk';

import { latestBenchmark } from './benchmark';

function run(id: number, batch: string, target: string, started: string, correctness: number | null, skipped?: string): EvalRun {
  return {
    id,
    batch,
    target,
    started_at: started,
    summary: skipped ? { status: 'skipped', skipped } : { status: 'finished', metrics: { correctness } },
  };
}

describe('latestBenchmark', () => {
  it('test_takes_the_most_recent_batch_and_its_mean_correctness_per_strategy', () => {
    const benchmark = latestBenchmark([
      run(1, 'old', 'traditional', '2026-10-01T00:00:00Z', 0.2),
      run(2, 'new', 'traditional', '2026-10-09T00:00:00Z', 0.7),
      run(3, 'new', 'graph', '2026-10-09T00:05:00Z', 0.5),
      run(4, 'new', 'traditional+rerank=llm', '2026-10-09T00:06:00Z', 0.9),
      run(5, 'new', 'agentic', '2026-10-09T00:07:00Z', null, 'offline provider'),
    ]);

    expect(benchmark?.batch).toBe('new');
    expect(benchmark?.scores).toEqual({ traditional: 0.7, graph: 0.5, agentic: null });
  });

  it('test_no_runs_is_no_benchmark', () => {
    expect(latestBenchmark([])).toBeNull();
  });
});
