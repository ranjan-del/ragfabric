// Compare's state (Phase 9, Task 7): order, scheduling, failures, the summary.

import { Run } from '@ragfabric/sdk';

import { CompareState } from './compare-state';

function done(state: CompareState, strategy: string, latency: number, cost: number | null = null): void {
  const column = state.column(strategy as never);
  column.answer.start();
  column.answer.apply({ event: 'done', data: { run_id: 1, latency_ms: latency, usage: {} } });
  column.run.set({ estimated_cost_usd: cost } as Run);
}

describe('CompareState', () => {
  it('test_four_columns_in_a_fixed_order', () => {
    const state = new CompareState();
    expect(state.columns.map((c) => c.strategy)).toEqual(['traditional', 'vectorless', 'agentic', 'graph']);
  });

  it('test_sequential_starts_one_and_the_next_only_when_the_previous_ends', () => {
    const state = new CompareState();
    expect(state.begin('q', false)).toEqual(['traditional']);
    expect(state.finished('traditional')).toEqual(['vectorless']);
    expect(state.finished('vectorless')).toEqual(['agentic']);
    expect(state.finished('agentic')).toEqual(['graph']);
    expect(state.finished('graph')).toEqual([]);
    expect(state.measuredInParallel()).toBeFalse();
  });

  it('test_parallel_starts_all_four_at_once', () => {
    const state = new CompareState();
    expect(state.begin('q', true)).toEqual(['traditional', 'vectorless', 'agentic', 'graph']);
    expect(state.finished('agentic')).toEqual([]);
    expect(state.measuredInParallel()).toBeTrue();
  });

  it('test_one_failure_leaves_the_others_running_and_the_queue_moving', () => {
    const state = new CompareState();
    state.begin('q', false);
    state.column('traditional').answer.start();
    state.column('traditional').answer.fail('Agentic needs a model.');
    expect(state.finished('traditional')).toEqual(['vectorless']);
    expect(state.column('traditional').answer.status()).toBe('failed');
    expect(state.column('vectorless').answer.status()).toBe('idle');
  });

  it('test_the_summary_marks_fastest_cheapest_and_best_benchmark_ignoring_nulls', () => {
    const state = new CompareState();
    state.begin('q', false);
    done(state, 'traditional', 900, 0.002);
    done(state, 'vectorless', 120, null);
    done(state, 'agentic', 4000, 0.009);
    state.column('graph').answer.start();
    state.column('graph').answer.fail('No graph.');
    state.benchmark.set({ batch: 'b1', scores: { traditional: 0.6, vectorless: null, agentic: 0.8 } });

    const summary = state.summary();
    expect(summary.fastest).toBe('vectorless');
    expect(summary.cheapest).toBe('traditional');
    expect(summary.bestBenchmark).toBe('agentic');
  });

  it('test_no_cost_anywhere_means_no_cheapest_rather_than_a_zero', () => {
    const state = new CompareState();
    state.begin('q', false);
    done(state, 'traditional', 900, null);
    expect(state.summary().cheapest).toBeNull();
    expect(state.cost('traditional')).toBe('n/a');
    expect(state.benchmarkScore('traditional')).toBe('n/a');
  });

  it('test_begin_clears_the_previous_comparison', () => {
    const state = new CompareState();
    state.begin('first', false);
    done(state, 'traditional', 10, 0.1);
    state.begin('second', true);
    expect(state.question()).toBe('second');
    expect(state.column('traditional').answer.status()).toBe('idle');
    expect(state.column('traditional').run()).toBeNull();
  });
});
