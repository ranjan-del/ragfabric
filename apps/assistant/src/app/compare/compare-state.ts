import { WritableSignal, computed, signal } from '@angular/core';
import { Run, STRATEGIES, StrategyName } from '@ragfabric/sdk';

import { AnswerState } from '../answer/answer-state';
import { score, usd } from '../answer/format';

export interface CompareColumn {
  strategy: StrategyName;
  answer: AnswerState;
  /** The stored run, fetched after `done`; its cost is the estimate shown. */
  run: WritableSignal<Run | null>;
}

/** Mean correctness per strategy from one evaluation batch (decision D16). */
export interface Benchmark {
  batch: string;
  scores: Partial<Record<string, number | null>>;
}

export interface CompareSummary {
  fastest: StrategyName | null;
  cheapest: StrategyName | null;
  bestBenchmark: StrategyName | null;
}

function best(
  columns: readonly CompareColumn[],
  value: (c: CompareColumn) => number | null | undefined,
  better: (a: number, b: number) => boolean,
): StrategyName | null {
  let winner: StrategyName | null = null;
  let winning: number | null = null;
  for (const column of columns) {
    const v = value(column);
    if (v == null) {
      continue;
    }
    if (winning === null || better(v, winning)) {
      winner = column.strategy;
      winning = v;
    }
  }
  return winner;
}

/**
 * One question across the four strategies (Phase 9, decisions D12 to D16).
 *
 * Pure state, no HTTP: the page starts what `begin` and `finished` return and
 * feeds each column's events into its AnswerState. Sequential by default,
 * because a local model serves one request at a time and four at once would
 * measure the queue; parallel on request, and the latencies are then labelled.
 */
export class CompareState {
  readonly columns: readonly CompareColumn[] = STRATEGIES.map((strategy) => ({
    strategy,
    answer: new AnswerState(),
    run: signal<Run | null>(null),
  }));
  readonly question = signal('');
  readonly measuredInParallel = signal(false);
  /** Null until loaded; the note explains why there is none when there is none. */
  readonly benchmark = signal<Benchmark | null>(null);
  readonly benchmarkNote = signal<string | null>(null);

  private queue: StrategyName[] = [];

  column(strategy: StrategyName): CompareColumn {
    const found = this.columns.find((c) => c.strategy === strategy);
    if (found === undefined) {
      throw new Error(`no column for ${strategy}`);
    }
    return found;
  }

  readonly running = computed(() => this.columns.some((c) => c.answer.status() === 'running'));

  /** Reset and return the strategies to start now. */
  begin(question: string, parallel: boolean): StrategyName[] {
    this.question.set(question);
    this.measuredInParallel.set(parallel);
    for (const column of this.columns) {
      column.answer.reset();
      column.run.set(null);
    }
    const order = [...STRATEGIES];
    if (parallel) {
      this.queue = [];
      return order;
    }
    this.queue = order.slice(1);
    return order.slice(0, 1);
  }

  /** A column ended (answered or failed); return what to start next. */
  finished(_strategy: StrategyName): StrategyName[] {
    const next = this.queue.shift();
    return next === undefined ? [] : [next];
  }

  /** Stop scheduling: whatever has not started never will. */
  cancel(): void {
    this.queue = [];
  }

  readonly summary = computed<CompareSummary>(() => {
    const benchmark = this.benchmark();
    return {
      fastest: best(this.columns, (c) => c.answer.done()?.latency_ms, (a, b) => a < b),
      cheapest: best(this.columns, (c) => c.run()?.estimated_cost_usd, (a, b) => a < b),
      bestBenchmark: best(this.columns, (c) => benchmark?.scores[c.strategy], (a, b) => a > b),
    };
  });

  cost(strategy: StrategyName): string {
    return usd(this.column(strategy).run()?.estimated_cost_usd);
  }

  benchmarkScore(strategy: StrategyName): string {
    return score(this.benchmark()?.scores[strategy]);
  }
}
