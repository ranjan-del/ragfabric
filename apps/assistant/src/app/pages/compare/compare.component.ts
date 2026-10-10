import { ChangeDetectionStrategy, Component, OnDestroy, OnInit, effect, inject, signal, untracked } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { Citation, RagFabricError, StrategyName } from '@ragfabric/sdk';
import { Subscription } from 'rxjs';

import { AnswerViewComponent } from '../../answer/answer-view.component';
import { count, ms } from '../../answer/format';
import { SourceViewerComponent } from '../../answer/source-viewer.component';
import { CompareState } from '../../compare/compare-state';
import { latestBenchmark } from '../../evaluation/benchmark';
import { Collection } from '../../models';
import { AskService } from '../../services/ask.service';
import { AuthService } from '../../services/auth.service';
import { CollectionService } from '../../services/collection.service';
import { EvaluationService } from '../../services/evaluation.service';
import { RunService } from '../../services/run.service';

/**
 * Compare: one question, the four strategies side by side.
 *
 * Each column is an ordinary POST /api/ask with the strategy named, so each is
 * recorded as its own run with its own trace. Cost comes from the stored run
 * (D15); the benchmark score is the latest evaluation batch's mean
 * correctness, never a score for this question (D16).
 */
@Component({
  selector: 'app-compare',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, RouterLink, AnswerViewComponent, SourceViewerComponent],
  template: `
    <div class="page-head">
      <h1>Compare</h1>
      <p>One question, all four strategies, side by side: what each answered, from which sources, and at what cost.</p>
    </div>

    <form class="card compare-form" (ngSubmit)="compare()">
      <label class="sr-only" for="cq">Question</label>
      <input id="cq" name="question" [(ngModel)]="question" placeholder="e.g. Who owns Project Atlas and what does it depend on?" />
      <div class="controls">
        <label class="control">
          <span>Collection</span>
          <select name="collection" [(ngModel)]="collectionId">
            <option [ngValue]="null">All I can read</option>
            @for (c of collections(); track c.id) {<option [ngValue]="c.id">{{ c.name }}</option>}
          </select>
        </label>
        <label class="control narrow">
          <span>Passages</span>
          <input type="number" name="topK" min="1" max="50" [(ngModel)]="topK" />
        </label>
        <label class="check">
          <input type="checkbox" name="parallel" [(ngModel)]="parallel" data-test="parallel" />
          Run all four at once
        </label>
        <button class="btn btn-primary" type="submit" [disabled]="state.running() || !question.trim()">
          {{ state.running() ? 'Comparing...' : 'Compare' }}
        </button>
      </div>
      <p class="muted small">
        One after another by default: a local model serves one request at a time, so running all four
        at once measures the queue as much as the strategy.
      </p>
    </form>

    @if (state.question()) {
      @if (state.measuredInParallel()) {
        <p class="alert alert-warning" data-test="contention">
          These latencies were measured with all four strategies running at once, so they include contention.
        </p>
      }
      @if (state.benchmarkNote(); as note) {
        <p class="muted small" data-test="benchmark-note">{{ note }}</p>
      }
      <div class="grid">
        @for (col of state.columns; track col.strategy) {
          <section class="card column" [attr.data-test]="'column-' + col.strategy" [attr.aria-label]="col.strategy">
            <header class="col-head">
              <h2>{{ col.strategy }}</h2>
              <span class="badge" [attr.data-status]="col.answer.status()">{{ statusLabel(col.answer.status()) }}</span>
            </header>
            <div class="flags">
              @if (state.summary().fastest === col.strategy) {<span class="badge badge-success" data-test="fastest">fastest</span>}
              @if (state.summary().cheapest === col.strategy) {<span class="badge badge-success" data-test="cheapest">cheapest</span>}
              @if (state.summary().bestBenchmark === col.strategy) {<span class="badge badge-success" data-test="best">best benchmark</span>}
            </div>
            @if (col.answer.status() !== 'idle') {
              <app-answer-view [state]="col.answer" (cite)="viewing.set($event)" />
            } @else {
              <p class="muted">Waiting for its turn.</p>
            }
            <dl class="metrics">
              <div><dt>Latency</dt><dd data-test="latency">{{ ms(col.answer.done()?.latency_ms) }}</dd></div>
              <div><dt>Sources cited</dt><dd data-test="cited">{{ cited(col.answer.status(), col.answer.citations()) }}</dd></div>
              <div><dt>LLM calls</dt><dd>{{ count(col.answer.done()?.usage?.llm_calls) }}</dd></div>
              <div><dt>Retrieval calls</dt><dd>{{ count(col.answer.done()?.usage?.retrieval_calls) }}</dd></div>
              <div><dt>Embedding calls</dt><dd>{{ count(col.answer.done()?.usage?.embedding_calls) }}</dd></div>
              <div><dt>Tokens in / out</dt><dd>{{ count(col.answer.done()?.usage?.input_tokens) }} / {{ count(col.answer.done()?.usage?.output_tokens) }}</dd></div>
              <div><dt>Cost (estimate)</dt><dd data-test="cost">{{ state.cost(col.strategy) }}</dd></div>
              <div><dt>Benchmark score (latest batch)</dt><dd data-test="benchmark">{{ state.benchmarkScore(col.strategy) }}</dd></div>
            </dl>
            @if (col.answer.done(); as d) {
              <a class="btn btn-sm" [routerLink]="['/trace', d.run_id]">Trace</a>
            }
          </section>
        }
      </div>
    }

    <app-source-viewer [citation]="viewing()" (closed)="viewing.set(null)" />
  `,
  styles: `
    .compare-form { display: grid; gap: var(--space-3); padding: var(--space-4); }
    .compare-form > input { font: inherit; padding: var(--space-3); border: 1px solid var(--border);
      border-radius: var(--radius-sm); background: var(--surface); color: var(--text); }
    .controls { display: flex; flex-wrap: wrap; gap: var(--space-3); align-items: end; }
    .control { display: grid; gap: var(--space-1); font-size: 0.8rem; color: var(--text-muted); }
    .control select, .control input { font: inherit; color: var(--text); background: var(--surface);
      border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.4rem 0.6rem; }
    .narrow input { width: 5rem; }
    .check { display: flex; gap: var(--space-2); align-items: center; font-size: 0.9rem; white-space: nowrap; }
    .small { font-size: 0.8rem; margin: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: var(--space-4); margin-top: var(--space-4); }
    .column { padding: var(--space-4); display: grid; gap: var(--space-3); align-content: start; }
    .col-head { display: flex; justify-content: space-between; align-items: center; }
    .col-head h2 { margin: 0; font-size: 1.05rem; text-transform: capitalize; }
    .flags { display: flex; gap: var(--space-1); flex-wrap: wrap; min-height: 1.4rem; }
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-2); margin: 0; }
    .metrics dt { font-size: 0.72rem; color: var(--text-muted); }
    .metrics dd { margin: 0; }
    .column > a { justify-self: start; }
    .check input { width: auto; margin: 0; }
  `,
})
export class CompareComponent implements OnInit, OnDestroy {
  private readonly asker = inject(AskService);
  private readonly runs = inject(RunService);
  private readonly auth = inject(AuthService);
  private readonly evaluation = inject(EvaluationService);
  private readonly collectionService = inject(CollectionService);
  private subscriptions: Subscription[] = [];

  readonly state = new CompareState();
  readonly collections = signal<Collection[]>([]);
  readonly viewing = signal<Citation | null>(null);
  readonly ms = ms;
  readonly count = count;

  private benchmarkRequested = false;

  constructor() {
    // The profile may still be loading when the page opens directly, so the
    // admin check reacts to it rather than reading it once.
    effect(() => {
      const admin = this.auth.isAdmin();
      untracked(() => this.loadBenchmark(admin));
    });
  }

  question = '';
  collectionId: number | null = null;
  topK = 8;
  parallel = false;

  ngOnInit(): void {
    this.collectionService.list().subscribe({ next: (list) => this.collections.set(list) });
  }

  ngOnDestroy(): void {
    this.stop();
  }

  compare(): void {
    const question = this.question.trim();
    if (!question || this.state.running()) {
      return;
    }
    this.stop();
    for (const strategy of this.state.begin(question, this.parallel)) {
      this.start(strategy, question);
    }
  }

  private start(strategy: StrategyName, question: string): void {
    const column = this.state.column(strategy);
    const subscription = this.asker.run(
      column.answer,
      question,
      { strategy, top_k: this.topK, collection_id: this.collectionId ?? undefined },
      () => {
        const done = column.answer.done();
        if (done !== null) {
          // Cost is only on the stored run (D15); a failed read leaves it n/a.
          this.runs.get(done.run_id).subscribe({ next: (run) => column.run.set(run), error: () => undefined });
        }
        for (const next of this.state.finished(strategy)) {
          this.start(next, question);
        }
      },
    );
    this.subscriptions.push(subscription);
  }

  private stop(): void {
    this.state.cancel();
    for (const subscription of this.subscriptions) {
      subscription.unsubscribe();
    }
    this.subscriptions = [];
  }

  private loadBenchmark(admin: boolean): void {
    if (!admin) {
      this.state.benchmarkNote.set('Benchmark scores come from the evaluation API, which only admins can read.');
      return;
    }
    if (this.benchmarkRequested) {
      return;
    }
    this.benchmarkRequested = true;
    this.evaluation.runs().subscribe({
      next: (runs) => {
        const benchmark = latestBenchmark(runs);
        this.state.benchmark.set(benchmark);
        this.state.benchmarkNote.set(
          benchmark === null
            ? 'No evaluation batch has been run yet (ragfabric eval run), so there is no benchmark score.'
            : `Benchmark score: mean correctness in batch "${benchmark.batch}" on the shared question set, not a score for this question.`,
        );
      },
      error: (error: unknown) =>
        this.state.benchmarkNote.set(
          error instanceof RagFabricError && error.status === 404
            ? 'This server has no evaluation API yet, so there is no benchmark score.'
            : 'The benchmark scores could not be loaded.',
        ),
    });
  }

  /** Zero is a real count once the answer is done; before that nothing was measured. */
  cited(status: string, citations: readonly Citation[]): string {
    return status === 'done' ? String(citations.filter((c) => c.used).length) : 'n/a';
  }

  statusLabel(status: string): string {
    return { idle: 'waiting', running: 'answering', done: 'done', failed: 'failed' }[status] ?? status;
  }
}
