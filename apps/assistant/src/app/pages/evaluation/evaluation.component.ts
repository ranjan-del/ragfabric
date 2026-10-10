import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { EvalDashboard, EvalRun, QualityPoint, RagFabricError } from '@ragfabric/sdk';

import { count, ms, percent, score, usd } from '../../answer/format';
import { METRICS, batchNames, runsOf, trendPath, trendsByTarget } from '../../evaluation/batches';
import { describeError } from '../../services/http-error';
import { EvaluationService } from '../../services/evaluation.service';
import { BarChartComponent, BarRow } from '../../ui';

function unavailable(error: unknown, what: string): string {
  if (error instanceof RagFabricError && error.status === 404) {
    return 'This server has no evaluation API yet. It arrives with RagFabric 0.5.0 (ragfabric eval).';
  }
  return describeError(error, `The ${what} could not be loaded.`);
}

/**
 * Evaluation: the latest benchmark batch, a per category breakdown, and the
 * operational dashboards. Admin only, like the API it reads.
 *
 * Built against Phase 8's planned API (decision D20) and to be re-verified
 * when Phase 8 merges. Every value the API leaves null is shown as "n/a".
 */
@Component({
  selector: 'app-evaluation',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, BarChartComponent],
  template: `
    <div class="page-head">
      <h1>Evaluation</h1>
      <p>How each strategy scored on the shared question set, and how the live system is behaving.</p>
    </div>

    @if (runsError()) {
      <p class="alert alert-info" data-test="runs-error">{{ runsError() }}</p>
    } @else if (loaded() && names().length === 0) {
      <p class="alert alert-info" data-test="no-batches">No evaluation batch has been run yet. Run <code>make eval</code> or <code>ragfabric eval run</code>.</p>
    } @else if (names().length > 0) {
      <section class="card block" aria-label="Latest batch">
        <div class="head">
          <h2>Batch</h2>
          <select name="batch" [ngModel]="batch()" (ngModelChange)="batch.set($event)" data-test="batch-select">
            @for (b of names(); track b) {<option [value]="b">{{ b }}</option>}
          </select>
        </div>
        @if (first(); as f) {
          <p class="muted small" data-test="batch-meta">
            {{ f.started_at ?? 'n/a' }} · commit {{ f.commit || 'n/a' }} · LLM {{ f.llm_model || 'n/a' }} ·
            embeddings {{ f.embedding_model || 'n/a' }} · judge {{ judge(f) }}
          </p>
        }
        <p class="muted small">One run on one machine, not a benchmark: scores are means over the questions each target answered.</p>
        <div class="scroll">
          <table class="grid-table" data-test="batch-table">
            <thead>
              <tr>
                <th>Target</th><th>Questions</th>
                @for (m of metrics; track m.key) {<th>{{ m.label }}</th>}
                <th>p50</th><th>p95</th><th>Cost (estimate)</th><th>Errors</th>
              </tr>
            </thead>
            <tbody>
              @for (run of batchRuns(); track run.id) {
                @if (run.summary.skipped) {
                  <tr data-test="skipped-row">
                    <th scope="row">{{ run.target }}</th>
                    <td [attr.colspan]="metrics.length + 5" class="muted">Skipped: {{ run.summary.skipped }}</td>
                  </tr>
                } @else {
                  <tr data-test="target-row">
                    <th scope="row">{{ run.target }}</th>
                    <td>{{ count(run.summary.questions) }}</td>
                    @for (m of metrics; track m.key) {<td>{{ score(run.summary.metrics?.[m.key]) }}</td>}
                    <td>{{ ms(run.summary.latency_ms?.p50) }}</td>
                    <td>{{ ms(run.summary.latency_ms?.p95) }}</td>
                    <td>{{ usd(run.summary.estimated_cost_usd) }}@if (run.summary.cost_unknown) {<span class="muted"> ({{ run.summary.cost_unknown }} unpriced)</span>}</td>
                    <td>{{ count(run.summary.errors) }}</td>
                  </tr>
                }
              }
            </tbody>
          </table>
        </div>
      </section>

      <section class="card block" aria-label="Per category">
        <div class="head">
          <h2>Per category</h2>
          <select name="target" [ngModel]="target()" (ngModelChange)="target.set($event)" data-test="target-select">
            @for (run of scoredRuns(); track run.id) {<option [value]="run.target">{{ run.target }}</option>}
          </select>
        </div>
        <div class="scroll">
          <table class="grid-table" data-test="category-table">
            <thead><tr><th>Category</th><th>Questions</th>@for (m of metrics; track m.key) {<th>{{ m.label }}</th>}</tr></thead>
            <tbody>
              @for (row of categories(); track row[0]) {
                <tr data-test="category-row">
                  <th scope="row">{{ row[0] }}</th>
                  <td>{{ count(row[1].questions) }}</td>
                  @for (m of metrics; track m.key) {<td>{{ score(row[1][m.key]) }}</td>}
                </tr>
              }
            </tbody>
          </table>
        </div>
      </section>
    }

    <section class="dash" aria-label="Dashboards">
      <div class="head">
        <h2>Live system</h2>
        <select name="days" [ngModel]="days()" (ngModelChange)="changeDays($event)" data-test="days">
          <option [ngValue]="7">7 days</option>
          <option [ngValue]="30">30 days</option>
          <option [ngValue]="90">90 days</option>
        </select>
      </div>
      @if (dashboardError()) {
        <p class="alert alert-info" data-test="dashboard-error">{{ dashboardError() }}</p>
      } @else if (dashboard(); as d) {
        <div class="tiles">
          <section class="card block" data-test="latency-chart">
            <h3>Latency p50 per strategy</h3>
            <ui-bar-chart [rows]="latencyRows('p50')" [format]="msFormat" label="Latency p50" />
            <h3>Latency p95 per strategy</h3>
            <ui-bar-chart [rows]="latencyRows('p95')" [format]="msFormat" label="Latency p95" />
          </section>
          <section class="card block" data-test="cost-chart">
            <h3>Estimated cost per day</h3>
            <ui-bar-chart [rows]="costRows()" [format]="usdFormat" label="Cost per day" />
          </section>
          <section class="card block" data-test="fallback">
            <h3>Fallback rate</h3>
            <p class="big">{{ percent(d.fallback_rate.rate) }}</p>
            <p class="muted small">{{ d.fallback_rate.fallbacks }} of {{ d.fallback_rate.runs }} runs in the last {{ d.window_days }} days fell back to traditional.</p>
            <h3>LLM calls per strategy</h3>
            <ui-bar-chart [rows]="callRows()" [format]="countFormat" label="LLM calls per strategy" />
          </section>
          <section class="card block" data-test="quality-trend">
            <h3>Quality trend (mean correctness per batch)</h3>
            @for (t of trends(); track t.target) {
              <div class="trend">
                <span class="trend-label">{{ t.target }}</span>
                <svg viewBox="0 0 200 40" preserveAspectRatio="none" role="img" [attr.aria-label]="t.target + ' correctness over batches'">
                  <path [attr.d]="path(t.points)" fill="none" stroke="currentColor" stroke-width="2" />
                </svg>
                <span class="muted">{{ score(lastPoint(t.points)) }}</span>
              </div>
            } @empty {
              <p class="muted">No evaluation batches yet.</p>
            }
          </section>
        </div>
      } @else {
        <div class="empty-state"><span class="spinner spinner-brand"></span></div>
      }
    </section>
  `,
  styles: `
    :host { display: grid; gap: var(--space-4); }
    .block { padding: var(--space-4); display: grid; gap: var(--space-3); align-content: start; }
    .head { display: flex; justify-content: space-between; align-items: center; gap: var(--space-3); }
    h2 { margin: 0; font-size: 1.05rem; }
    h3 { margin: 0; font-size: 0.9rem; }
    select { font: inherit; color: var(--text); background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius-sm); padding: 0.3rem 0.5rem; }
    .small { font-size: 0.8rem; margin: 0; }
    .scroll { overflow-x: auto; }
    .grid-table { border-collapse: collapse; font-size: 0.8rem; width: 100%; }
    .grid-table th, .grid-table td { padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
    .grid-table th[scope='row'], .grid-table thead th:first-child { text-align: left; }
    .dash { display: grid; gap: var(--space-3); }
    .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); gap: var(--space-4); }
    .big { font-size: 1.8rem; margin: 0; }
    .trend { display: grid; grid-template-columns: 7rem 1fr 3rem; gap: var(--space-2); align-items: center; color: var(--brand); }
    .trend-label { color: var(--text); font-size: 0.85rem; }
    .trend svg { width: 100%; height: 2.5rem; background: var(--surface-2); border-radius: 6px; }
  `,
})
export class EvaluationComponent implements OnInit {
  private readonly evaluation = inject(EvaluationService);

  readonly metrics = METRICS;
  readonly runs = signal<EvalRun[]>([]);
  readonly loaded = signal(false);
  readonly runsError = signal<string | null>(null);
  readonly batch = signal('');
  readonly target = signal('');
  readonly dashboard = signal<EvalDashboard | null>(null);
  readonly dashboardError = signal<string | null>(null);
  readonly days = signal(30);

  readonly count = count;
  readonly ms = ms;
  readonly score = score;
  readonly usd = usd;
  readonly percent = percent;
  readonly msFormat = (v: number) => ms(v);
  readonly usdFormat = (v: number) => usd(v);
  readonly countFormat = (v: number) => count(v);

  readonly names = computed(() => batchNames(this.runs()));
  readonly trends = computed(() => trendsByTarget(this.dashboard()?.quality_trend ?? []));
  readonly batchRuns = computed(() => runsOf(this.runs(), this.batch()));
  readonly scoredRuns = computed(() => this.batchRuns().filter((r) => !r.summary.skipped));
  // A skipped target never ran, so its row carries no models or commit worth showing.
  readonly first = computed(() => this.scoredRuns()[0] ?? this.batchRuns()[0] ?? null);
  readonly categories = computed(() => {
    const run = this.scoredRuns().find((r) => r.target === this.target());
    return Object.entries(run?.summary.by_category ?? {});
  });

  ngOnInit(): void {
    this.evaluation.runs().subscribe({
      next: (runs) => {
        this.runs.set(runs);
        this.loaded.set(true);
        const latest = batchNames(runs)[0] ?? '';
        this.batch.set(latest);
        this.target.set(this.scoredRuns()[0]?.target ?? '');
      },
      error: (error: unknown) => this.runsError.set(unavailable(error, 'evaluation runs')),
    });
    this.loadDashboard();
  }

  changeDays(days: number): void {
    this.days.set(days);
    this.loadDashboard();
  }

  private loadDashboard(): void {
    this.dashboard.set(null);
    this.dashboardError.set(null);
    this.evaluation.dashboard(this.days()).subscribe({
      next: (d) => this.dashboard.set(d),
      error: (error: unknown) => this.dashboardError.set(unavailable(error, 'dashboard')),
    });
  }

  judge(run: EvalRun): string {
    const meta = run.summary.meta;
    const kind = meta?.judge_kind ?? run.judge ?? 'n/a';
    const model = meta?.judge_model ? ` (${meta.judge_model})` : '';
    const version = meta?.judge_prompt_version ? `, rubric ${meta.judge_prompt_version}` : '';
    return `${kind}${model}${version}`;
  }

  latencyRows(which: 'p50' | 'p95'): BarRow[] {
    return Object.entries(this.dashboard()?.latency_ms ?? {}).map(([strategy, l]) => ({
      label: strategy,
      value: which === 'p50' ? l.p50 : l.p95,
      note: `${l.runs} runs`,
    }));
  }

  costRows(): BarRow[] {
    return (this.dashboard()?.cost_per_day ?? []).map((c) => ({
      label: c.date,
      value: c.estimated_cost_usd,
      note: c.unpriced_runs > 0 ? `${c.unpriced_runs} unpriced` : undefined,
    }));
  }

  callRows(): BarRow[] {
    return Object.entries(this.dashboard()?.calls_per_strategy ?? {}).map(([strategy, c]) => ({
      label: strategy,
      value: c.llm_calls,
      note: `${count(c.retrieval_calls)} retrieval, ${c.runs} runs`,
    }));
  }

  path(points: readonly QualityPoint[]): string {
    return trendPath(points, 200, 40);
  }

  lastPoint(points: readonly QualityPoint[]): number | null {
    return points.length === 0 ? null : (points[points.length - 1]?.correctness ?? null);
  }
}
