// Evaluation and dashboards (Phase 9, Task 9), against fakes shaped like
// Phase 8's planned API (decision D20). Re-verify when Phase 8 merges.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { EvalDashboard, EvalRun } from '@ragfabric/sdk';

import { trendPath } from '../../evaluation/batches';
import { EvaluationComponent } from './evaluation.component';

const META = { judge_kind: 'llm', judge_model: 'llama3.1:8b', judge_prompt_version: 'v1' };
const RUNS: EvalRun[] = [
  {
    id: 1, batch: 'old', target: 'traditional', started_at: '2026-10-01T00:00:00Z', commit: 'aaa',
    summary: { meta: META, status: 'finished', questions: 24, metrics: { correctness: 0.1 } },
  },
  {
    id: 2, batch: 'b2', target: 'traditional', started_at: '2026-10-09T00:00:00Z', commit: 'abc1234',
    llm_model: 'llama3.1:8b', embedding_model: 'nomic-embed-text',
    summary: {
      meta: META, status: 'finished', questions: 24, errors: 1,
      metrics: { precision: 0.42, recall: 0.8, hit: 0.9, reciprocal_rank: 0.66, correctness: 0.71, faithfulness: null, context_relevance: 0.5, citation_correct: 0.88 },
      by_category: { exact_match: { questions: 3, correctness: 1.0 }, multi_hop: { questions: 3, correctness: null } },
      latency_ms: { p50: 820, p95: 2400 }, estimated_cost_usd: null, cost_unknown: 24,
    },
  },
  {
    id: 3, batch: 'b2', target: 'agentic', started_at: '2026-10-09T00:10:00Z',
    summary: { meta: META, status: 'skipped', skipped: 'the offline provider cannot run the agent' },
  },
];
const DASHBOARD: EvalDashboard = {
  window_days: 30,
  since: '2026-09-10T00:00:00+00:00',
  runs: 8,
  latency_ms: {
    traditional: { p50: 800, p95: 2000, runs: 40 },
    graph: { p50: null, p95: null, runs: 0 },
  },
  cost_per_day: [{ date: '2026-10-09', estimated_cost_usd: null, runs: 12, unpriced_runs: 12 }],
  calls_per_strategy: { agentic: { llm_calls: 30, retrieval_calls: 12, runs: 6 } },
  fallback_rate: { rate: 0.25, fallbacks: 2, runs: 8 },
  quality_trend: [
    { run_id: 1, batch: 'old', target: 'traditional', started_at: 'x', judge: 'llm', hit: null, reciprocal_rank: null, correctness: 0.1, citation_correct: null },
    { run_id: 2, batch: 'b2', target: 'traditional', started_at: 'y', judge: 'llm', hit: null, reciprocal_rank: null, correctness: 0.71, citation_correct: null },
  ],
};

describe('EvaluationComponent', () => {
  let fixture: ComponentFixture<EvaluationComponent>;
  let http: HttpTestingController;
  const all = (s: string) => Array.from(fixture.nativeElement.querySelectorAll(s)) as HTMLElement[];
  const text = (s: string) =>
    (fixture.nativeElement.querySelector(s) as HTMLElement | null)?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(EvaluationComponent);
    fixture.detectChanges();
  });

  afterEach(() => http.verify());

  function load(runs: EvalRun[] | null = RUNS, dashboard: EvalDashboard | null = DASHBOARD, status = 200): void {
    const r = http.expectOne('/api/eval/runs?limit=100');
    const d = http.expectOne('/api/eval/dashboard?days=30');
    if (status === 200) {
      r.flush(runs);
      d.flush(dashboard);
    } else {
      r.flush({ detail: 'Not Found' }, { status, statusText: 'x' });
      d.flush({ detail: 'Not Found' }, { status, statusText: 'x' });
    }
    fixture.detectChanges();
  }

  it('test_the_latest_batch_has_one_row_per_target_with_skipped_ones_and_the_judge', () => {
    load();
    expect(fixture.componentInstance.batch()).toBe('b2');
    const rows = all('[data-test="target-row"]');
    expect(rows.length).toBe(1);
    expect(rows[0].textContent).toContain('traditional');
    expect(rows[0].textContent).toContain('0.71');
    expect(rows[0].textContent).toContain('820 ms');
    expect(rows[0].textContent).toContain('24 unpriced');
    expect(text('[data-test="skipped-row"]')).toContain('Skipped: the offline provider cannot run the agent');
    expect(text('[data-test="batch-meta"]')).toContain('llm (llama3.1:8b), rubric v1');
    expect(text('[data-test="batch-meta"]')).toContain('abc1234');
  });

  it('test_unmeasured_values_are_na_never_zero', () => {
    load();
    const row = all('[data-test="target-row"]')[0];
    expect(row.textContent).toContain('n/a');
    expect(text('[data-test="cost-chart"]')).toContain('n/a');
    expect(text('[data-test="cost-chart"]')).toContain('12 unpriced');
    expect(text('[data-test="latency-chart"]')).toContain('n/a');
  });

  it('test_the_per_category_breakdown_follows_the_chosen_target', () => {
    load();
    const rows = all('[data-test="category-row"]');
    expect(rows.map((r) => r.querySelector('th')?.textContent)).toEqual(['exact_match', 'multi_hop']);
    expect(rows[0].textContent).toContain('1.00');
    expect(rows[1].textContent).toContain('n/a');
  });

  it('test_the_four_dashboards_render', () => {
    load();
    expect(all('[data-test="latency-chart"] [data-test="bar-row"]').length).toBe(4);
    expect(text('[data-test="fallback"]')).toContain('25%');
    expect(text('[data-test="fallback"]')).toContain('2 of 8 runs in the last 30 days');
    expect(text('[data-test="fallback"]')).toContain('agentic');
    expect(fixture.nativeElement.querySelector('[data-test="quality-trend"] path').getAttribute('d')).toBe('M0.0,36.0 L200.0,11.6');
  });

  it('test_a_server_without_the_evaluation_api_says_so', () => {
    load(null, null, 404);
    expect(text('[data-test="runs-error"]')).toContain('no evaluation API yet');
    expect(text('[data-test="dashboard-error"]')).toContain('no evaluation API yet');
  });

  it('test_no_batches_yet_points_at_the_command', () => {
    load([]);
    expect(text('[data-test="no-batches"]')).toContain('make eval');
  });
});

describe('trendPath', () => {
  it('test_a_null_point_breaks_the_line_instead_of_dropping_to_zero', () => {
    const path = trendPath(
      [{ correctness: 0.5 }, { correctness: null }, { correctness: 1 }],
      100,
      10,
    );
    expect(path).toBe('M0.0,5.0 M100.0,0.0');
  });
});
