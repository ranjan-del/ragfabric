// Compare (Phase 9, Task 7): four explicit /api/ask calls, one column each.

import { HttpEventType, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, TestRequest, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { AuthService } from '../../services/auth.service';
import { CompareComponent } from './compare.component';

function sse(events: [string, unknown][]): string {
  return events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join('');
}

function answer(runId: number, latency: number, text = 'Atlas is owned by Priya [1].'): [string, unknown][] {
  return [
    ['token', { text }],
    ['citations', { citations: [{ marker: '[1]', snippet: 's', used: true, filename: 'project-atlas.md' }] }],
    ['done', { run_id: runId, latency_ms: latency, usage: { llm_calls: 1, retrieval_calls: 1, embedding_calls: 1, input_tokens: 5, output_tokens: 3 } }],
  ];
}

describe('CompareComponent', () => {
  let fixture: ComponentFixture<CompareComponent>;
  let http: HttpTestingController;
  const text = (selector: string) =>
    (fixture.nativeElement.querySelector(selector) as HTMLElement | null)?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

  function setup(admin: boolean): void {
    TestBed.configureTestingModule({
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: AuthService, useValue: { isAdmin: () => admin } },
      ],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(CompareComponent);
    fixture.detectChanges();
    http.expectOne('/api/collections').flush([]);
  }

  afterEach(() => http.verify());

  function compare(parallel = false): void {
    fixture.componentInstance.question = 'Who owns Atlas?';
    fixture.componentInstance.parallel = parallel;
    fixture.componentInstance.compare();
    fixture.detectChanges();
  }

  function next(strategy: string): TestRequest {
    const req = http.expectOne((r) => r.url === '/api/ask' && r.body.strategy === strategy);
    expect(req.request.body).toEqual({ strategy, top_k: 8, query: 'Who owns Atlas?', stream: true });
    return req;
  }

  function finish(req: TestRequest, runId: number, latency: number, cost: number | null): void {
    req.flush(sse(answer(runId, latency)));
    http.expectOne(`/api/runs/${runId}`).flush({ id: runId, estimated_cost_usd: cost });
    fixture.detectChanges();
  }

  it('test_sequential_compare_sends_four_explicit_strategies_one_after_another', () => {
    setup(false);
    compare();

    const traditional = next('traditional');
    http.expectNone((r) => r.url === '/api/ask' && r.body.strategy === 'vectorless');
    finish(traditional, 1, 900, 0.002);
    finish(next('vectorless'), 2, 120, null);
    finish(next('agentic'), 3, 4000, 0.009);
    finish(next('graph'), 4, 700, 0.001);

    expect(text('[data-test="column-vectorless"] [data-test="latency"]')).toBe('120 ms');
    expect(text('[data-test="column-vectorless"] [data-test="cost"]')).toBe('n/a');
    expect(text('[data-test="column-graph"] [data-test="cost"]')).toBe('$0.00100');
    expect(text('[data-test="column-vectorless"]')).toContain('fastest');
    expect(text('[data-test="column-graph"]')).toContain('cheapest');
    expect(text('[data-test="column-traditional"]')).toContain('Atlas is owned by Priya');
    expect(text('[data-test="benchmark-note"]')).toContain('only admins');
    expect(text('[data-test="column-traditional"] [data-test="benchmark"]')).toBe('n/a');
  });

  it('test_one_strategy_failing_shows_its_reason_and_the_others_complete', () => {
    setup(false);
    compare(true);
    expect(text('[data-test="contention"]')).toContain('all four strategies running at once');

    const reqs = ['traditional', 'vectorless', 'agentic', 'graph'].map((s) => next(s));
    reqs[2].flush('{"detail":"The agentic strategy needs a language model."}', { status: 422, statusText: 'x' });
    finish(reqs[0], 1, 10, null);
    finish(reqs[1], 2, 20, null);
    finish(reqs[3], 3, 30, null);

    expect(text('[data-test="column-agentic"] [role="alert"]')).toBe('The agentic strategy needs a language model.');
    expect(text('[data-test="column-graph"]')).toContain('done');
  });

  it('test_admins_see_the_latest_batch_benchmark_score', () => {
    setup(true);
    http.expectOne('/api/eval/runs?limit=100').flush([
      { id: 1, name: 'b1', strategy: 'traditional', started_at: '2026-10-09T00:00:00Z', summary: { means: { correctness: 0.61 } } },
      { id: 2, name: 'b1', strategy: 'graph', started_at: '2026-10-09T00:01:00Z', summary: { means: { correctness: 0.74 } } },
    ]);
    compare();
    fixture.detectChanges();

    expect(text('[data-test="column-traditional"] [data-test="benchmark"]')).toBe('0.61');
    expect(text('[data-test="column-graph"] [data-test="best"]')).toBe('best benchmark');
    expect(text('[data-test="column-agentic"] [data-test="benchmark"]')).toBe('n/a');
    expect(text('[data-test="benchmark-note"]')).toContain('not a score for this question');
    next('traditional').flush(sse([]));
    fixture.detectChanges();
    next('vectorless');
  });

  it('test_a_server_without_the_evaluation_api_says_so', () => {
    setup(true);
    http.expectOne('/api/eval/runs?limit=100').flush({ detail: 'Not Found' }, { status: 404, statusText: 'Not Found' });
    compare();
    expect(text('[data-test="benchmark-note"]')).toContain('no evaluation API yet');
    next('traditional');
  });

  it('test_streaming_text_shows_in_its_column_while_it_arrives', () => {
    setup(false);
    compare();
    const req = next('traditional');
    const wire = sse([['token', { text: 'Partial ' }]]);
    req.event({ type: HttpEventType.DownloadProgress, loaded: wire.length, partialText: wire });
    fixture.detectChanges();
    expect(text('[data-test="column-traditional"] [data-test="answer"]')).toBe('Partial');
    expect(text('[data-test="column-vectorless"]')).toContain('Waiting for its turn');
  });
});
