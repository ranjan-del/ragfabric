// Trace (Phase 9, Task 8).

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { Run } from '@ragfabric/sdk';

import { RunDetailsService } from '../../services/run-details.service';
import { TraceComponent } from './trace.component';
import { waterfall } from './waterfall';

const RUN: Run = {
  id: 9,
  question: 'Who does Platform report to?',
  mode: 'auto',
  requested_strategy: 'auto',
  selected_strategy: 'graph',
  fallback_from: null,
  router_confidence: 0.82,
  router_reasoning: 'Names a relationship.',
  answer: 'Platform reports to Engineering [1].',
  latency_ms: 1000,
  retrieval_latency_ms: 400,
  generation_latency_ms: 600,
  llm_calls: 2,
  retrieval_calls: 1,
  input_tokens: 120,
  output_tokens: 30,
  estimated_cost_usd: null,
  llm_model: 'llama3.1:8b',
  embedding_model: 'nomic-embed-text',
  trace: [
    { name: 'answer_stream', started_ms: 400, duration_ms: 600, attributes: {} },
    { name: 'match_entities', started_ms: 0, duration_ms: 300, attributes: { matched: 2 } },
  ],
  created_at: '2026-10-10T10:00:00Z',
  sources: [
    { rank: 1, chunk_id: 11, document_id: 3, score: null, cited: true, page: null },
    { rank: 2, chunk_id: 12, document_id: 4, score: 0.5, cited: false, page: 2 },
  ],
};

describe('waterfall', () => {
  it('test_spans_in_start_order_with_widths_proportional_to_duration', () => {
    const bars = waterfall(RUN.trace, 1000);
    expect(bars.map((b) => b.span.name)).toEqual(['match_entities', 'answer_stream']);
    expect(bars[0].left).toBe(0);
    expect(bars[0].width).toBe(30);
    expect(bars[1].left).toBe(40);
    expect(bars[1].width).toBe(60);
  });

  it('test_the_axis_stretches_to_the_last_span_and_junk_is_left_out', () => {
    const bars = waterfall([{ name: 'a', started_ms: 0, duration_ms: 2000 }, { nope: true }], 1000);
    expect(bars.length).toBe(1);
    expect(bars[0].width).toBe(100);
  });
});

describe('TraceComponent', () => {
  let fixture: ComponentFixture<TraceComponent>;
  let http: HttpTestingController;
  const text = (selector: string) =>
    (fixture.nativeElement.querySelector(selector) as HTMLElement | null)?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

  function load(id: string, run: Run | null, status = 200): void {
    TestBed.configureTestingModule({
      providers: [provideRouter([]), provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(TraceComponent);
    fixture.componentRef.setInput('runId', id);
  }

  function flush(run: Run | null, status = 200): void {
    fixture.detectChanges();
    const req = http.expectOne(`/api/runs/${fixture.componentRef.instance.runId()}`);
    if (status === 200) {
      req.flush(run);
    } else {
      req.flush({ detail: 'Run not found.' }, { status, statusText: 'x' });
    }
    fixture.detectChanges();
  }

  afterEach(() => http.verify());

  it('test_the_stored_run_is_drawn_with_routing_cost_and_sources', () => {
    load('9', RUN);
    flush(RUN);

    expect(text('[data-test="question"]')).toBe(RUN.question);
    expect(text('[data-test="selected"]')).toBe('graph');
    expect(text('[data-test="router-confidence"]')).toBe('82%');
    expect(text('[data-test="router-reasoning"]')).toContain('Names a relationship.');
    expect(text('[data-test="latency"]')).toBe('1.00 s');
    expect(text('[data-test="cost"]')).toBe('n/a');
    const spans = Array.from(fixture.nativeElement.querySelectorAll('[data-test="span"]')) as HTMLElement[];
    expect(spans.map((s) => s.querySelector('strong')?.textContent)).toEqual(['match_entities', 'answer_stream']);
    expect(spans[0].textContent).toContain('matched=2');
    const sources = Array.from(fixture.nativeElement.querySelectorAll('[data-test="source"]')) as HTMLElement[];
    expect(sources[0].textContent).toContain('yes');
    expect(sources[0].textContent).toContain('n/a');
  });

  it('test_a_run_not_streamed_here_says_its_graph_path_is_not_stored', () => {
    load('9', RUN);
    flush(RUN);
    expect(text('[data-test="not-stored"]')).toContain('not stored with a run');
  });

  it('test_a_run_streamed_here_shows_its_graph_path', () => {
    load('9', RUN);
    TestBed.inject(RunDetailsService).remember(9, {
      question: RUN.question,
      superseded: null,
      retrieval: {
        chunks: 1,
        strategy: 'graph',
        trace: [],
        sub_questions: [],
        router: null,
        fallback_from: null,
        subgraph: {
          nodes: [
            { id: 1, name: 'Platform Team', entity_type: 'team', depth: 0 },
            { id: 2, name: 'Engineering', entity_type: 'department', depth: 1 },
          ],
          edges: [
            { id: 5, source_id: 1, target_id: 2, relation_type: 'REPORTS_TO', walked_as: 'REPORTS_TO', reversed: false, confidence: 0.9, source_chunk_ids: [11] },
          ],
          truncated: false,
        },
      },
    });
    flush(RUN);
    expect(text('[data-test="graph-path"]')).toContain('Platform Team REPORTS_TO Engineering');
  });

  it('test_a_manual_run_reports_no_routing_and_a_rule_reports_no_confidence', () => {
    load('9', RUN);
    flush({ ...RUN, mode: 'manual', router_confidence: null, router_reasoning: null });
    expect(text('[data-test="router-confidence"]')).toBe('n/a (not routed)');

    TestBed.resetTestingModule();
    load('9', RUN);
    flush({ ...RUN, router_confidence: null });
    expect(text('[data-test="router-confidence"]')).toBe('rule based, none measured');
  });

  it('test_a_missing_run_says_not_found', () => {
    load('77', null);
    flush(null, 404);
    expect(text('[data-test="error"]')).toContain('Run not found');
  });
});
