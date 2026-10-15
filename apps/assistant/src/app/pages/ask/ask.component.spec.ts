// Ask (Phase 9, Task 6): AUTO and MANUAL, the router card, the streamed answer,
// the corrected answer, clickable citations and the source viewer.

import { HttpEventType, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, TestRequest, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { Citation, RouterDecision } from '@ragfabric/sdk';

import { RunDetailsService } from '../../services/run-details.service';
import { AskComponent } from './ask.component';

const ROUTER: RouterDecision = {
  selected_strategy: 'vectorless',
  source: 'classifier',
  decisive: true,
  confidence: 0.9,
  reasoning: 'The question names an exact code.',
  query_type: 'exact_match',
  estimated_complexity: 'low',
  expected_cost_level: 'low',
  expected_latency_level: 'low',
};

const CITATIONS: Citation[] = [
  {
    marker: '[1]',
    chunk_id: 11,
    document_id: 4,
    filename: 'it-runbook.md',
    page: null,
    score: 0.812,
    snippet: 'BW-7731 means the quota is full. Restart the sync job.',
    used: true,
    highlights: [{ term: 'BW-7731', start: 0, end: 7 }],
    supporting_span: { text: 'BW-7731 means the quota is full.', start: 0, end: 32 },
  },
  { marker: '[2]', chunk_id: 12, document_id: 5, filename: 'org-chart.md', snippet: 'x', used: false },
];

function sse(events: [string, unknown][]): string {
  return events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join('');
}

function retrieval(overrides: Record<string, unknown> = {}) {
  return {
    chunks: 2,
    strategy: 'vectorless',
    trace: [],
    sub_questions: [],
    subgraph: null,
    router: ROUTER,
    fallback_from: null,
    ...overrides,
  };
}

describe('AskComponent', () => {
  let fixture: ComponentFixture<AskComponent>;
  let http: HttpTestingController;

  const el = (selector: string) => fixture.nativeElement.querySelector(selector) as HTMLElement | null;
  const text = (selector: string) => el(selector)?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideRouter([]), provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(AskComponent);
    fixture.detectChanges();
    http.expectOne('/api/collections').flush([]);
  });

  afterEach(() => http.verify());

  function ask(question: string, mode: 'auto' | 'manual' = 'auto', strategy = 'traditional'): TestRequest {
    const component = fixture.componentInstance;
    component.question = question;
    component.mode = mode;
    component.strategy = strategy as never;
    component.ask();
    fixture.detectChanges();
    return http.expectOne('/api/ask');
  }

  function stream(req: TestRequest, events: [string, unknown][]): void {
    const wire = sse(events);
    req.event({ type: HttpEventType.DownloadProgress, loaded: wire.length, partialText: wire });
    fixture.detectChanges();
  }

  function finish(req: TestRequest, events: [string, unknown][]): void {
    req.flush(sse(events));
    fixture.detectChanges();
  }

  it('test_auto_sends_strategy_auto_and_manual_sends_the_picked_one', () => {
    const auto = ask('what is BW-7731');
    expect(auto.request.body).toEqual({ strategy: 'auto', top_k: 8, query: 'what is BW-7731', stream: true });
    finish(auto, [['done', { run_id: 1, latency_ms: 5, usage: {} }]]);

    const manual = ask('what is BW-7731', 'manual', 'graph');
    expect(manual.request.body.strategy).toBe('graph');
    finish(manual, [['done', { run_id: 2, latency_ms: 5, usage: {} }]]);
  });

  it('test_tokens_stream_in_and_the_router_card_explains_the_choice', () => {
    const req = ask('what is BW-7731');
    stream(req, [
      ['retrieval', retrieval()],
      ['token', { text: 'It means the ' }],
    ]);
    expect(text('[data-test="answer"]')).toContain('It means the');
    expect(text('[data-test="strategy"]')).toBe('vectorless');
    expect(text('[data-test="source"]')).toBe('classifier');
    expect(text('[data-test="confidence"]')).toContain('90%');
    expect(text('[data-test="reasoning"]')).toBe('The question names an exact code.');

    finish(req, [
      ['retrieval', retrieval()],
      ['token', { text: 'It means the ' }],
      ['token', { text: 'quota is full [1].' }],
      ['citations', { citations: CITATIONS }],
      ['done', { run_id: 9, latency_ms: 1530, usage: { llm_calls: 2, retrieval_calls: 1, embedding_calls: 0, input_tokens: 10, output_tokens: 4 } }],
    ]);
    expect(text('[data-test="answer"]')).toBe('It means the quota is full [1].');
    expect(text('[data-test="meta"]')).toContain('1.53 s');
    expect(el('[data-test="trace-link"]')?.getAttribute('href')).toBe('/trace/9');
  });

  it('test_a_rule_decision_shows_no_invented_confidence_and_a_fallback_is_called_out', () => {
    const req = ask('who does Platform report to');
    finish(req, [
      [
        'retrieval',
        retrieval({
          strategy: 'traditional',
          fallback_from: 'graph',
          router: { ...ROUTER, source: 'signals', confidence: null, selected_strategy: 'graph' },
        }),
      ],
      ['done', { run_id: 3, latency_ms: 5, usage: {} }],
    ]);

    expect(text('[data-test="confidence"]')).toBe('Rule based, no confidence measured');
    expect(text('[data-test="fallback"]')).toContain('graph');
    expect(text('[data-test="meta"]')).toContain('n/a');
  });

  it('test_superseded_replaces_the_streamed_text_and_says_why', () => {
    const req = ask('q');
    stream(req, [['token', { text: 'Something unsupported.' }]]);
    finish(req, [
      ['token', { text: 'Something unsupported.' }],
      ['superseded', { text: 'Only this is supported [1].', reason: 'citation contract' }],
      ['citations', { citations: CITATIONS }],
      ['done', { run_id: 4, latency_ms: 5, usage: {} }],
    ]);

    expect(text('[data-test="answer"]')).toBe('Only this is supported [1].');
    expect(text('[data-test="superseded"]')).toContain('citation contract');
  });

  it('test_a_marker_click_opens_the_source_viewer_on_that_citation_and_downloads', () => {
    const req = ask('q');
    finish(req, [
      ['token', { text: 'Quota is full [1].' }],
      ['citations', { citations: CITATIONS }],
      ['done', { run_id: 5, latency_ms: 5, usage: {} }],
    ]);

    (el('.answer-text .marker') as HTMLButtonElement).click();
    fixture.detectChanges();

    expect(text('[data-test="filename"]')).toBe('it-runbook.md');
    expect(text('[data-test="score"]')).toBe('0.812');
    expect(el('[data-test="snippet"] .hl')?.textContent).toBe('BW-7731');
    expect(el('[data-test="snippet"] .support')).not.toBeNull();

    (el('[data-test="download"]') as HTMLButtonElement).click();
    http.expectOne('/api/documents/4/download').flush(new Blob(['x']));
  });

  it('test_cited_sources_are_shown_and_uncited_ones_folded_away_but_still_openable', () => {
    const req = ask('q');
    finish(req, [
      ['token', { text: 'A [1].' }],
      ['citations', { citations: [CITATIONS[1], CITATIONS[0]] }],
      ['done', { run_id: 6, latency_ms: 5, usage: {} }],
    ]);
    const cited = Array.from(fixture.nativeElement.querySelectorAll('[aria-label="Sources cited"] .chip')) as HTMLElement[];
    expect(cited.length).toBe(1);
    expect(cited[0].textContent).toContain('it-runbook.md');
    expect(text('.uncited summary')).toBe('1 more retrieved, not cited');
    (fixture.nativeElement.querySelector('.uncited .chip') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(text('[data-test="filename"]')).toBe('org-chart.md');
  });

  it('test_the_run_details_are_kept_for_the_trace_page', () => {
    const req = ask('what is BW-7731');
    finish(req, [
      ['retrieval', retrieval()],
      ['done', { run_id: 12, latency_ms: 5, usage: {} }],
    ]);
    const details = TestBed.inject(RunDetailsService).get(12);
    expect(details?.question).toBe('what is BW-7731');
    expect(details?.retrieval?.router?.reasoning).toBe(ROUTER.reasoning);
  });

  it('test_a_refused_request_shows_the_servers_reason', () => {
    const req = ask('q', 'manual', 'agentic');
    req.flush('{"detail":"The agentic strategy needs a language model."}', {
      status: 422,
      statusText: 'Unprocessable',
    });
    fixture.detectChanges();
    expect(text('[data-test="error"]')).toBe('The agentic strategy needs a language model.');
  });
});
