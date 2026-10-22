// The answer view on a graph answer: [E k] markers are chips, the path is
// collapsed, and an empty walk is explained.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Citation, Subgraph } from '@ragfabric/sdk';

import { AnswerState } from './answer-state';
import { AnswerViewComponent } from './answer-view.component';

const GRAPH: Subgraph = {
  nodes: [
    { id: 1, name: 'Anjali Rao', entity_type: 'person', depth: 0 },
    { id: 2, name: 'Rohan Iyer', entity_type: 'person', depth: 1 },
  ],
  edges: [
    { id: 11, source_id: 1, target_id: 2, relation_type: 'REPORTS_TO', walked_as: 'REPORTS_TO', reversed: false, confidence: 0.9, source_chunk_ids: [4] },
  ],
  truncated: false,
};
const CITATIONS: Citation[] = [
  { marker: '[1]', chunk_id: 4, document_id: 2, filename: 'org-chart.md', page: 1, snippet: 'Anjali Rao reports to Rohan Iyer.', used: true },
];

function retrieval(subgraph: Subgraph | null) {
  return { chunks: 1, strategy: subgraph ? 'graph' : 'traditional', trace: [], sub_questions: [], subgraph, router: null, fallback_from: null };
}

describe('AnswerViewComponent on a graph answer', () => {
  let fixture: ComponentFixture<AnswerViewComponent>;
  let state: AnswerState;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    fixture = TestBed.createComponent(AnswerViewComponent);
    state = new AnswerState();
  });

  function answer(subgraph: Subgraph | null, text: string): void {
    state.start();
    state.apply({ event: 'retrieval', data: retrieval(subgraph) } as never);
    state.apply({ event: 'token', data: { text } } as never);
    state.apply({ event: 'citations', data: { citations: CITATIONS } } as never);
    state.apply({ event: 'done', data: { run_id: 1, latency_ms: 5, refused: false, usage: {} } } as never);
    fixture.componentRef.setInput('state', state);
    fixture.detectChanges();
  }
  const q = (s: string) => fixture.nativeElement.querySelector(s) as HTMLElement | null;
  const modal = (s: string) => document.querySelector(s) as HTMLElement | null;

  it('test_an_edge_marker_is_a_chip_not_raw_text', () => {
    answer(GRAPH, 'Anjali reports to Rohan [E 1] [1].');
    const chip = q('[data-test="edge-marker"]') as HTMLButtonElement;

    expect(chip.tagName).toBe('BUTTON');
    expect(chip.textContent).toBe('[E 1]');
    expect(chip.getAttribute('aria-label')).toBe('Open relationship E 1');
    // The marker text exists only inside its button, never as loose text.
    const loose = Array.from(q('[data-test="answer"]')!.querySelectorAll('span')).map((s) => s.textContent).join('');
    expect(loose).not.toContain('[E 1]');
  });

  it('test_the_chip_opens_the_edge_and_the_chunk_that_backs_it_and_escape_closes_it', () => {
    answer(GRAPH, 'Anjali reports to Rohan [E 1] [1].');
    const chip = q('[data-test="edge-marker"]') as HTMLButtonElement;
    chip.focus();
    chip.click();
    fixture.detectChanges();

    expect(modal('[data-test="edge-reading"]')!.textContent).toContain('Anjali Rao');
    expect(modal('[data-test="backing-file"]')!.textContent).toBe('org-chart.md');
    expect(modal('[data-test="backing-text"]')!.textContent).toContain('Anjali Rao reports to Rohan Iyer.');

    modal('.modal')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    fixture.detectChanges();
    expect(modal('[data-test="edge-viewer"]')).toBeNull();
  });

  it('test_the_walked_path_sits_under_the_answer_collapsed', () => {
    answer(GRAPH, 'Anjali reports to Rohan [E 1].');
    const details = q('[data-test="graph-path"]') as HTMLDetailsElement;
    expect(details.open).toBeFalse();
    expect(details.textContent).toContain('2 entities');
  });

  it('test_a_marker_the_walk_does_not_have_stays_text', () => {
    answer(GRAPH, 'Anjali reports to Rohan [E 5].');
    expect(q('[data-test="edge-marker"]')).toBeNull();
    expect(q('[data-test="answer"]')!.textContent).toContain('[E 5]');
  });

  it('test_a_strategy_that_walks_no_graph_shows_no_path', () => {
    answer(null, 'Leave is 24 days [1].');
    expect(q('[data-test="graph-path"]')).toBeNull();
    expect(q('[data-test="graph-empty"]')).toBeNull();
  });

  it('test_an_empty_walk_gives_the_apis_reason', () => {
    answer({ nodes: [], edges: [], truncated: false, empty_reason: 'no_entity_matched' }, 'I could not find that.');
    expect(q('[data-test="graph-empty"]')!.textContent).toContain('No entity in the question matched the knowledge graph.');
    expect(q('[data-test="graph-path"]')).toBeNull();
  });
});
