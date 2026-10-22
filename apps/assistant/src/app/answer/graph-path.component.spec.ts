// The walked path under a graph answer: collapsed, readable, honest when empty.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Citation, Subgraph } from '@ragfabric/sdk';

import { WalkedEdge } from './graph';
import { GraphPathComponent } from './graph-path.component';

const GRAPH: Subgraph = {
  nodes: [
    { id: 1, name: 'Anjali Rao', entity_type: 'person', depth: 0 },
    { id: 2, name: 'Rohan Iyer', entity_type: 'person', depth: 1 },
    { id: 3, name: 'Kavita Menon', entity_type: 'person', depth: 2 },
  ],
  edges: [
    { id: 11, source_id: 1, target_id: 2, relation_type: 'REPORTS_TO', walked_as: 'REPORTS_TO', reversed: false, confidence: 0.9, source_chunk_ids: [4] },
    { id: 12, source_id: 2, target_id: 3, relation_type: 'REPORTS_TO', walked_as: 'REPORTS_TO', reversed: false, confidence: null, source_chunk_ids: [8] },
  ],
  truncated: false,
};
const CITATION: Citation = { marker: '[1]', chunk_id: 4, filename: 'org-chart.md', page: 1, snippet: 'x', used: true };

describe('GraphPathComponent', () => {
  let fixture: ComponentFixture<GraphPathComponent>;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(GraphPathComponent);
  });

  function show(graph: Subgraph, citations: Citation[] = [CITATION]): void {
    fixture.componentRef.setInput('graph', graph);
    fixture.componentRef.setInput('citations', citations);
    fixture.detectChanges();
  }
  const q = (s: string) => fixture.nativeElement.querySelector(s) as HTMLElement | null;
  const all = (s: string) => Array.from(fixture.nativeElement.querySelectorAll(s)) as HTMLElement[];

  it('test_the_path_is_collapsed_by_default_with_a_count_in_the_summary', () => {
    show(GRAPH);
    const details = q('[data-test="graph-path"]') as HTMLDetailsElement;
    expect(details.open).toBeFalse();
    expect(details.querySelector('summary')!.textContent).toContain('3 entities');
    expect(details.querySelector('summary')!.textContent).toContain('2 relationships');
  });

  it('test_each_edge_shows_its_relation_confidence_and_a_short_source_reference', () => {
    show(GRAPH);
    const rows = all('[data-test="edge"]');
    expect(rows.length).toBe(2);
    expect(rows[0].textContent).toContain('Anjali Rao');
    expect(rows[0].textContent).toContain('reports to');
    expect(rows[0].textContent).toContain('Rohan Iyer');
    expect(rows[0].textContent).toContain('confidence 90%');
    expect(rows[0].querySelector('[data-test="edge-source"]')!.textContent).toBe('org-chart.md p1');
    // Unmeasured confidence is said, never shown as a number.
    expect(rows[1].textContent).toContain('confidence not measured');
  });

  it('test_a_source_not_among_the_cited_passages_is_named_once_the_path_is_opened', () => {
    show(GRAPH);
    http.expectNone('/api/chunks/8/context?neighbours=0');
    const details = q('[data-test="graph-path"]') as HTMLDetailsElement;
    details.open = true;
    details.dispatchEvent(new Event('toggle'));
    fixture.detectChanges();

    http.expectOne('/api/chunks/8/context?neighbours=0').flush({
      chunk_id: 8,
      document_id: 2,
      filename: 'projects.md',
      chunk: { chunk_id: 8, chunk_index: 0, page: 2, text: 'text' },
      before: [],
      after: [],
    });
    fixture.detectChanges();

    expect(all('[data-test="edge-source"]')[1].textContent).toBe('projects.md p2');
  });

  it('test_an_edge_chip_is_a_button_that_emits_its_edge', () => {
    show(GRAPH);
    // A closed <details> hides its chips from the keyboard; opening it is the summary's job.
    (q('[data-test="graph-path"]') as HTMLDetailsElement).open = true;
    let opened: WalkedEdge | null = null;
    fixture.componentInstance.openEdge.subscribe((e) => (opened = e));
    const chip = all('[data-test="edge-chip"]')[1] as HTMLButtonElement;

    expect(chip.tagName).toBe('BUTTON');
    expect(chip.textContent).toBe('[E 2]');
    chip.focus();
    expect(document.activeElement).toBe(chip);
    chip.click();

    expect(opened!.k).toBe(2);
    expect(opened!.to).toBe('Kavita Menon');
  });

  for (const [reason, words] of [
    ['no_graph_coverage', 'No knowledge graph covers these documents yet.'],
    ['no_entity_matched', 'No entity in the question matched the knowledge graph.'],
    ['no_walkable_edges', 'The matched entities have no relationships to follow.'],
  ] as const) {
    it(`test_empty_reason_${reason}_is_shown_in_plain_words_and_no_path_is_drawn`, () => {
      show({ nodes: [], edges: [], truncated: false, empty_reason: reason });
      expect(q('[data-test="graph-empty"]')!.textContent).toContain(words);
      expect(q('[data-test="graph-path"]')).toBeNull();
    });
  }

  it('test_a_walk_with_no_edges_and_no_reason_still_says_so', () => {
    show({ nodes: [], edges: [], truncated: false });
    expect(q('[data-test="graph-empty"]')!.textContent).toContain('The walk kept no relationships.');
  });

  it('test_a_truncated_walk_says_the_budget_cut_it', () => {
    show({ ...GRAPH, truncated: true });
    expect(q('.note')!.textContent).toContain('node budget');
  });
});
