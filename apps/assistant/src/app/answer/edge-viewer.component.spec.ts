// The edge behind an [E k] marker, and the chunk that backs it.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Citation } from '@ragfabric/sdk';

import { EdgeViewerComponent } from './edge-viewer.component';
import { WalkedEdge } from './graph';

function edge(chunks: number[], over: Partial<WalkedEdge['edge']> = {}): WalkedEdge {
  return {
    k: 2,
    from: 'Anjali Rao',
    to: 'Rohan Iyer',
    relation: 'REPORTS_TO',
    edge: { id: 11, source_id: 1, target_id: 2, relation_type: 'REPORTS_TO', walked_as: 'REPORTS_TO', reversed: false, confidence: 0.9, source_chunk_ids: chunks, ...over },
  };
}
const CITATION: Citation = {
  marker: '[1]',
  chunk_id: 4,
  filename: 'org-chart.md',
  page: 1,
  snippet: 'The Platform Team is led by Anjali Rao, who reports to Rohan Iyer.',
  used: true,
};

describe('EdgeViewerComponent', () => {
  let fixture: ComponentFixture<EdgeViewerComponent>;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(EdgeViewerComponent);
  });

  function open(e: WalkedEdge | null, citations: Citation[] = [CITATION]): void {
    fixture.componentRef.setInput('edge', e);
    fixture.componentRef.setInput('citations', citations);
    fixture.detectChanges();
  }
  const q = (s: string) => document.querySelector(s) as HTMLElement | null;
  const all = (s: string) => Array.from(document.querySelectorAll(s)) as HTMLElement[];

  it('test_nothing_is_drawn_while_no_edge_is_open', () => {
    open(null);
    expect(q('[data-test="edge-viewer"]')).toBeNull();
  });

  it('test_shows_the_edge_as_walked_with_confidence_and_the_cited_passage', () => {
    open(edge([4]));
    expect(q('[data-test="edge-reading"]')!.textContent).toContain('Anjali Rao');
    expect(q('[data-test="edge-reading"]')!.textContent).toContain('Rohan Iyer');
    expect(q('[data-test="edge-confidence"]')!.textContent).toBe('90%');
    expect(q('[data-test="backing-file"]')!.textContent).toBe('org-chart.md');
    expect(q('[data-test="backing-text"]')!.textContent).toContain('who reports to Rohan Iyer');
    expect(q('[data-test="backing"]')!.textContent).toContain('cited in the answer as [1]');
    http.expectNone('/api/chunks/4/context?neighbours=0');
  });

  it('test_a_backing_chunk_outside_the_answers_passages_is_read_from_the_store_and_labelled', () => {
    open(edge([8]));
    expect(q('[data-test="backing"]')!.textContent).toContain('Loading');

    http.expectOne('/api/chunks/8/context?neighbours=0').flush({
      chunk_id: 8,
      document_id: 3,
      filename: 'projects.md',
      chunk: { chunk_id: 8, chunk_index: 0, page: 2, text: 'Project Lighthouse is sponsored by Rohan Iyer.' },
      before: [],
      after: [],
    });
    fixture.detectChanges();

    expect(q('[data-test="backing-file"]')!.textContent).toBe('projects.md');
    expect(q('[data-test="backing"]')!.textContent).toContain('page 2');
    expect(q('[data-test="backing"]')!.textContent).toContain('not one of the passages the answer was written from');
    expect(q('[data-test="backing-text"]')!.textContent).toContain('sponsored by Rohan Iyer');
  });

  it('test_a_chunk_that_cannot_be_read_says_so_instead_of_inventing_text', () => {
    open(edge([8]));
    http.expectOne('/api/chunks/8/context?neighbours=0').flush({ detail: 'Not found' }, { status: 404, statusText: 'Not Found' });
    fixture.detectChanges();

    expect(q('[data-test="backing"] .alert-error')).not.toBeNull();
    expect(q('[data-test="backing-text"]')).toBeNull();
  });

  it('test_an_edge_with_no_readable_source_chunk_says_so', () => {
    open(edge([], { confidence: null }));
    expect(q('[data-test="no-backing"]')!.textContent).toContain('No source passage that you may read');
    expect(q('[data-test="edge-confidence"]')!.textContent).toBe('not measured');
    expect(all('[data-test="backing"]').length).toBe(0);
  });

  it('test_escape_closes_the_viewer', () => {
    open(edge([4]));
    let closed = false;
    fixture.componentInstance.closed.subscribe(() => (closed = true));
    q('.modal')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(closed).toBeTrue();
  });
});
