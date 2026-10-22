// The source viewer: the server cuts a snippet, so text formats can be widened
// from the original file on request.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Citation } from '@ragfabric/sdk';

import { SourceViewerComponent } from './source-viewer.component';

const FULL = 'Priya Nair is the Head of People Operations. '.repeat(4) + 'Aisha Bello leads Data Platform. ' + 'Closing note. '.repeat(4);
const AT = FULL.indexOf('Priya');
const START = FULL.indexOf('Priya Nair is the Head of People Operations. Priya');
const BODY = FULL.slice(START, START + 100);
const CITATION: Citation = {
  marker: '[3]',
  document_id: 9,
  filename: 'atlas.md',
  snippet: BODY + '...',
  used: true,
};

describe('SourceViewerComponent', () => {
  let fixture: ComponentFixture<SourceViewerComponent>;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(SourceViewerComponent);
  });

  function open(citation: Citation): void {
    fixture.componentRef.setInput('citation', citation);
    fixture.detectChanges();
  }
  const q = (s: string) => fixture.nativeElement.querySelector(s) as HTMLElement | null;

  it('test_a_cut_text_snippet_can_be_widened_to_show_the_cited_fact', async () => {
    expect(AT).toBe(0);
    open(CITATION);
    expect(q('[data-test="snippet"]')?.textContent).not.toContain('Aisha Bello');

    q('[data-test="expand"]')!.click();
    http.expectOne('/api/documents/9/download').flush(new Blob([FULL]));
    await new Promise((resolve) => setTimeout(resolve, 50));
    fixture.detectChanges();

    expect(q('[data-test="snippet"]')?.textContent).toContain('Aisha Bello leads Data Platform.');
    expect(q('[data-test="expand"]')).toBeNull();
  });

  it('test_no_expand_for_a_whole_passage_or_a_binary_format', () => {
    open({ ...CITATION, snippet: 'A complete short passage.' });
    expect(q('[data-test="expand"]')).toBeNull();
    // A PDF has no readable original, and without a chunk id there is nothing to ask for.
    open({ ...CITATION, filename: 'policy.pdf' });
    expect(q('[data-test="expand"]')).toBeNull();
  });

  it('test_a_cut_pdf_snippet_is_widened_from_the_stored_chunks', async () => {
    const chunkText = 'Intro words. ' + BODY + ' The hotel limit is 180 US dollars per night.';
    open({ ...CITATION, filename: 'policy.pdf', chunk_id: 31, snippet: BODY + '...' });
    expect(q('[data-test="snippet"]')?.textContent).not.toContain('hotel limit');

    q('[data-test="expand"]')!.click();
    const req = http.expectOne('/api/chunks/31/context?neighbours=1');
    req.flush({
      chunk_id: 31,
      document_id: 9,
      filename: 'policy.pdf',
      chunk: { chunk_id: 31, chunk_index: 2, page: 1, text: chunkText },
      before: [{ chunk_id: 30, chunk_index: 1, page: 1, text: 'Earlier page text. ' }],
      after: [{ chunk_id: 32, chunk_index: 3, page: 2, text: ' Later page text.' }],
    });
    fixture.detectChanges();

    const shown = q('[data-test="snippet"]')?.textContent ?? '';
    expect(shown).toContain('The hotel limit is 180 US dollars per night.');
    expect(shown).toContain('Earlier page text.');
    expect(shown).toContain('Later page text.');
    expect(q('[data-test="expand"]')).toBeNull();
    http.verify();
  });

  it('test_a_refused_chunk_read_is_reported_not_swallowed', () => {
    open({ ...CITATION, filename: 'deck.pptx', chunk_id: 44 });
    q('[data-test="expand"]')!.click();
    http.expectOne('/api/chunks/44/context?neighbours=1').flush({ detail: 'Chunk not found.' }, { status: 404, statusText: 'Not Found' });
    fixture.detectChanges();
    expect(q('[role="alert"]')).not.toBeNull();
  });

  it('test_says_so_when_the_passage_is_not_in_the_file', async () => {
    open(CITATION);
    q('[data-test="expand"]')!.click();
    http.expectOne('/api/documents/9/download').flush(new Blob(['a different file']));
    await new Promise((resolve) => setTimeout(resolve, 50));
    fixture.detectChanges();
    expect(q('[role="alert"]')?.textContent).toContain('could not be found');
  });
});
