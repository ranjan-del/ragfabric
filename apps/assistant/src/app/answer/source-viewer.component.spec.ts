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
    open({ ...CITATION, filename: 'policy.pdf' });
    expect(q('[data-test="expand"]')).toBeNull();
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
