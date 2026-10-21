// Documents upload: a rejected file shows the server's own reason, per file,
// and the page copy matches what the server accepts.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DocumentsComponent } from './documents.component';

describe('DocumentsComponent', () => {
  let fixture: ComponentFixture<DocumentsComponent>;
  let http: HttpTestingController;

  const text = (selector: string) =>
    (fixture.nativeElement.querySelector(selector) as HTMLElement | null)?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(DocumentsComponent);
    fixture.detectChanges();
    http.expectOne((r) => r.url.startsWith('/api/documents') && r.method === 'GET').flush({ items: [], total: 0 });
    http.expectOne((r) => r.url.startsWith('/api/collections')).flush([]);
    fixture.detectChanges();
  });

  afterEach(() => http.verify());

  it('test_each_rejected_file_shows_the_servers_reason', () => {
    const component = fixture.componentInstance as unknown as { uploadFiles(files: File[]): void };
    component.uploadFiles([
      new File(['x'], 'malware.exe'),
      new File(['x'], 'fake.pdf'),
      new File(['x'], 'big.txt'),
    ]);
    const uploads = http.match((r) => r.url.startsWith('/api/documents/upload'));
    expect(uploads.length).toBe(3);
    uploads[0].flush({ detail: 'The .exe type is not accepted here.' }, { status: 400, statusText: 'Bad Request' });
    uploads[1].flush({ detail: 'The content is not a PDF.' }, { status: 415, statusText: 'Unsupported Media Type' });
    uploads[2].flush({ detail: 'The file is larger than 1 MB.' }, { status: 413, statusText: 'Payload Too Large' });
    http.match((r) => r.url.startsWith('/api/documents') && r.method === 'GET').forEach((r) => r.flush({ items: [], total: 0 }));
    fixture.detectChanges();

    const shown = text('[data-test="upload-errors"]');
    expect(shown).toContain('malware.exe: The .exe type is not accepted here.');
    expect(shown).toContain('fake.pdf: The content is not a PDF.');
    expect(shown).toContain('big.txt: The file is larger than 1 MB.');
    expect(shown).not.toContain('Failed to upload');
  });

  it('test_the_copy_lists_markdown_and_does_not_claim_an_offline_embedder', () => {
    const page = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(page).toContain('Markdown');
    expect(page).not.toContain('offline');
    expect(fixture.nativeElement.querySelector('input[type=file]').getAttribute('accept')).toContain('.md');
  });
});
