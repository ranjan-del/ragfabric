// The Angular transport for @ragfabric/sdk (Phase 9, Task 4).
//
// The SDK describes every call; this service executes it through HttpClient
// (decision D5), so the auth interceptor still adds the token and redirects on
// a 401, and specs keep HttpTestingController.

import { HttpEventType, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { AskEvent, RagFabricError, requests } from '@ragfabric/sdk';

import { authInterceptor } from '../interceptors/auth.interceptor';
import { SdkHttp } from './sdk-http.service';

describe('SdkHttp', () => {
  let sdk: SdkHttp;
  let http: HttpTestingController;

  beforeEach(() => {
    localStorage.setItem('rag_token', 'jwt-123');
    TestBed.configureTestingModule({
      providers: [
        provideRouter([]),
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    sdk = TestBed.inject(SdkHttp);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
    localStorage.removeItem('rag_token');
  });

  it('test_send_issues_the_builders_method_url_and_body_through_the_interceptor', () => {
    let result: unknown;
    sdk.send(requests.admin.grants.create(1, 2, 'read')).subscribe((r) => (result = r));

    const req = http.expectOne('/api/admin/grants');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ group_id: 1, collection_id: 2, permission: 'read' });
    expect(req.request.headers.get('Authorization')).toBe('Bearer jwt-123');
    req.flush({ id: 5 });

    expect(result).toEqual({ id: 5 });
  });

  it('test_an_error_becomes_a_ragfabric_error_with_the_server_detail', () => {
    let error: unknown;
    sdk.send(requests.runs.get(3)).subscribe({ error: (e) => (error = e) });

    http
      .expectOne('/api/runs/3')
      .flush({ detail: 'Run not found.' }, { status: 404, statusText: 'Not Found' });

    expect(error instanceof RagFabricError).toBeTrue();
    expect((error as RagFabricError).status).toBe(404);
    expect((error as RagFabricError).detail).toBe('Run not found.');
  });

  it('test_stream_turns_download_progress_into_events_across_chunk_boundaries', () => {
    const events: AskEvent[] = [];
    let completed = false;
    sdk.stream(requests.askStream('q', { strategy: 'auto' })).subscribe({
      next: (e) => events.push(e),
      complete: () => (completed = true),
    });

    const req = http.expectOne('/api/ask');
    expect(req.request.body).toEqual({ strategy: 'auto', query: 'q', stream: true });
    expect(req.request.responseType).toBe('text');

    const wire =
      'event: token\ndata: {"text":"Le"}\n\n' +
      'event: token\ndata: {"text":"ave"}\n\n' +
      'event: done\ndata: {"run_id":4,"latency_ms":9,"usage":{}}\n\n';
    const cut = wire.indexOf('"ave"');
    req.event({ type: HttpEventType.DownloadProgress, loaded: cut, partialText: wire.slice(0, cut) });
    expect(events.map((e) => e.event)).toEqual(['token']);

    req.event({ type: HttpEventType.DownloadProgress, loaded: wire.length, partialText: wire });
    req.flush(wire);

    expect(events.map((e) => e.event)).toEqual(['token', 'token', 'done']);
    expect(completed).toBeTrue();
  });

  it('test_a_refused_stream_errors_with_the_parsed_detail', () => {
    let error: unknown;
    sdk.stream(requests.askStream('q', { strategy: 'agentic' })).subscribe({
      error: (e) => (error = e),
    });

    http
      .expectOne('/api/ask')
      .flush('{"detail":"Agentic needs a model."}', { status: 422, statusText: 'Unprocessable' });

    expect((error as RagFabricError).detail).toBe('Agentic needs a model.');
  });
});
