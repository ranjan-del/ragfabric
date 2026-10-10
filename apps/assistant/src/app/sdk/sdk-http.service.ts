import {
  HttpClient,
  HttpDownloadProgressEvent,
  HttpErrorResponse,
  HttpEventType,
} from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { ApiRequest, AskEvent, RagFabricError, SseDecoder, toAskEvent } from '@ragfabric/sdk';
import { Observable, catchError, throwError } from 'rxjs';

/** Parse a text error body (a stream's error arrives as text) so its `detail` survives. */
function parsed(body: unknown): unknown {
  if (typeof body !== 'string') {
    return body;
  }
  try {
    return JSON.parse(body);
  } catch {
    return body;
  }
}

export function toRagFabricError(error: unknown): unknown {
  if (error instanceof HttpErrorResponse) {
    return new RagFabricError(error.status, parsed(error.error), error.statusText);
  }
  return error;
}

/**
 * Executes @ragfabric/sdk requests through Angular's HttpClient (decision D5).
 *
 * The SDK owns every path, body and response type; this service owns only how
 * a request travels. Going through HttpClient keeps the auth interceptor (the
 * bearer token, the 401 redirect) in one place for every call, streams
 * included, and lets specs drive the app with HttpTestingController.
 *
 * This is the only file in the app allowed to import HttpClient
 * (`npm run check:sdk-only`).
 */
@Injectable({ providedIn: 'root' })
export class SdkHttp {
  private readonly http = inject(HttpClient);

  send<T>(request: ApiRequest<T>): Observable<T> {
    return this.http
      .request(request.method, request.path, {
        body: request.body,
        headers: request.headers ? { ...request.headers } : undefined,
        responseType: request.responseType as 'json',
      })
      .pipe(catchError((error: unknown) => throwError(() => toRagFabricError(error)))) as Observable<T>;
  }

  /**
   * Execute a streaming request (POST /api/ask with `stream: true`).
   *
   * EventSource cannot POST or send a header (decision D6), so the response is
   * read as text with download progress on: each progress event carries the
   * text so far, and the new part is fed to the SDK's decoder, which buffers
   * across chunk boundaries. Unsubscribing aborts the request.
   */
  stream(request: ApiRequest<unknown>): Observable<AskEvent> {
    return new Observable<AskEvent>((subscriber) => {
      const decoder = new SseDecoder();
      let seen = 0;
      const feed = (text: string): void => {
        if (text.length <= seen) {
          return;
        }
        const fresh = text.slice(seen);
        seen = text.length;
        for (const message of decoder.push(fresh)) {
          subscriber.next(toAskEvent(message));
        }
      };
      const subscription = this.http
        .request(request.method, request.path, {
          body: request.body,
          headers: request.headers ? { ...request.headers } : undefined,
          observe: 'events',
          reportProgress: true,
          responseType: 'text',
        })
        .subscribe({
          next: (event) => {
            if (event.type === HttpEventType.DownloadProgress) {
              feed((event as HttpDownloadProgressEvent).partialText ?? '');
            } else if (event.type === HttpEventType.Response) {
              feed(event.body ?? '');
              for (const message of decoder.end()) {
                subscriber.next(toAskEvent(message));
              }
              subscriber.complete();
            }
          },
          error: (error: unknown) => subscriber.error(toRagFabricError(error)),
        });
      return () => subscription.unsubscribe();
    });
  }
}
