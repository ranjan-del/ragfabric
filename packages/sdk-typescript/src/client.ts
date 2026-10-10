// A fetch based client for scripts, Node and front ends other than the
// bundled Angular app (which executes the same requests through HttpClient).

import { RagFabricError } from './errors.js';
import type { EvalDashboard, EvalRun, EvalRunDetail } from './evaluation.js';
import type { ApiRequest, AskOptions, UploadOptions } from './requests.js';
import { requests } from './requests.js';
import { SseDecoder, toAskEvent } from './sse.js';
import type { Answer, AskEvent, DocumentItem, Run, User } from './types.js';

export interface ClientOptions {
  /** e.g. "http://localhost:8000". Empty (the default) means same origin. */
  baseUrl?: string;
  /** A JWT from POST /api/auth/login, sent as `Authorization: Bearer`. */
  token?: string;
  /** An `rf_` API key, sent as `X-API-Key`. When both are given the key wins, as in the Python SDK. */
  apiKey?: string;
  /** A fetch implementation; defaults to the global one. Tests pass a fake. */
  fetch?: typeof fetch;
}

export interface CallOptions {
  signal?: AbortSignal;
}

async function errorBody(response: Response): Promise<unknown> {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export class RagFabricClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;
  private token: string | undefined;
  private readonly apiKey: string | undefined;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? '').replace(/\/+$/, '');
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.token = options.token;
    this.apiKey = options.apiKey;
  }

  private headers(request: ApiRequest<unknown>): Record<string, string> {
    const headers: Record<string, string> = { ...request.headers };
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    } else if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    const body = request.body;
    if (body !== undefined && typeof body !== 'string' && !(body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }
    return headers;
  }

  private async raw(request: ApiRequest<unknown>, options: CallOptions): Promise<Response> {
    const body = request.body;
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${request.path}`, {
        method: request.method,
        headers: this.headers(request),
        body:
          body === undefined
            ? undefined
            : typeof body === 'string' || body instanceof FormData
              ? body
              : JSON.stringify(body),
        signal: options.signal,
      });
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw error;
      }
      throw new RagFabricError(0, null);
    }
    if (!response.ok) {
      throw new RagFabricError(response.status, await errorBody(response), response.statusText);
    }
    return response;
  }

  /** Execute any request from `requests` and return its parsed response. */
  async send<T>(request: ApiRequest<T>, options: CallOptions = {}): Promise<T> {
    const response = await this.raw(request, options);
    if (request.responseType === 'blob') {
      return (await response.blob()) as T;
    }
    const text = await response.text();
    return (text === '' ? null : JSON.parse(text)) as T;
  }

  /** Execute a streaming request and yield each event as it arrives. */
  async *stream(request: ApiRequest<unknown>, options: CallOptions = {}): AsyncGenerator<AskEvent> {
    const response = await this.raw(request, options);
    if (response.body === null) {
      return;
    }
    const decoder = new SseDecoder();
    const reader = response.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        for (const message of decoder.push(value)) {
          yield toAskEvent(message);
        }
      }
      for (const message of decoder.end()) {
        yield toAskEvent(message);
      }
    } finally {
      reader.releaseLock();
    }
  }

  /** Sign in and keep the token for later calls. */
  async login(email: string, password: string): Promise<User> {
    const token = await this.send(requests.auth.login(email, password));
    this.token = token.access_token;
    return this.send(requests.auth.me());
  }

  /** The finished, cited answer (`stream: false`). */
  ask(query: string, options: AskOptions = {}, call: CallOptions = {}): Promise<Answer> {
    return this.send(requests.ask(query, options), call);
  }

  /**
   * The answer as it streams: `retrieval`, `token`..., optionally
   * `superseded` (replace what was drawn with its text), `citations`, `done`.
   */
  askStream(query: string, options: AskOptions = {}, call: CallOptions = {}): AsyncGenerator<AskEvent> {
    return this.stream(requests.askStream(query, options), call);
  }

  run(id: number): Promise<Run> {
    return this.send(requests.runs.get(id));
  }

  documents(): Promise<DocumentItem[]> {
    return this.send(requests.documents.list()).then((list) => list.items);
  }

  upload(file: Blob, options: UploadOptions = {}): Promise<DocumentItem> {
    return this.send(requests.documents.upload(file, options));
  }

  evaluationRuns(limit?: number): Promise<EvalRun[]> {
    return this.send(requests.evaluation.runs(limit));
  }

  evaluationRun(id: number): Promise<EvalRunDetail> {
    return this.send(requests.evaluation.run(id));
  }

  evaluationDashboard(days?: number): Promise<EvalDashboard> {
    return this.send(requests.evaluation.dashboard(days));
  }
}
