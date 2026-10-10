// Every call the API offers, described as data (decision D5).
//
// A builder returns an ApiRequest: method, path (with its query string), body
// and the kind of response. It sends nothing. RagFabricClient executes requests
// with fetch; the Angular app executes the same requests through HttpClient so
// its interceptor and its tests keep working. Either way this file is the only
// place in the repository's TypeScript that spells an API path, and the
// response types come from the generated `paths`, so a server change that is
// regenerated but not followed here fails to compile.

import type { EvalDashboard, EvalRun, EvalRunDetail } from './evaluation.js';
import type { paths } from './generated/schema.js';
import type {
  AdminUserCreate,
  AnalyticsOverview,
  ApiKeyCreate,
  AskRequest,
  CollectionCreate,
  CollectionUpdate,
  GroupUpdate,
  OverrideCreate,
  PermissionUpdate,
  ProviderConfigUpdate,
  SearchRequest,
  UsageStats,
} from './types.js';

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';

/**
 * One API call, not yet sent. `T` is the parsed response type.
 *
 * `body` is a plain object (sent as JSON), a FormData (multipart) or a string
 * (sent as is, with `headers` naming its type). `responseType` is `blob` only
 * for a file download.
 */
export interface ApiRequest<T> {
  readonly method: HttpMethod;
  readonly path: string;
  readonly body?: unknown;
  readonly headers?: Readonly<Record<string, string>>;
  readonly responseType: 'json' | 'blob';
  /** Never set; carries `T` for the type checker. */
  readonly __response?: T;
}

type Response<P extends keyof paths, M extends keyof paths[P]> = paths[P][M] extends {
  response: infer R;
}
  ? R
  : never;

type Query = Record<string, string | number | boolean | null | undefined>;

function withQuery(path: string, query: Query = {}): string {
  const parts = Object.entries(query)
    .filter(([, value]) => value !== undefined && value !== null)
    .map(([name, value]) => `${encodeURIComponent(name)}=${encodeURIComponent(String(value))}`);
  return parts.length === 0 ? path : `${path}?${parts.join('&')}`;
}

function get<T>(path: string, query?: Query): ApiRequest<T> {
  return { method: 'GET', path: withQuery(path, query), responseType: 'json' };
}

function send<T>(method: HttpMethod, path: string, body?: unknown): ApiRequest<T> {
  return body === undefined
    ? { method, path, responseType: 'json' }
    : { method, path, body, responseType: 'json' };
}

/** Drop keys whose value is undefined, so an unset option is absent rather than null. */
function defined<T extends object>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as T;
}

export type AskOptions = Omit<AskRequest, 'query' | 'stream'>;
export type SearchOptions = Omit<SearchRequest, 'query' | 'mode'>;

export interface UploadOptions {
  collection_id?: number | null;
  chunk_size?: number | null;
  chunk_overlap?: number | null;
  /** The file name to send; defaults to the File's own name. */
  filename?: string;
}

export const requests = {
  auth: {
    /** OAuth2 password form, as the server expects: urlencoded, `username` is the email. */
    login(email: string, password: string): ApiRequest<Response<'/api/auth/login', 'post'>> {
      const form = new URLSearchParams();
      form.set('username', email);
      form.set('password', password);
      return {
        method: 'POST',
        path: '/api/auth/login',
        body: form.toString(),
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        responseType: 'json',
      };
    },
    register(email: string, password: string): ApiRequest<Response<'/api/auth/register', 'post'>> {
      return send('POST', '/api/auth/register', { email, password });
    },
    me: (): ApiRequest<Response<'/api/auth/me', 'get'>> => get('/api/auth/me'),
    logout: (): ApiRequest<unknown> => send('POST', '/api/auth/logout'),
  },

  /**
   * POST /api/ask. With `stream` false the response is the finished answer;
   * use `askStream` for the event stream. `strategy` left unset is not sent,
   * so the server's `router.mode` decides.
   */
  ask(query: string, options: AskOptions = {}): ApiRequest<Response<'/api/ask', 'post'>> {
    return send('POST', '/api/ask', defined({ ...options, query, stream: false }));
  },
  /** POST /api/ask as a server sent events stream; execute with a transport's stream method. */
  askStream(query: string, options: AskOptions = {}): ApiRequest<never> {
    return send('POST', '/api/ask', defined({ ...options, query, stream: true }));
  },

  search: {
    query: (query: string, options: SearchOptions = {}): ApiRequest<Response<'/api/search/query', 'post'>> =>
      send('POST', '/api/search/query', defined({ ...options, query })),
    semantic: (query: string, options: SearchOptions = {}): ApiRequest<Response<'/api/search/semantic', 'post'>> =>
      send('POST', '/api/search/semantic', defined({ ...options, query, mode: 'semantic' })),
    hybrid: (query: string, options: SearchOptions = {}): ApiRequest<Response<'/api/search/hybrid', 'post'>> =>
      send('POST', '/api/search/hybrid', defined({ ...options, query, mode: 'hybrid' })),
  },

  runs: {
    get: (id: number): ApiRequest<Response<'/api/runs/{run_id}', 'get'>> => get(`/api/runs/${id}`),
    ingestion: (id: number): ApiRequest<Response<'/api/runs/ingestion/{run_id}', 'get'>> =>
      get(`/api/runs/ingestion/${id}`),
  },

  documents: {
    list: (filter: { collection_id?: number | null; format?: string | null } = {}): ApiRequest<
      Response<'/api/documents', 'get'>
    > => get('/api/documents', filter),
    get: (id: number): ApiRequest<Response<'/api/documents/{document_id}', 'get'>> =>
      get(`/api/documents/${id}`),
    upload(file: Blob, options: UploadOptions = {}): ApiRequest<Response<'/api/documents/upload', 'post'>> {
      const form = new FormData();
      const name = options.filename ?? (file as Blob & { name?: string }).name;
      if (name === undefined) {
        form.append('file', file);
      } else {
        form.append('file', file, name);
      }
      for (const field of ['collection_id', 'chunk_size', 'chunk_overlap'] as const) {
        const value = options[field];
        if (value !== undefined && value !== null) {
          form.append(field, String(value));
        }
      }
      return send('POST', '/api/documents/upload', form);
    },
    /** The original file, as a Blob. */
    download: (id: number): ApiRequest<Blob> => ({
      method: 'GET',
      path: `/api/documents/${id}/download`,
      responseType: 'blob',
    }),
    move: (id: number, collectionId: number | null): ApiRequest<Response<'/api/documents/{document_id}/move', 'post'>> =>
      send('POST', `/api/documents/${id}/move`, { collection_id: collectionId }),
    delete: (id: number): ApiRequest<unknown> => send('DELETE', `/api/documents/${id}`),
  },

  collections: {
    list: (): ApiRequest<Response<'/api/collections', 'get'>> => get('/api/collections'),
    create: (body: CollectionCreate): ApiRequest<Response<'/api/collections', 'post'>> =>
      send('POST', '/api/collections', body),
    get: (id: number): ApiRequest<Response<'/api/collections/{collection_id}', 'get'>> =>
      get(`/api/collections/${id}`),
    update: (id: number, changes: CollectionUpdate): ApiRequest<Response<'/api/collections/{collection_id}', 'put'>> =>
      send('PUT', `/api/collections/${id}`, changes),
    delete: (id: number): ApiRequest<unknown> => send('DELETE', `/api/collections/${id}`),
  },

  analytics: {
    overview: (): ApiRequest<AnalyticsOverview> => get('/api/analytics/overview'),
    usage: (): ApiRequest<UsageStats> => get('/api/analytics/usage'),
  },

  admin: {
    users: {
      list: (): ApiRequest<Response<'/api/admin/users', 'get'>> => get('/api/admin/users'),
      create: (body: AdminUserCreate): ApiRequest<Response<'/api/admin/users', 'post'>> =>
        send('POST', '/api/admin/users', body),
      delete: (id: number): ApiRequest<unknown> => send('DELETE', `/api/admin/users/${id}`),
      setPermissions: (
        id: number,
        changes: PermissionUpdate,
      ): ApiRequest<Response<'/api/admin/users/{user_id}/permissions', 'put'>> =>
        send('PUT', `/api/admin/users/${id}/permissions`, changes),
    },
    groups: {
      list: (): ApiRequest<Response<'/api/admin/groups', 'get'>> => get('/api/admin/groups'),
      create: (name: string, description: string): ApiRequest<Response<'/api/admin/groups', 'post'>> =>
        send('POST', '/api/admin/groups', { name, description }),
      update: (id: number, changes: GroupUpdate): ApiRequest<Response<'/api/admin/groups/{group_id}', 'put'>> =>
        send('PUT', `/api/admin/groups/${id}`, changes),
      delete: (id: number): ApiRequest<unknown> => send('DELETE', `/api/admin/groups/${id}`),
      members: (id: number): ApiRequest<Response<'/api/admin/groups/{group_id}/members', 'get'>> =>
        get(`/api/admin/groups/${id}/members`),
      addMember: (id: number, userId: number): ApiRequest<unknown> =>
        send('POST', `/api/admin/groups/${id}/members`, { user_id: userId }),
      removeMember: (id: number, userId: number): ApiRequest<unknown> =>
        send('DELETE', `/api/admin/groups/${id}/members/${userId}`),
    },
    grants: {
      list: (): ApiRequest<Response<'/api/admin/grants', 'get'>> => get('/api/admin/grants'),
      create: (
        groupId: number,
        collectionId: number,
        permission: 'read' | 'write',
      ): ApiRequest<Response<'/api/admin/grants', 'post'>> =>
        send('POST', '/api/admin/grants', {
          group_id: groupId,
          collection_id: collectionId,
          permission,
        }),
      delete: (id: number): ApiRequest<unknown> => send('DELETE', `/api/admin/grants/${id}`),
    },
    overrides: {
      list: (): ApiRequest<Response<'/api/admin/overrides', 'get'>> => get('/api/admin/overrides'),
      create: (body: OverrideCreate): ApiRequest<Response<'/api/admin/overrides', 'post'>> =>
        send('POST', '/api/admin/overrides', body),
    },
    keys: {
      list: (): ApiRequest<Response<'/api/admin/keys', 'get'>> => get('/api/admin/keys'),
      /** The plaintext key is in this response and in no other, ever. */
      create: (body: ApiKeyCreate): ApiRequest<Response<'/api/admin/keys', 'post'>> =>
        send('POST', '/api/admin/keys', body),
      revoke: (id: number): ApiRequest<unknown> => send('DELETE', `/api/admin/keys/${id}`),
    },
    providers: {
      read: (): ApiRequest<Response<'/api/admin/providers', 'get'>> => get('/api/admin/providers'),
      write: (update: ProviderConfigUpdate): ApiRequest<Response<'/api/admin/providers', 'put'>> =>
        send('PUT', '/api/admin/providers', update),
      test: (target: 'llm' | 'embeddings'): ApiRequest<Response<'/api/admin/providers/test', 'post'>> =>
        send('POST', '/api/admin/providers/test', { target }),
    },
  },

  /** Phase 8's read only evaluation API. Provisional: see evaluation.ts. */
  evaluation: {
    runs: (limit?: number): ApiRequest<EvalRun[]> => get('/api/eval/runs', { limit }),
    run: (id: number): ApiRequest<EvalRunDetail> => get(`/api/eval/runs/${id}`),
    dashboard: (days?: number): ApiRequest<EvalDashboard> => get('/api/eval/dashboard', { days }),
  },
} as const;
