import { Injectable } from '@angular/core';

/**
 * Shared API base configuration. In production the built app is served behind a
 * reverse proxy (nginx / Firebase Hosting rewrites) that forwards `/api` to the
 * FastAPI backend, so a relative base works in every environment.
 */
@Injectable({ providedIn: 'root' })
export class ApiService {
  readonly baseUrl = '/api';
}
