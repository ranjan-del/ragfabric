import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { ApiService } from './api.service';
import { ProviderConfig, ProviderConfigWritten, ProviderTestResult } from '../models';

export interface ProviderUpdate {
  llm?: { provider: string; model: string | null; base_url: string | null };
  embeddings?: {
    provider: string;
    model: string | null;
    dim: number | null;
    base_url: string | null;
  };
}

/**
 * Provider selection.
 *
 * Note what this service cannot do: send or receive an API key. The server
 * reads secrets from its own environment and reports only whether they are
 * present, so there is no shape here that could carry one.
 */
@Injectable({ providedIn: 'root' })
export class ProvidersService {
  private readonly http = inject(HttpClient);
  private readonly api = inject(ApiService);

  read(): Observable<ProviderConfig> {
    return this.http.get<ProviderConfig>(`${this.api.baseUrl}/admin/providers`);
  }

  write(update: ProviderUpdate): Observable<ProviderConfigWritten> {
    return this.http.put<ProviderConfigWritten>(`${this.api.baseUrl}/admin/providers`, update);
  }

  test(target: 'llm' | 'embeddings'): Observable<ProviderTestResult> {
    return this.http.post<ProviderTestResult>(`${this.api.baseUrl}/admin/providers/test`, {
      target,
    });
  }
}
