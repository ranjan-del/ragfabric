import { Injectable, inject } from '@angular/core';
import { ProviderConfigUpdate, requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';
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
  private readonly sdk = inject(SdkHttp);

  read(): Observable<ProviderConfig> {
    return this.sdk.send(requests.admin.providers.read());
  }

  write(update: ProviderUpdate): Observable<ProviderConfigWritten> {
    // The pickers offer only the provider names the server accepts, and the
    // server validates the body (a 422 names the field), so the looser string
    // type the form works in is narrowed here rather than in every signal.
    return this.sdk.send(requests.admin.providers.write(update as ProviderConfigUpdate));
  }

  test(target: 'llm' | 'embeddings'): Observable<ProviderTestResult> {
    return this.sdk.send(requests.admin.providers.test(target));
  }
}
