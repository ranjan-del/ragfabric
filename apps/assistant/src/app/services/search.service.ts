import { Injectable, inject } from '@angular/core';
import { requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';
import { AnswerResponse, SearchResults } from '../models';

export interface QueryOptions {
  top_k?: number;
  collection_id?: number | null;
  mode?: 'semantic' | 'hybrid';
}

@Injectable({ providedIn: 'root' })
export class SearchService {
  private readonly sdk = inject(SdkHttp);

  ask(query: string, opts: QueryOptions = {}): Observable<AnswerResponse> {
    return this.sdk.send(
      requests.search.query(query, {
        top_k: opts.top_k ?? 5,
        collection_id: opts.collection_id ?? null,
        mode: opts.mode ?? 'semantic',
      }),
    );
  }

  search(
    query: string,
    mode: 'semantic' | 'hybrid',
    opts: QueryOptions = {},
  ): Observable<SearchResults> {
    const options = { top_k: opts.top_k ?? 5, collection_id: opts.collection_id ?? null };
    return this.sdk.send(
      mode === 'hybrid' ? requests.search.hybrid(query, options) : requests.search.semantic(query, options),
    );
  }
}
