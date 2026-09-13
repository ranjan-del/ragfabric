import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiService } from './api.service';
import { AnswerResponse, SearchResults } from '../models';

export interface QueryOptions {
  top_k?: number;
  collection_id?: number | null;
  mode?: 'semantic' | 'hybrid';
}

@Injectable({ providedIn: 'root' })
export class SearchService {
  constructor(private http: HttpClient, private api: ApiService) {}

  ask(query: string, opts: QueryOptions = {}): Observable<AnswerResponse> {
    return this.http.post<AnswerResponse>(`${this.api.baseUrl}/search/query`, {
      query,
      top_k: opts.top_k ?? 5,
      collection_id: opts.collection_id ?? null,
      mode: opts.mode ?? 'semantic',
    });
  }

  search(
    query: string,
    mode: 'semantic' | 'hybrid',
    opts: QueryOptions = {}
  ): Observable<SearchResults> {
    return this.http.post<SearchResults>(`${this.api.baseUrl}/search/${mode}`, {
      query,
      mode,
      top_k: opts.top_k ?? 5,
      collection_id: opts.collection_id ?? null,
    });
  }
}
