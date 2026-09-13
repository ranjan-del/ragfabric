import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiService } from './api.service';
import { Collection, CollectionDetail } from '../models';

@Injectable({ providedIn: 'root' })
export class CollectionService {
  constructor(private http: HttpClient, private api: ApiService) {}

  list(): Observable<Collection[]> {
    return this.http.get<Collection[]>(`${this.api.baseUrl}/collections`);
  }

  create(name: string, description: string): Observable<Collection> {
    return this.http.post<Collection>(`${this.api.baseUrl}/collections`, {
      name,
      description,
    });
  }

  get(id: number): Observable<CollectionDetail> {
    return this.http.get<CollectionDetail>(`${this.api.baseUrl}/collections/${id}`);
  }

  delete(id: number): Observable<unknown> {
    return this.http.delete(`${this.api.baseUrl}/collections/${id}`);
  }
}
