import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiService } from './api.service';
import { DocumentItem, DocumentList } from '../models';

@Injectable({ providedIn: 'root' })
export class DocumentService {
  constructor(private http: HttpClient, private api: ApiService) {}

  list(collectionId?: number): Observable<DocumentList> {
    const query = collectionId != null ? `?collection_id=${collectionId}` : '';
    return this.http.get<DocumentList>(`${this.api.baseUrl}/documents${query}`);
  }

  upload(file: File, collectionId?: number): Observable<DocumentItem> {
    const form = new FormData();
    form.append('file', file);
    if (collectionId != null) {
      form.append('collection_id', String(collectionId));
    }
    return this.http.post<DocumentItem>(
      `${this.api.baseUrl}/documents/upload`,
      form
    );
  }

  delete(id: number): Observable<unknown> {
    return this.http.delete(`${this.api.baseUrl}/documents/${id}`);
  }
}
