import { Injectable, inject } from '@angular/core';
import { requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';
import { DocumentItem, DocumentList } from '../models';

@Injectable({ providedIn: 'root' })
export class DocumentService {
  private readonly sdk = inject(SdkHttp);

  list(collectionId?: number): Observable<DocumentList> {
    return this.sdk.send(requests.documents.list({ collection_id: collectionId }));
  }

  upload(file: File, collectionId?: number): Observable<DocumentItem> {
    return this.sdk.send(requests.documents.upload(file, { collection_id: collectionId }));
  }

  /** The original file, for the source viewer's download. */
  download(id: number): Observable<Blob> {
    return this.sdk.send(requests.documents.download(id));
  }

  delete(id: number): Observable<unknown> {
    return this.sdk.send(requests.documents.delete(id));
  }
}
