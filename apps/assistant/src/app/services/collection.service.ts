import { Injectable, inject } from '@angular/core';
import { requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';
import { Collection, CollectionDetail } from '../models';

@Injectable({ providedIn: 'root' })
export class CollectionService {
  private readonly sdk = inject(SdkHttp);

  list(): Observable<Collection[]> {
    return this.sdk.send(requests.collections.list());
  }

  create(name: string, description: string): Observable<Collection> {
    return this.sdk.send(requests.collections.create({ name, description }));
  }

  get(id: number): Observable<CollectionDetail> {
    return this.sdk.send(requests.collections.get(id));
  }

  update(id: number, changes: { name?: string; description?: string }): Observable<Collection> {
    return this.sdk.send(requests.collections.update(id, changes));
  }

  delete(id: number): Observable<unknown> {
    return this.sdk.send(requests.collections.delete(id));
  }
}
