import { Injectable, inject } from '@angular/core';
import { ChunkContext } from '@ragfabric/sdk';
import { Observable, shareReplay } from 'rxjs';

import { DocumentService } from './document.service';

/**
 * The stored chunk behind a graph edge, for chunks that were not among the
 * passages an answer was written from (so no citation carries their text).
 * Cached per chunk for the life of the tab; a failed read is not cached.
 */
@Injectable({ providedIn: 'root' })
export class ChunkRefService {
  private readonly documents = inject(DocumentService);
  private readonly cache = new Map<number, Observable<ChunkContext>>();

  get(chunkId: number): Observable<ChunkContext> {
    let found = this.cache.get(chunkId);
    if (found === undefined) {
      found = this.documents.chunkContext(chunkId, 0).pipe(shareReplay({ bufferSize: 1, refCount: false }));
      this.cache.set(chunkId, found);
      found.subscribe({ error: () => this.cache.delete(chunkId) });
    }
    return found;
  }
}
