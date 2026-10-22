import { ChangeDetectionStrategy, Component, computed, effect, inject, input, output, signal, untracked } from '@angular/core';
import { ChunkContext, Citation } from '@ragfabric/sdk';

import { ChunkRefService } from '../services/chunk-ref.service';
import { describeError } from '../services/http-error';
import { ModalComponent } from '../ui';
import { percent } from './format';
import { WalkedEdge, relationLabel } from './graph';

/** One chunk that backs an edge: from the answer's own passages when it is there, else read from the store. */
interface Backing {
  chunkId: number;
  /** The `[n]` marker when the answer was written from this passage. */
  marker: string | null;
  filename: string | null;
  page: number | null;
  text: string | null;
  state: 'ready' | 'loading' | 'failed';
  error: string | null;
}

/**
 * The edge behind an `[E k]` marker: the relationship as walked, how sure the
 * extraction model was, and the chunk or chunks that back it (ADR 0012: a
 * relationship claim cites its edge and a chunk).
 */
@Component({
  selector: 'app-edge-viewer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [ModalComponent],
  template: `
    <ui-modal [open]="edge() !== null" [title]="title()" (closed)="closed.emit()">
      @if (edge(); as e) {
        <div data-test="edge-viewer">
          <p class="reading" data-test="edge-reading">
            <strong>{{ e.from }}</strong>
            <code>{{ label(e) }}</code>
            <strong>{{ e.to }}</strong>
          </p>
          <dl class="facts">
            <div><dt>Relation</dt><dd>{{ relationText(e) }}</dd></div>
            <div><dt>Extraction confidence</dt><dd data-test="edge-confidence">{{ confidence(e) }}</dd></div>
          </dl>
          <h3>Backed by</h3>
          @if (backing().length === 0) {
            <p class="muted" data-test="no-backing">
              No source passage that you may read backs this edge, so nothing can be shown for it.
            </p>
          }
          @for (b of backing(); track b.chunkId) {
            <section class="backing" data-test="backing">
              <p class="where">
                <span data-test="backing-file">{{ b.filename ?? 'unknown source' }}</span>@if (b.page != null) {, page {{ b.page }}}
                <span class="muted">
                  @if (b.marker) { (cited in the answer as {{ b.marker }}) } @else { (not one of the passages the answer was written from) }
                </span>
              </p>
              @if (b.state === 'loading') {
                <p class="muted">Loading the passage...</p>
              } @else if (b.state === 'failed') {
                <p class="alert alert-error" role="alert">{{ b.error }}</p>
              } @else {
                <blockquote class="snippet" data-test="backing-text">{{ b.text }}</blockquote>
              }
            </section>
          }
        </div>
      }
    </ui-modal>
  `,
  styles: `
    .reading { display: flex; flex-wrap: wrap; gap: var(--space-2); align-items: baseline; font-size: 1.05rem; margin: 0 0 var(--space-3); }
    .facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-2); margin: 0 0 var(--space-3); }
    .facts dt { font-size: 0.75rem; color: var(--text-muted); }
    .facts dd { margin: 0; overflow-wrap: anywhere; }
    h3 { font-size: 0.95rem; margin: 0 0 var(--space-2); }
    .backing { margin-bottom: var(--space-3); }
    .where { margin: 0 0 var(--space-1); font-size: 0.85rem; overflow-wrap: anywhere; }
    .snippet { margin: 0; padding: var(--space-3); background: var(--surface-2); border-radius: var(--radius-sm);
      white-space: pre-wrap; line-height: 1.6; overflow-wrap: anywhere; }
  `,
})
export class EdgeViewerComponent {
  private readonly chunks = inject(ChunkRefService);

  readonly edge = input<WalkedEdge | null>(null);
  readonly citations = input<readonly Citation[]>([]);
  readonly closed = output<void>();

  private readonly loaded = signal<Record<number, ChunkContext | 'failed' | string>>({});

  readonly title = computed(() => {
    const e = this.edge();
    return e ? `Relationship [E ${e.k}]` : '';
  });

  readonly backing = computed<Backing[]>(() => {
    const e = this.edge();
    if (e === null) {
      return [];
    }
    const loaded = this.loaded();
    return e.edge.source_chunk_ids.map((chunkId): Backing => {
      const citation = this.citations().find((c) => c.chunk_id === chunkId);
      if (citation) {
        return {
          chunkId,
          marker: citation.marker,
          filename: citation.filename ?? null,
          page: citation.page ?? null,
          text: citation.snippet,
          state: 'ready',
          error: null,
        };
      }
      const got = loaded[chunkId];
      if (got === undefined) {
        return { chunkId, marker: null, filename: null, page: null, text: null, state: 'loading', error: null };
      }
      if (typeof got === 'string') {
        return { chunkId, marker: null, filename: null, page: null, text: null, state: 'failed', error: got };
      }
      return {
        chunkId,
        marker: null,
        filename: got.filename,
        page: got.chunk.page ?? null,
        text: got.chunk.text,
        state: 'ready',
        error: null,
      };
    });
  });

  constructor() {
    // Read the backing chunks that no citation carries, once each.
    effect(() => {
      const e = this.edge();
      const citations = this.citations();
      if (e === null) {
        return;
      }
      for (const id of e.edge.source_chunk_ids) {
        if (citations.some((c) => c.chunk_id === id) || untracked(this.loaded)[id] !== undefined) {
          continue;
        }
        this.chunks.get(id).subscribe({
          next: (ctx) => this.loaded.update((m) => ({ ...m, [id]: ctx })),
          error: (err: unknown) =>
            this.loaded.update((m) => ({ ...m, [id]: describeError(err, 'The passage could not be loaded.') })),
        });
      }
    });
  }

  label(e: WalkedEdge): string {
    return relationLabel(e.relation);
  }

  relationText(e: WalkedEdge): string {
    const stored = e.edge.relation_type;
    return e.edge.reversed
      ? `${relationLabel(e.relation)} (walked against the stored ${relationLabel(stored)})`
      : relationLabel(e.relation);
  }

  confidence(e: WalkedEdge): string {
    return e.edge.confidence == null ? 'not measured' : percent(e.edge.confidence);
  }
}
