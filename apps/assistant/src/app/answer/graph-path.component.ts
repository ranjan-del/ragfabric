import { ChangeDetectionStrategy, Component, computed, effect, inject, input, output, signal } from '@angular/core';
import { Citation, Subgraph } from '@ragfabric/sdk';

import { ChunkRefService } from '../services/chunk-ref.service';
import { percent } from './format';
import { WalkedEdge, emptyGraphReason, relationLabel, walkedEdges } from './graph';

/**
 * The path the graph strategy walked, collapsed by default: the entities it
 * reached, then each relationship in the order the answer's `[E k]` markers
 * number them, with its relation, how sure extraction was and where it was
 * read. When the walk found nothing, the API's reason is shown in plain words.
 */
@Component({
  selector: 'app-graph-path',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (reason(); as why) {
      <p class="empty" data-test="graph-empty" role="status">
        <strong>Graph found nothing.</strong> {{ why }}
      </p>
    } @else if (graph(); as g) {
      <details class="path" data-test="graph-path" (toggle)="onToggle($event)">
        <summary>
          Graph path: {{ g.nodes.length }} {{ g.nodes.length === 1 ? 'entity' : 'entities' }},
          {{ edges().length }} {{ edges().length === 1 ? 'relationship' : 'relationships' }} walked
        </summary>
        @if (g.truncated) {
          <p class="muted note">The walk was cut by the node budget, so some entities are not shown.</p>
        }
        <ul class="entities" aria-label="Entities reached">
          @for (n of g.nodes; track n.id) {
            <li data-test="entity">{{ n.name }} <span class="muted">{{ n.entity_type }}, {{ n.depth === 0 ? 'matched' : n.depth + (n.depth === 1 ? ' hop' : ' hops') }}</span></li>
          }
        </ul>
        <ol class="edges" aria-label="Relationships walked">
          @for (e of edges(); track e.k) {
            <li data-test="edge">
              <button type="button" class="marker" data-test="edge-chip" [attr.aria-label]="'Open relationship E ' + e.k" (click)="openEdge.emit(e)">[E {{ e.k }}]</button>
              <span class="reading">
                {{ e.from }} <code>{{ label(e) }}</code> {{ e.to }}
              </span>
              <span class="muted detail">
                confidence {{ conf(e) }}, source <span data-test="edge-source">{{ source(e) }}</span>
              </span>
            </li>
          }
        </ol>
      </details>
    }
  `,
  styles: `
    :host { display: block; }
    .path { font-size: 0.9rem; margin-top: var(--space-3); }
    .path summary { cursor: pointer; color: var(--text-muted); }
    .path summary:focus-visible, .marker:focus-visible { outline: none; box-shadow: var(--focus-ring); }
    .empty { margin: var(--space-3) 0 0; font-size: 0.9rem; padding: var(--space-2) var(--space-3); background: var(--surface-2); border-radius: var(--radius-sm); }
    .note { margin: var(--space-2) 0 0; }
    .entities { list-style: none; display: flex; flex-wrap: wrap; gap: var(--space-2); padding: 0; margin: var(--space-3) 0; }
    .entities li { border: 1px solid var(--border); border-radius: 999px; padding: 0.1rem 0.6rem; background: var(--surface-2); }
    .edges { list-style: none; padding: 0; margin: 0; display: grid; gap: var(--space-2); }
    .edges li { display: grid; grid-template-columns: auto minmax(0, 1fr); column-gap: var(--space-2); align-items: baseline; }
    .reading { overflow-wrap: anywhere; }
    .detail { grid-column: 2; font-size: 0.8rem; overflow-wrap: anywhere; }
    .marker { border: 0; background: var(--brand-soft); color: var(--brand-700); border-radius: 6px;
      padding: 0 0.3rem; font: inherit; font-size: 0.85em; cursor: pointer; white-space: nowrap; }
    code { font-size: 0.8em; }
  `,
})
export class GraphPathComponent {
  private readonly chunks = inject(ChunkRefService);

  readonly graph = input<Subgraph | null>(null);
  readonly citations = input<readonly Citation[]>([]);
  readonly openEdge = output<WalkedEdge>();

  readonly edges = computed(() => walkedEdges(this.graph()));
  readonly reason = computed(() => {
    const g = this.graph();
    if (g === null) {
      return null;
    }
    return emptyGraphReason(g) ?? (g.edges.length === 0 ? 'The walk kept no relationships.' : null);
  });

  private readonly opened = signal(false);
  private readonly files = signal<Record<number, string>>({});

  constructor() {
    // The file name of an edge's first source chunk, read only once the path is opened.
    effect(() => {
      if (!this.opened()) {
        return;
      }
      for (const e of this.edges()) {
        const id = e.edge.source_chunk_ids[0];
        if (id === undefined || this.citations().some((c) => c.chunk_id === id)) {
          continue;
        }
        this.chunks.get(id).subscribe({
          next: (ctx) =>
            this.files.update((m) => ({
              ...m,
              [id]: ctx.filename + (ctx.chunk.page == null ? '' : ` p${ctx.chunk.page}`),
            })),
          error: () => this.files.update((m) => ({ ...m, [id]: `chunk ${id}` })),
        });
      }
    });
  }

  onToggle(event: Event): void {
    this.opened.set((event.target as HTMLDetailsElement).open);
  }

  label(e: WalkedEdge): string {
    return relationLabel(e.relation);
  }

  conf(e: WalkedEdge): string {
    return e.edge.confidence == null ? 'not measured' : percent(e.edge.confidence);
  }

  /** A short source reference: the file (and page) of the first chunk that backs the edge. */
  source(e: WalkedEdge): string {
    const id = e.edge.source_chunk_ids[0];
    if (id === undefined) {
      return 'none you may read';
    }
    const citation = this.citations().find((c) => c.chunk_id === id);
    if (citation) {
      return (citation.filename ?? `chunk ${id}`) + (citation.page == null ? '' : ` p${citation.page}`);
    }
    return this.files()[id] ?? (this.opened() ? 'loading' : `chunk ${id}`);
  }
}
