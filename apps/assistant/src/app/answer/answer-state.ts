import { computed, signal } from '@angular/core';
import {
  AskEvent,
  Citation,
  DoneEvent,
  RetrievalEvent,
  SupersededEvent,
} from '@ragfabric/sdk';

const GRAPH_REASONS: Record<string, string> = {
  no_graph_coverage: 'No knowledge graph covers these documents yet.',
  no_entity_matched: 'No entity in the question matched the knowledge graph.',
  no_walkable_edges: 'The matched entities have no relationships to follow.',
};

export type AnswerStatus = 'idle' | 'running' | 'done' | 'failed';

/**
 * The state of one streamed answer from POST /api/ask.
 *
 * A plain class rather than a service, because Compare holds four of them at
 * once and Ask holds one. It applies the SDK's typed events in the order the
 * server sends them (retrieval, tokens, optionally superseded, citations,
 * done) and keeps everything the pages draw.
 */
export class AnswerState {
  readonly status = signal<AnswerStatus>('idle');
  readonly text = signal('');
  readonly retrieval = signal<RetrievalEvent['data'] | null>(null);
  readonly superseded = signal<SupersededEvent['data'] | null>(null);
  readonly citations = signal<Citation[]>([]);
  readonly done = signal<DoneEvent['data'] | null>(null);
  readonly error = signal<string | null>(null);

  /**
   * The server declined to answer. The `done` event carries an explicit `refused`
   * flag, which is used when present. A server that predates the flag does not send
   * it, so the old reading (the contract's phrase and no citation marker) is kept
   * as the fallback.
   */
  readonly refused = computed(() => {
    if (this.status() !== 'done') {
      return false;
    }
    const flag = this.done()?.refused;
    if (typeof flag === 'boolean') {
      return flag;
    }
    return this.text().toLowerCase().includes('could not find') && !/\[\d+\]/.test(this.text());
  });

  /** Why a refusal happened, in wording safe to show, or null when the API gave none. */
  readonly refusalReason = computed<string | null>(() => {
    if (!this.refused()) {
      return null;
    }
    const retrieval = this.retrieval();
    const graph = retrieval?.subgraph?.empty_reason;
    if (graph) {
      return GRAPH_REASONS[graph] ?? 'The knowledge graph had nothing to answer from.';
    }
    if (retrieval && retrieval.chunks === 0) {
      return 'No passages matched this question.';
    }
    return null;
  });

  /** Back to nothing asked. */
  reset(): void {
    this.text.set('');
    this.retrieval.set(null);
    this.superseded.set(null);
    this.citations.set([]);
    this.done.set(null);
    this.error.set(null);
    this.status.set('idle');
  }

  start(): void {
    this.reset();
    this.status.set('running');
  }

  apply(event: AskEvent): void {
    switch (event.event) {
      case 'retrieval':
        this.retrieval.set(event.data);
        break;
      case 'token':
        this.text.update((text) => text + event.data.text);
        break;
      case 'superseded':
        // The streamed text failed the citation contract and was repaired.
        // The repaired text is what the server recorded, so it replaces what
        // was drawn (decision D7); the reason stays visible.
        this.text.set(event.data.text);
        this.superseded.set(event.data);
        break;
      case 'citations':
        this.citations.set(event.data.citations);
        break;
      case 'done':
        this.done.set(event.data);
        this.status.set('done');
        break;
      case 'unknown':
        break;
    }
  }

  /** The stream closed. Without a `done` event the answer is not final. */
  complete(): void {
    if (this.status() === 'running') {
      this.fail('The stream ended before the answer finished.');
    }
  }

  fail(message: string): void {
    this.error.set(message);
    this.status.set('failed');
  }
}
