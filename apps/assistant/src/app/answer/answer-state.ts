import { signal } from '@angular/core';
import {
  AskEvent,
  Citation,
  DoneEvent,
  RetrievalEvent,
  SupersededEvent,
} from '@ragfabric/sdk';

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
