import { Injectable } from '@angular/core';
import { RetrievalEvent, SupersededEvent } from '@ragfabric/sdk';

export interface RunDetails {
  question: string;
  retrieval: RetrievalEvent['data'] | null;
  superseded: SupersededEvent['data'] | null;
}

/**
 * What a streamed run reported that the server does not store with the run.
 *
 * The walked sub graph, the agent's sub questions and the full router
 * decision travel on the `retrieval` event but are not persisted (decision
 * D17), so Trace can only show them for runs this browser session streamed.
 * Kept in memory only: nothing here outlives the tab, and nothing is written
 * to storage another user of the machine could read.
 */
@Injectable({ providedIn: 'root' })
export class RunDetailsService {
  private readonly runs = new Map<number, RunDetails>();

  remember(runId: number, details: RunDetails): void {
    this.runs.set(runId, details);
  }

  get(runId: number): RunDetails | null {
    return this.runs.get(runId) ?? null;
  }
}
