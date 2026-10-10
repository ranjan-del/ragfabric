import { Injectable, inject } from '@angular/core';
import { AskEvent, AskOptions, requests } from '@ragfabric/sdk';
import { Observable, Subscription } from 'rxjs';

import { AnswerState } from '../answer/answer-state';
import { SdkHttp } from '../sdk/sdk-http.service';
import { describeError } from './http-error';
import { RunDetailsService } from './run-details.service';

/** Streams answers from POST /api/ask into an AnswerState. */
@Injectable({ providedIn: 'root' })
export class AskService {
  private readonly sdk = inject(SdkHttp);
  private readonly details = inject(RunDetailsService);

  stream(query: string, options: AskOptions): Observable<AskEvent> {
    return this.sdk.stream(requests.askStream(query, options));
  }

  /**
   * Start streaming into `state`. Unsubscribe to abort the request. `onEnd`
   * runs once when the stream finishes or fails, which is how Compare starts
   * its next column.
   */
  run(state: AnswerState, query: string, options: AskOptions, onEnd?: () => void): Subscription {
    state.start();
    return this.stream(query, options).subscribe({
      next: (event) => {
        state.apply(event);
        if (event.event === 'done') {
          this.details.remember(event.data.run_id, {
            question: query,
            retrieval: state.retrieval(),
            superseded: state.superseded(),
          });
        }
      },
      error: (error: unknown) => {
        state.fail(describeError(error, 'The answer could not be streamed.'));
        onEnd?.();
      },
      complete: () => {
        state.complete();
        onEnd?.();
      },
    });
  }
}
