import { Injectable, inject } from '@angular/core';
import { EvalDashboard, EvalRun, EvalRunDetail, requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';

/**
 * Phase 8's read only evaluation API (admin only). Provisional: built against
 * Phase 8's planned shapes and to be re-verified when it merges (decision D20).
 */
@Injectable({ providedIn: 'root' })
export class EvaluationService {
  private readonly sdk = inject(SdkHttp);

  runs(limit = 100): Observable<EvalRun[]> {
    return this.sdk.send(requests.evaluation.runs(limit));
  }

  run(id: number): Observable<EvalRunDetail> {
    return this.sdk.send(requests.evaluation.run(id));
  }

  dashboard(days = 30): Observable<EvalDashboard> {
    return this.sdk.send(requests.evaluation.dashboard(days));
  }
}
