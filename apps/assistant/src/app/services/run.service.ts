import { Injectable, inject } from '@angular/core';
import { Run, requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';

/** Stored runs: GET /api/runs/{id}, the Trace page's source and Compare's cost. */
@Injectable({ providedIn: 'root' })
export class RunService {
  private readonly sdk = inject(SdkHttp);

  get(id: number): Observable<Run> {
    return this.sdk.send(requests.runs.get(id));
  }
}
