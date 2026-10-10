import { Injectable, inject } from '@angular/core';
import { requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';
import { AnalyticsOverview, UsageStats } from '../models';

@Injectable({ providedIn: 'root' })
export class AnalyticsService {
  private readonly sdk = inject(SdkHttp);

  overview(): Observable<AnalyticsOverview> {
    return this.sdk.send(requests.analytics.overview());
  }

  usage(): Observable<UsageStats> {
    return this.sdk.send(requests.analytics.usage());
  }
}
