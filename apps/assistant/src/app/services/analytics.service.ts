import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiService } from './api.service';
import { AnalyticsOverview, UsageStats } from '../models';

@Injectable({ providedIn: 'root' })
export class AnalyticsService {
  constructor(private http: HttpClient, private api: ApiService) {}

  overview(): Observable<AnalyticsOverview> {
    return this.http.get<AnalyticsOverview>(`${this.api.baseUrl}/analytics/overview`);
  }

  usage(): Observable<UsageStats> {
    return this.http.get<UsageStats>(`${this.api.baseUrl}/analytics/usage`);
  }
}
