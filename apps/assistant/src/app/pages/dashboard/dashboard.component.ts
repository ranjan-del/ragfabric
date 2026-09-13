import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';

import { AnalyticsService } from '../../services/analytics.service';
import { AuthService } from '../../services/auth.service';
import { AnalyticsOverview, UsageStats } from '../../models';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
})
export class DashboardComponent implements OnInit {
  overview = signal<AnalyticsOverview | null>(null);
  usage = signal<UsageStats | null>(null);
  loading = signal(true);

  constructor(
    private analytics: AnalyticsService,
    public auth: AuthService
  ) {}

  ngOnInit(): void {
    this.analytics.overview().subscribe({
      next: (data) => {
        this.overview.set(data);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
    this.analytics.usage().subscribe({ next: (data) => this.usage.set(data) });
  }

  get tiles() {
    const o = this.overview();
    if (!o) return [];
    return [
      { label: 'Documents', value: o.documents, icon: '▤', hint: `${o.ready_documents} ready` },
      { label: 'Collections', value: o.collections, icon: '◫', hint: 'grouped sets' },
      { label: 'Indexed chunks', value: o.chunks, icon: '⛁', hint: 'searchable spans' },
      { label: 'Questions asked', value: o.queries, icon: '✦', hint: 'all time' },
    ];
  }
}
