import { Component, OnInit, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';

import { AnalyticsService } from '../../services/analytics.service';
import { AnalyticsOverview, UsageStats } from '../../models';

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './analytics.component.html',
  styleUrl: './analytics.component.scss',
})
export class AnalyticsComponent implements OnInit {
  overview = signal<AnalyticsOverview | null>(null);
  usage = signal<UsageStats | null>(null);
  loading = signal(true);

  readonly maxChunks = computed(() => {
    const docs = this.usage()?.top_documents ?? [];
    return docs.reduce((m, d) => Math.max(m, d.chunks), 0) || 1;
  });

  constructor(private analytics: AnalyticsService) {}

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

  barWidth(chunks: number): string {
    return `${Math.max(6, (chunks / this.maxChunks()) * 100)}%`;
  }
}
