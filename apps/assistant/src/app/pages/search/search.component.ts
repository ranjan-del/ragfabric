import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

import { SearchService } from '../../services/search.service';
import { CollectionService } from '../../services/collection.service';
import { AnswerResponse, Collection } from '../../models';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './search.component.html',
  styleUrl: './search.component.scss',
})
export class SearchComponent implements OnInit {
  query = '';
  mode: 'semantic' | 'hybrid' = 'semantic';
  collectionId: number | null = null;

  collections = signal<Collection[]>([]);
  answer = signal<AnswerResponse | null>(null);
  loading = signal(false);
  error = signal<string | null>(null);
  asked = signal(false);

  // Terms the backend told us to highlight, captured from the last answer and
  // consumed by highlight() when rendering snippets.
  private highlightTerms: string[] = [];

  constructor(
    private search: SearchService,
    private collectionsSvc: CollectionService,
    private sanitizer: DomSanitizer
  ) {}

  ngOnInit(): void {
    this.collectionsSvc.list().subscribe({
      next: (data) => this.collections.set(data),
    });
  }

  ask(): void {
    const q = this.query.trim();
    if (!q) return;
    this.loading.set(true);
    this.error.set(null);
    this.asked.set(true);

    this.search
      .ask(q, { mode: this.mode, collection_id: this.collectionId })
      .subscribe({
        next: (res) => {
          this.answer.set(res);
          this.highlightTerms = res.highlights.map((h) => h.term);
          this.loading.set(false);
        },
        error: (err) => {
          this.loading.set(false);
          this.error.set(
            (err as { error?: { detail?: string } })?.error?.detail ||
              'Something went wrong. Please try again.'
          );
        },
      });
  }

  confidenceLabel(score: number): string {
    if (score >= 0.6) return 'High';
    if (score >= 0.3) return 'Medium';
    return 'Low';
  }

  confidenceClass(score: number): string {
    if (score >= 0.6) return 'high';
    if (score >= 0.3) return 'medium';
    return 'low';
  }

  /** Escape HTML then wrap any matched highlight terms in <mark>. */
  highlight(text: string): SafeHtml {
    const escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    if (!this.highlightTerms.length) {
      return this.sanitizer.bypassSecurityTrustHtml(escaped);
    }
    const unique = [...new Set(this.highlightTerms)].filter(Boolean);
    const escapedTerms = unique.map((t) =>
      t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    );
    const pattern = new RegExp(`\\b(${escapedTerms.join('|')})`, 'gi');
    return this.sanitizer.bypassSecurityTrustHtml(
      escaped.replace(pattern, '<mark>$1</mark>')
    );
  }
}
