import { ChangeDetectionStrategy, Component, computed, inject, input, output, signal } from '@angular/core';
import { Citation } from '@ragfabric/sdk';

import { DocumentService } from '../services/document.service';
import { describeError } from '../services/http-error';
import { ModalComponent } from '../ui';
import { score } from './format';
import { snippetSegments } from './segments';

/**
 * The source behind one citation: where it came from, the passage with the
 * query terms and the quoted sentence marked, and the original document.
 */
@Component({
  selector: 'app-source-viewer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [ModalComponent],
  template: `
    <ui-modal [open]="citation() !== null" [title]="title()" (closed)="closed.emit()">
      @if (citation(); as c) {
        <dl class="facts">
          <div><dt>Document</dt><dd data-test="filename">{{ c.filename ?? 'unknown' }}</dd></div>
          <div><dt>Page</dt><dd>{{ c.page ?? 'n/a' }}</dd></div>
          <div><dt>Score</dt><dd data-test="score">{{ scoreText() }}</dd></div>
          <div><dt>Cited in the answer</dt><dd>{{ c.used ? 'yes' : 'no, retrieved only' }}</dd></div>
        </dl>
        <blockquote class="snippet" data-test="snippet">
          @for (part of parts(); track $index) {
            <span [class.hl]="part.highlight" [class.support]="part.support">{{ part.text }}</span>
          }
        </blockquote>
        <p class="legend muted">
          <span class="support">Underlined</span>: the sentence the answer quoted.
          <span class="hl">Highlighted</span>: words from your question.
        </p>
        @if (error()) {
          <p class="alert alert-error" role="alert">{{ error() }}</p>
        }
      }
      <div modal-actions>
        @if (citation()?.document_id != null) {
          <button type="button" class="btn btn-sm" data-test="download" (click)="download()" [disabled]="downloading()">
            Download original
          </button>
        }
      </div>
    </ui-modal>
  `,
  styles: `
    .facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-2); margin: 0 0 var(--space-3); }
    .facts dt { font-size: 0.75rem; color: var(--text-muted); }
    .facts dd { margin: 0; overflow-wrap: anywhere; }
    .snippet { margin: 0; padding: var(--space-3); background: var(--surface-2); border-radius: var(--radius-sm);
      white-space: pre-wrap; line-height: 1.6; }
    .hl { background: var(--warning-soft); }
    .support { text-decoration: underline; text-decoration-color: var(--brand); text-underline-offset: 3px; }
    .legend { font-size: 0.8rem; }
  `,
})
export class SourceViewerComponent {
  private readonly documents = inject(DocumentService);

  readonly citation = input<Citation | null>(null);
  readonly closed = output<void>();
  readonly downloading = signal(false);
  readonly error = signal<string | null>(null);

  readonly title = computed(() => {
    const c = this.citation();
    return c ? `Source ${c.marker}` : '';
  });
  readonly scoreText = computed(() => {
    const value = this.citation()?.score;
    return value == null ? 'not measured (graph traversal)' : score(value, 3);
  });
  readonly parts = computed(() => {
    const c = this.citation();
    return c ? snippetSegments(c.snippet, c.highlights ?? [], c.supporting_span ?? null) : [];
  });

  download(): void {
    const c = this.citation();
    if (c?.document_id == null) {
      return;
    }
    this.downloading.set(true);
    this.error.set(null);
    this.documents.download(c.document_id).subscribe({
      next: (blob) => {
        this.downloading.set(false);
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = c.filename ?? `document-${c.document_id}`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 0);
      },
      error: (err: unknown) => {
        this.downloading.set(false);
        this.error.set(describeError(err, 'The document could not be downloaded.'));
      },
    });
  }
}
