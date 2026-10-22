import { Component, OnInit, signal, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { DocumentService } from '../../services/document.service';
import { describeError } from '../../services/http-error';
import { CollectionService } from '../../services/collection.service';
import { Collection, DocumentItem } from '../../models';

@Component({
    selector: 'app-documents',
    imports: [CommonModule, FormsModule],
    templateUrl: './documents.component.html',
    changeDetection: ChangeDetectionStrategy.Eager,
    styleUrl: './documents.component.scss'
})
export class DocumentsComponent implements OnInit {
  readonly accept = '.pdf,.docx,.pptx,.txt,.csv,.md';
  documents = signal<DocumentItem[]>([]);
  collections = signal<Collection[]>([]);
  uploadCollection: number | null = null;

  loading = signal(true);
  uploading = signal(false);
  dragging = signal(false);
  /** One line per rejected file, with the server's own reason. */
  errors = signal<string[]>([]);

  constructor(
    private docs: DocumentService,
    private collectionsSvc: CollectionService
  ) {}

  ngOnInit(): void {
    this.refresh();
    this.collectionsSvc.list().subscribe({
      next: (data) => this.collections.set(data),
    });
  }

  refresh(): void {
    this.loading.set(true);
    this.docs.list().subscribe({
      next: (data) => {
        this.documents.set(data.items);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) {
      this.uploadFiles(Array.from(input.files));
      input.value = '';
    }
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    if (event.dataTransfer?.files?.length) {
      this.uploadFiles(Array.from(event.dataTransfer.files));
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(true);
  }

  onDragLeave(): void {
    this.dragging.set(false);
  }

  private uploadFiles(files: File[]): void {
    this.errors.set([]);
    let remaining = files.length;
    this.uploading.set(true);
    for (const file of files) {
      this.docs.upload(file, this.uploadCollection ?? undefined).subscribe({
        next: () => {
          remaining -= 1;
          if (remaining === 0) {
            this.uploading.set(false);
            this.refresh();
          }
        },
        error: (err) => {
          remaining -= 1;
          this.errors.update((list) => [
            ...list,
            `${file.name}: ${describeError(err, 'the upload failed.')}`,
          ]);
          if (remaining === 0) {
            this.uploading.set(false);
            this.refresh();
          }
        },
      });
    }
  }

  remove(doc: DocumentItem): void {
    if (!confirm(`Delete "${doc.filename}"? This removes it from search.`)) return;
    this.docs.delete(doc.id).subscribe({ next: () => this.refresh() });
  }

  statusClass(status: string): string {
    if (status === 'ready') return 'badge-success';
    if (status === 'failed') return 'badge-danger';
    return 'badge-warning';
  }

  collectionName(id: number | null): string {
    if (id == null) return '—';
    return this.collections().find((c) => c.id === id)?.name ?? `#${id}`;
  }
}
