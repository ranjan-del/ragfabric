import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { CollectionService } from '../../services/collection.service';
import { Collection } from '../../models';

@Component({
  selector: 'app-collections',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './collections.component.html',
  styleUrl: './collections.component.scss',
})
export class CollectionsComponent implements OnInit {
  collections = signal<Collection[]>([]);
  loading = signal(true);
  creating = signal(false);
  error = signal<string | null>(null);

  name = '';
  description = '';

  constructor(private svc: CollectionService) {}

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.svc.list().subscribe({
      next: (data) => {
        this.collections.set(data);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  create(): void {
    const name = this.name.trim();
    if (!name) return;
    this.creating.set(true);
    this.error.set(null);
    this.svc.create(name, this.description.trim()).subscribe({
      next: () => {
        this.name = '';
        this.description = '';
        this.creating.set(false);
        this.refresh();
      },
      error: (err) => {
        this.creating.set(false);
        this.error.set(
          (err as { error?: { detail?: string } })?.error?.detail ||
            'Failed to create collection.'
        );
      },
    });
  }

  remove(collection: Collection): void {
    if (
      !confirm(
        `Delete "${collection.name}" and its ${collection.document_count} document(s)?`
      )
    )
      return;
    this.svc.delete(collection.id).subscribe({ next: () => this.refresh() });
  }
}
