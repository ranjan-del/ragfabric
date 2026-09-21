import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';

import { CollectionService } from '../../services/collection.service';
import { describeError } from '../../services/http-error';
import { Collection } from '../../models';
import {
  ButtonComponent,
  ConfirmDialogComponent,
  EmptyStateComponent,
  InputComponent,
  ModalComponent,
  ToastService,
} from '../../ui';

/**
 * Collections, on the console primitives (Task 14).
 *
 * Three things changed here. A collection can now be renamed, which it could
 * not be: the only way to fix a name was to delete the collection, and that
 * cascades to every document in it. A failed load now renders as an error
 * rather than as the same empty page an account with no collections gets.
 * And delete asks for the name to be typed, after saying how many documents
 * go with it, instead of a browser confirm() that reads as noise.
 */
@Component({
  selector: 'app-collections',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    ButtonComponent,
    ConfirmDialogComponent,
    EmptyStateComponent,
    InputComponent,
    ModalComponent,
  ],
  templateUrl: './collections.component.html',
  styleUrl: './collections.component.scss',
})
export class CollectionsComponent implements OnInit {
  private readonly svc = inject(CollectionService);
  private readonly toasts = inject(ToastService);

  readonly collections = signal<Collection[]>([]);
  readonly loading = signal(true);
  readonly creating = signal(false);
  readonly loadError = signal('');
  readonly error = signal('');

  readonly name = signal('');
  readonly description = signal('');

  readonly editing = signal<Collection | null>(null);
  readonly editName = signal('');
  readonly editDescription = signal('');
  readonly editError = signal('');

  readonly pendingDelete = signal<Collection | null>(null);

  readonly deleteConsequences = computed(() => {
    const collection = this.pendingDelete();
    if (collection === null) {
      return [];
    }
    return [
      `${collection.document_count} document(s) in this collection are deleted with it.`,
      'Their chunks are removed from the vector and lexical indexes.',
      'Any grant that gave a group access to this collection is removed.',
    ];
  });

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.loadError.set('');
    this.svc.list().subscribe({
      next: (data) => {
        this.collections.set(data);
        this.loading.set(false);
      },
      error: (err) => {
        this.loading.set(false);
        // Previously swallowed, which made a broken API indistinguishable
        // from an account with no collections.
        this.loadError.set(describeError(err, 'Could not load collections.'));
      },
    });
  }

  create(): void {
    const name = this.name().trim();
    if (!name) {
      return;
    }
    this.creating.set(true);
    this.error.set('');
    this.svc.create(name, this.description().trim()).subscribe({
      next: () => {
        this.name.set('');
        this.description.set('');
        this.creating.set(false);
        this.refresh();
      },
      error: (err) => {
        this.creating.set(false);
        this.error.set(describeError(err, 'Failed to create collection.'));
      },
    });
  }

  openEdit(collection: Collection): void {
    this.editError.set('');
    this.editName.set(collection.name);
    this.editDescription.set(collection.description);
    this.editing.set(collection);
  }

  saveEdit(): void {
    const collection = this.editing();
    if (collection === null) {
      return;
    }
    const name = this.editName().trim();
    if (!name) {
      this.editError.set('A collection needs a name.');
      return;
    }
    this.svc.update(collection.id, { name, description: this.editDescription() }).subscribe({
      next: () => {
        this.editing.set(null);
        this.toasts.success('Collection updated.');
        this.refresh();
      },
      error: (err) => this.editError.set(describeError(err, 'Could not rename that collection.')),
    });
  }

  askDelete(collection: Collection): void {
    this.error.set('');
    this.pendingDelete.set(collection);
  }

  confirmDelete(): void {
    const collection = this.pendingDelete();
    if (collection === null) {
      return;
    }
    this.pendingDelete.set(null);
    this.svc.delete(collection.id).subscribe({
      next: () => {
        this.toasts.success(`Deleted ${collection.name}.`);
        this.refresh();
      },
      error: (err) => {
        const message = describeError(err, 'Could not delete that collection.');
        this.error.set(message);
        this.toasts.error(message);
      },
    });
  }
}
