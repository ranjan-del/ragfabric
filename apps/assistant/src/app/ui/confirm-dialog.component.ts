import { ChangeDetectionStrategy, Component, computed, input, output, signal } from '@angular/core';

import { ModalComponent } from './modal.component';

let nextId = 0;

/**
 * A destructive confirmation that cannot be clicked through.
 *
 * "Are you sure?" is answered yes by reflex, so the confirm button stays
 * disabled until the operator types the name of the thing being destroyed.
 * The dialog also lists what ELSE goes with it, because the surprising part
 * of deleting a user is never the user, it is the API keys that stop working
 * and the group memberships that disappear.
 */
@Component({
  selector: 'ui-confirm-dialog',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [ModalComponent],
  template: `
    <ui-modal [open]="open()" [title]="title()" (closed)="cancel()">
      <p class="confirm-message">{{ message() }}</p>
      @if (consequences().length > 0) {
        <ul class="confirm-consequences">
          @for (line of consequences(); track line) {
            <li>{{ line }}</li>
          }
        </ul>
      }
      @if (confirmWord()) {
        <div class="field confirm-word">
          <label [attr.for]="wordId">Type <strong>{{ confirmWord() }}</strong> to confirm</label>
          <input
            [id]="wordId"
            type="text"
            autocomplete="off"
            [value]="typed()"
            (input)="onType($event)"
          />
        </div>
      }
      <div modal-actions class="row">
        <button type="button" class="btn btn-secondary" (click)="cancel()">Cancel</button>
        <button
          type="button"
          class="btn btn-danger confirm-action"
          [disabled]="!canConfirm()"
          (click)="confirm()"
        >
          {{ confirmLabel() }}
        </button>
      </div>
    </ui-modal>
  `,
})
export class ConfirmDialogComponent {
  readonly wordId = `ui-confirm-${nextId++}`;
  readonly open = input(false);
  readonly title = input('Are you sure?');
  readonly message = input('');
  readonly consequences = input<string[]>([]);
  /** The exact text the operator must type. Empty means a plain confirm. */
  readonly confirmWord = input('');
  readonly confirmLabel = input('Delete');
  readonly confirmed = output<void>();
  readonly cancelled = output<void>();

  readonly typed = signal('');
  readonly canConfirm = computed(
    () => !this.confirmWord() || this.typed().trim() === this.confirmWord(),
  );

  onType(event: Event): void {
    this.typed.set((event.target as HTMLInputElement).value);
  }

  confirm(): void {
    if (!this.canConfirm()) {
      return;
    }
    this.typed.set('');
    this.confirmed.emit();
  }

  cancel(): void {
    this.typed.set('');
    this.cancelled.emit();
  }
}
