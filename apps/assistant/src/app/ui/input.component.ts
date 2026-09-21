import { ChangeDetectionStrategy, Component, computed, input, model } from '@angular/core';

let nextId = 0;

/**
 * A labelled text field.
 *
 * The label is tied to the control with a generated id rather than left as
 * adjacent text, and an error is both announced (aria-invalid plus
 * aria-describedby) and written out. A red border on its own tells a screen
 * reader nothing and tells a colour blind operator nothing.
 */
@Component({
  selector: 'ui-input',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="field">
      <label [attr.for]="id">
        {{ label() }}
        @if (required()) {
          <span class="field-required" aria-hidden="true">*</span>
        }
      </label>
      <input
        [id]="id"
        [attr.type]="type()"
        [attr.placeholder]="placeholder() || null"
        [attr.autocomplete]="autocomplete()"
        [attr.aria-invalid]="error() ? 'true' : null"
        [attr.aria-describedby]="describedBy()"
        [attr.required]="required() ? '' : null"
        [disabled]="disabled()"
        [value]="value()"
        (input)="onInput($event)"
      />
      @if (hint() && !error()) {
        <p class="field-hint" [id]="hintId">{{ hint() }}</p>
      }
      @if (error()) {
        <p class="field-error" [id]="errorId">{{ error() }}</p>
      }
    </div>
  `,
})
export class InputComponent {
  private readonly uid = `ui-input-${nextId++}`;
  readonly id = this.uid;
  readonly hintId = `${this.uid}-hint`;
  readonly errorId = `${this.uid}-error`;

  readonly label = input('');
  readonly type = input('text');
  readonly placeholder = input('');
  readonly hint = input('');
  readonly error = input('');
  readonly required = input(false);
  readonly disabled = input(false);
  readonly autocomplete = input<string | null>(null);
  readonly value = model('');

  readonly describedBy = computed(() => {
    if (this.error()) {
      return this.errorId;
    }
    return this.hint() ? this.hintId : null;
  });

  onInput(event: Event): void {
    this.value.set((event.target as HTMLInputElement).value);
  }
}
