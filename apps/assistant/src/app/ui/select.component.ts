import { ChangeDetectionStrategy, Component, input, model } from '@angular/core';

export interface SelectOption {
  value: string;
  label: string;
}

let nextId = 0;

/** A labelled select. Same labelling contract as ui-input. */
@Component({
  selector: 'ui-select',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="field">
      <label [attr.for]="id">{{ label() }}</label>
      <select [id]="id" [disabled]="disabled()" [value]="value()" (change)="onChange($event)">
        @for (option of options(); track option.value) {
          <option [value]="option.value" [selected]="option.value === value()">
            {{ option.label }}
          </option>
        }
      </select>
      @if (hint()) {
        <p class="field-hint">{{ hint() }}</p>
      }
    </div>
  `,
})
export class SelectComponent {
  readonly id = `ui-select-${nextId++}`;
  readonly label = input('');
  readonly hint = input('');
  readonly disabled = input(false);
  readonly options = input<SelectOption[]>([]);
  readonly value = model('');

  onChange(event: Event): void {
    this.value.set((event.target as HTMLSelectElement).value);
  }
}
