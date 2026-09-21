import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md';

/**
 * The console's only button.
 *
 * It exists so that "this action is destructive" is said once, as a variant,
 * rather than re-derived from class names at every call site, and so that a
 * button which is busy is also disabled. A button that stays clickable while
 * its request is in flight is how an operator creates two users by accident.
 */
@Component({
  selector: 'ui-button',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <button
      [class]="classes()"
      [attr.type]="type()"
      [disabled]="disabled() || loading()"
      [attr.aria-busy]="loading() ? 'true' : null"
      (click)="clicked.emit($event)"
    >
      @if (loading()) {
        <span class="spinner spinner-inline" aria-hidden="true"></span>
      }
      <ng-content />
    </button>
  `,
})
export class ButtonComponent {
  readonly variant = input<ButtonVariant>('secondary');
  readonly size = input<ButtonSize>('md');
  readonly type = input<'button' | 'submit'>('button');
  readonly disabled = input(false);
  readonly loading = input(false);
  readonly clicked = output<MouseEvent>();

  readonly classes = computed(() => {
    const parts = ['btn', `btn-${this.variant()}`];
    if (this.size() === 'sm') {
      parts.push('btn-sm');
    }
    return parts.join(' ');
  });
}
