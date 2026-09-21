import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

export type BadgeTone = 'neutral' | 'brand' | 'success' | 'warning' | 'danger';

/**
 * A status pill.
 *
 * The dot is decorative and the tone is also written to a data attribute, so
 * the state is carried by the text and by a machine readable attribute, never
 * by the colour alone. A colour blind operator and a screen reader both get
 * the same answer as everyone else.
 */
@Component({
  selector: 'ui-badge',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span [class]="classes()" [attr.data-tone]="tone()">
      <span class="badge-dot" aria-hidden="true"></span><ng-content />
    </span>
  `,
})
export class BadgeComponent {
  readonly tone = input<BadgeTone>('neutral');
  readonly classes = computed(() => `badge badge-${this.tone()}`);
}
