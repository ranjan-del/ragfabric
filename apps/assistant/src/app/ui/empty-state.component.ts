import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * What a screen shows when there is genuinely nothing to show.
 *
 * Distinct from the loading and error states on purpose: "no users yet" and
 * "the request failed" look identical as a blank table, and only one of them
 * means the operator should do something about the server.
 */
@Component({
  selector: 'ui-empty-state',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="empty-state">
      <h3>{{ title() }}</h3>
      @if (message()) {
        <p class="muted">{{ message() }}</p>
      }
      <div class="empty-actions"><ng-content /></div>
    </div>
  `,
})
export class EmptyStateComponent {
  readonly title = input('Nothing here yet');
  readonly message = input('');
}
