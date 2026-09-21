import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { ToastService } from './toast.service';

/**
 * Renders the toast stack.
 *
 * aria-live is assertive because every toast here reports the outcome of
 * something the operator just did to access control, and a polite region can
 * wait behind whatever else is being read out.
 */
@Component({
  selector: 'ui-toasts',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="toast-stack" aria-live="assertive" aria-atomic="true">
      @for (toast of toasts.toasts(); track toast.id) {
        <div
          class="toast"
          [attr.data-tone]="toast.tone"
          [attr.role]="toast.tone === 'error' ? 'alert' : 'status'"
        >
          <span class="toast-message">{{ toast.message }}</span>
          <button type="button" aria-label="Dismiss" (click)="toasts.dismiss(toast.id)">
            &#10005;
          </button>
        </div>
      }
    </div>
  `,
})
export class ToastsComponent {
  readonly toasts = inject(ToastService);
}
