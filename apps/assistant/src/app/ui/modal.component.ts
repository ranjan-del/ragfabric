import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  effect,
  input,
  output,
  viewChild,
} from '@angular/core';

let nextId = 0;

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]),' +
  ' textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * A modal dialog that a keyboard user can actually leave.
 *
 * Three obligations, all of them load bearing and all of them tested:
 *
 * 1. Focus moves into the dialog when it opens. Otherwise the next Tab lands
 *    somewhere behind the dialog that the operator cannot see.
 * 2. Tab cycles within the dialog. Without the wrap, focus walks out of the
 *    dialog and into the page underneath while the dialog still covers it.
 * 3. Focus returns to whatever opened the dialog when it closes, so the
 *    keyboard position is where the operator left it rather than back at the
 *    top of the document.
 */
@Component({
  selector: 'ui-modal',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (open()) {
      <div class="modal-backdrop" (click)="requestClose()"></div>
      <div
        #panel
        class="modal"
        role="dialog"
        aria-modal="true"
        [attr.aria-labelledby]="titleId"
        (keydown)="onKeydown($event)"
      >
        <header class="modal-head">
          <h2 [id]="titleId">{{ title() }}</h2>
          <button type="button" class="btn btn-ghost btn-sm" aria-label="Close" (click)="requestClose()">
            &#10005;
          </button>
        </header>
        <div class="modal-body">
          <ng-content />
        </div>
        <footer class="modal-foot">
          <ng-content select="[modal-actions]" />
        </footer>
      </div>
    }
  `,
})
export class ModalComponent {
  readonly titleId = `ui-modal-${nextId++}-title`;
  readonly open = input(false);
  readonly title = input('');
  readonly closed = output<void>();

  private readonly panel = viewChild<ElementRef<HTMLElement>>('panel');
  private opener: HTMLElement | null = null;

  constructor() {
    effect(() => {
      if (this.open()) {
        this.opener = document.activeElement as HTMLElement | null;
        // Queued rather than immediate: the panel is created by the @if in
        // this same change detection pass, so it is not focusable yet.
        setTimeout(() => this.focusFirst());
      } else if (this.opener) {
        const opener = this.opener;
        this.opener = null;
        setTimeout(() => opener.focus());
      }
    });
  }

  requestClose(): void {
    this.closed.emit();
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.stopPropagation();
      this.requestClose();
      return;
    }
    if (event.key !== 'Tab') {
      return;
    }
    const items = this.focusable();
    if (items.length === 0) {
      return;
    }
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && active === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  }

  private focusable(): HTMLElement[] {
    const panel = this.panel()?.nativeElement;
    return panel ? Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE)) : [];
  }

  private focusFirst(): void {
    const items = this.focusable();
    if (items.length > 0) {
      items[0].focus();
    } else {
      this.panel()?.nativeElement.focus();
    }
  }
}
