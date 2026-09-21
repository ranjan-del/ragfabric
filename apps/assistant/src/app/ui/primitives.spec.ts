// UI primitives (Task 12).
//
// One spec file for the nine primitives, because they are small and their
// obligations overlap: render, disabled state, keyboard interaction, and the
// accessibility rules the console is held to. The two that carry real risk
// get the most attention here. Modal has to trap Tab and give focus back, or
// a keyboard user is stranded behind a dialog. Table has to distinguish
// "loading", "failed" and "genuinely empty", because collapsing the three
// into one blank table is how a broken console looks like a working one with
// no data.

import { Component, signal } from '@angular/core';
import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';

import { BadgeComponent } from './badge.component';
import { ButtonComponent } from './button.component';
import { ConfirmDialogComponent } from './confirm-dialog.component';
import { EmptyStateComponent } from './empty-state.component';
import { InputComponent } from './input.component';
import { ModalComponent } from './modal.component';
import { SelectComponent } from './select.component';
import { TableComponent } from './table.component';
import { ThemeService } from './theme.service';
import { ToastService } from './toast.service';
import { ToastsComponent } from './toasts.component';

function el<T extends HTMLElement>(fixture: ComponentFixture<unknown>, selector: string): T {
  const found = fixture.nativeElement.querySelector(selector) as T | null;
  if (found === null) {
    throw new Error(`no element matched ${selector}`);
  }
  return found;
}

describe('ui-button', () => {
  @Component({
    imports: [ButtonComponent],
    template: `
      <ui-button [variant]="'danger'" [disabled]="disabled()" (clicked)="clicks = clicks + 1">
        Delete
      </ui-button>
    `,
  })
  class Host {
    readonly disabled = signal(false);
    clicks = 0;
  }

  let fixture: ComponentFixture<Host>;

  beforeEach(() => {
    fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
  });

  it('test_a_button_renders_its_label_and_variant', () => {
    const button = el<HTMLButtonElement>(fixture, 'button');
    expect(button.textContent?.trim()).toBe('Delete');
    expect(button.className).toContain('btn-danger');
  });

  it('test_a_disabled_button_does_not_emit_a_click', () => {
    fixture.componentInstance.disabled.set(true);
    fixture.detectChanges();
    const button = el<HTMLButtonElement>(fixture, 'button');

    expect(button.disabled).toBeTrue();
    button.click();
    expect(fixture.componentInstance.clicks).toBe(0);
  });

  it('test_a_button_is_reachable_by_keyboard', () => {
    const button = el<HTMLButtonElement>(fixture, 'button');
    button.focus();
    expect(document.activeElement).toBe(button);
  });
});

describe('ui-input', () => {
  @Component({
    imports: [InputComponent],
    template: `
      <ui-input [label]="'Email'" [error]="error()" [(value)]="value" [hint]="'Work address'" />
    `,
  })
  class Host {
    readonly error = signal('');
    value = '';
  }

  let fixture: ComponentFixture<Host>;

  beforeEach(() => {
    fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
  });

  it('test_the_label_is_tied_to_the_input', () => {
    const label = el<HTMLLabelElement>(fixture, 'label');
    const input = el<HTMLInputElement>(fixture, 'input');

    expect(input.id).toBeTruthy();
    expect(label.htmlFor).toBe(input.id);
  });

  it('test_typing_writes_back_through_the_model', () => {
    const input = el<HTMLInputElement>(fixture, 'input');
    input.value = 'someone@example.com';
    input.dispatchEvent(new Event('input'));
    fixture.detectChanges();

    expect(fixture.componentInstance.value).toBe('someone@example.com');
  });

  it('test_an_error_is_described_not_merely_coloured', () => {
    fixture.componentInstance.error.set('That email is already taken.');
    fixture.detectChanges();
    const input = el<HTMLInputElement>(fixture, 'input');
    const message = el<HTMLElement>(fixture, '.field-error');

    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toContain(message.id);
    expect(message.textContent).toContain('already taken');
  });
});

describe('ui-select', () => {
  @Component({
    imports: [SelectComponent],
    template: `
      <ui-select
        [label]="'Role'"
        [options]="[
          { value: 'user', label: 'User' },
          { value: 'admin', label: 'Admin' },
        ]"
        [(value)]="value"
      />
    `,
  })
  class Host {
    value = 'user';
  }

  it('test_a_select_labels_and_renders_every_option', () => {
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
    const select = el<HTMLSelectElement>(fixture, 'select');
    const label = el<HTMLLabelElement>(fixture, 'label');

    expect(label.htmlFor).toBe(select.id);
    expect(Array.from(select.options).map((o) => o.value)).toEqual(['user', 'admin']);

    select.value = 'admin';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(fixture.componentInstance.value).toBe('admin');
  });
});

describe('ui-badge', () => {
  @Component({
    imports: [BadgeComponent],
    template: `<ui-badge [tone]="'danger'">Revoked</ui-badge>`,
  })
  class Host {}

  it('test_colour_is_never_the_only_signal', () => {
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
    const badge = el<HTMLElement>(fixture, '.badge');

    expect(badge.textContent?.trim()).toBe('Revoked');
    expect(badge.getAttribute('data-tone')).toBe('danger');
  });
});

describe('ui-empty-state', () => {
  @Component({
    imports: [EmptyStateComponent],
    template: `<ui-empty-state [title]="'No users yet'" [message]="'Invite someone.'" />`,
  })
  class Host {}

  it('test_an_empty_state_explains_itself', () => {
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('No users yet');
    expect(fixture.nativeElement.textContent).toContain('Invite someone.');
  });
});

describe('ui-table', () => {
  @Component({
    imports: [TableComponent],
    template: `
      <ui-table
        [loading]="loading()"
        [error]="error()"
        [empty]="empty()"
        [density]="density()"
        (retried)="retries = retries + 1"
      >
        <tr head>
          <th>Email</th>
        </tr>
        <tr body>
          <td>someone@example.com</td>
        </tr>
      </ui-table>
    `,
  })
  class Host {
    readonly loading = signal(false);
    readonly error = signal('');
    readonly empty = signal(false);
    readonly density = signal<'comfortable' | 'compact'>('comfortable');
    retries = 0;
  }

  let fixture: ComponentFixture<Host>;

  beforeEach(() => {
    fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
  });

  it('test_rows_are_shown_when_there_is_nothing_wrong', () => {
    expect(fixture.nativeElement.querySelector('table')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('.table-skeleton')).toBeNull();
    expect(fixture.nativeElement.textContent).toContain('someone@example.com');
  });

  it('test_loading_shows_a_skeleton_rather_than_an_empty_table', () => {
    fixture.componentInstance.loading.set(true);
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.table-skeleton')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('table')).toBeNull();
  });

  it('test_a_failed_request_shows_an_error_not_a_blank_table', () => {
    fixture.componentInstance.error.set('Could not reach the server.');
    fixture.detectChanges();
    const alert = el<HTMLElement>(fixture, '[role="alert"]');

    expect(alert.textContent).toContain('Could not reach the server.');
    expect(fixture.nativeElement.querySelector('table')).toBeNull();

    el<HTMLButtonElement>(fixture, '.table-retry button').click();
    expect(fixture.componentInstance.retries).toBe(1);
  });

  it('test_an_error_outranks_a_loading_state', () => {
    fixture.componentInstance.loading.set(true);
    fixture.componentInstance.error.set('Could not reach the server.');
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('[role="alert"]')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('.table-skeleton')).toBeNull();
  });

  // The states are separate blocks in the template, so the projected rows are
  // destroyed and re-created as the table moves between them. This is the
  // test that the rows genuinely come back rather than being projected once
  // and lost on the first refresh.
  it('test_rows_come_back_after_a_refresh', () => {
    fixture.componentInstance.loading.set(true);
    fixture.detectChanges();
    fixture.componentInstance.loading.set(false);
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('table')).not.toBeNull();
    expect(fixture.nativeElement.textContent).toContain('someone@example.com');
  });

  it('test_compact_density_is_a_modifier_not_a_second_table', () => {
    fixture.componentInstance.density.set('compact');
    fixture.detectChanges();

    expect(el<HTMLElement>(fixture, 'table').className).toContain('density-compact');
  });
});

describe('ui-modal', () => {
  @Component({
    imports: [ModalComponent],
    template: `
      <button id="opener" type="button" (click)="open.set(true)">Open</button>
      <ui-modal [open]="open()" [title]="'Edit user'" (closed)="open.set(false)">
        <input id="first" />
        <input id="last" />
      </ui-modal>
    `,
  })
  class Host {
    readonly open = signal(false);
  }

  let fixture: ComponentFixture<Host>;

  beforeEach(() => {
    fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
  });

  function openIt(): void {
    el<HTMLButtonElement>(fixture, '#opener').focus();
    el<HTMLButtonElement>(fixture, '#opener').click();
    fixture.detectChanges();
  }

  it('test_a_modal_is_a_labelled_dialog', () => {
    openIt();
    const dialog = el<HTMLElement>(fixture, '[role="dialog"]');

    expect(dialog.getAttribute('aria-modal')).toBe('true');
    const labelledBy = dialog.getAttribute('aria-labelledby');
    expect(labelledBy).toBeTruthy();
    expect(dialog.querySelector(`#${labelledBy}`)?.textContent).toContain('Edit user');
  });

  it('test_a_modal_moves_focus_into_itself_when_it_opens', fakeAsync(() => {
    openIt();
    tick();

    expect(el<HTMLElement>(fixture, '[role="dialog"]').contains(document.activeElement)).toBeTrue();
  }));

  it('test_tab_is_trapped_inside_the_modal', fakeAsync(() => {
    openIt();
    tick();
    const dialog = el<HTMLElement>(fixture, '[role="dialog"]');
    const focusable = Array.from(
      dialog.querySelectorAll<HTMLElement>('button, input'),
    );
    const last = focusable[focusable.length - 1];

    last.focus();
    dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
    fixture.detectChanges();

    expect(document.activeElement).toBe(focusable[0]);
  }));

  it('test_escape_closes_the_modal', () => {
    openIt();
    el<HTMLElement>(fixture, '[role="dialog"]').dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    fixture.detectChanges();

    expect(fixture.componentInstance.open()).toBeFalse();
    expect(fixture.nativeElement.querySelector('[role="dialog"]')).toBeNull();
  });

  it('test_focus_returns_to_the_opener_when_the_modal_closes', fakeAsync(() => {
    const opener = el<HTMLButtonElement>(fixture, '#opener');
    openIt();
    tick();
    fixture.componentInstance.open.set(false);
    fixture.detectChanges();
    tick();

    expect(document.activeElement).toBe(opener);
  }));
});

describe('ui-confirm-dialog', () => {
  @Component({
    imports: [ConfirmDialogComponent],
    template: `
      <ui-confirm-dialog
        [open]="true"
        [title]="'Delete user'"
        [message]="'This cannot be undone.'"
        [consequences]="['Their API keys are revoked.']"
        [confirmWord]="'victim@example.com'"
        (confirmed)="confirmations = confirmations + 1"
      />
    `,
  })
  class Host {
    confirmations = 0;
  }

  let fixture: ComponentFixture<Host>;

  beforeEach(() => {
    fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
  });

  it('test_delete_requires_confirmation', () => {
    const confirm = el<HTMLButtonElement>(fixture, '.confirm-action');
    expect(confirm.disabled).toBeTrue();

    confirm.click();
    expect(fixture.componentInstance.confirmations).toBe(0);

    const input = el<HTMLInputElement>(fixture, '.confirm-word input');
    input.value = 'victim@example.com';
    input.dispatchEvent(new Event('input'));
    fixture.detectChanges();

    expect(el<HTMLButtonElement>(fixture, '.confirm-action').disabled).toBeFalse();
    el<HTMLButtonElement>(fixture, '.confirm-action').click();
    expect(fixture.componentInstance.confirmations).toBe(1);
  });

  it('test_the_dialog_says_what_else_it_removes', () => {
    expect(fixture.nativeElement.textContent).toContain('Their API keys are revoked.');
    expect(fixture.nativeElement.textContent).toContain('This cannot be undone.');
  });
});

describe('toasts', () => {
  @Component({
    imports: [ToastsComponent],
    template: `<ui-toasts />`,
  })
  class Host {}

  let fixture: ComponentFixture<Host>;
  let toasts: ToastService;

  beforeEach(() => {
    fixture = TestBed.createComponent(Host);
    toasts = TestBed.inject(ToastService);
    fixture.detectChanges();
  });

  it('test_an_error_toast_is_announced_assertively', () => {
    toasts.error('Could not save.');
    fixture.detectChanges();
    const toast = el<HTMLElement>(fixture, '.toast');

    expect(toast.textContent).toContain('Could not save.');
    expect(el<HTMLElement>(fixture, '.toast-stack').getAttribute('aria-live')).toBe('assertive');
  });

  it('test_a_toast_can_be_dismissed', () => {
    toasts.success('Saved.');
    fixture.detectChanges();
    el<HTMLButtonElement>(fixture, '.toast button').click();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.toast')).toBeNull();
  });
});

describe('ThemeService', () => {
  let theme: ThemeService;

  beforeEach(() => {
    localStorage.removeItem('rag_theme');
    document.documentElement.removeAttribute('data-theme');
    theme = TestBed.inject(ThemeService);
  });

  afterEach(() => {
    localStorage.removeItem('rag_theme');
    document.documentElement.removeAttribute('data-theme');
  });

  it('test_setting_a_theme_marks_the_document_and_is_remembered', () => {
    theme.set('dark');

    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(localStorage.getItem('rag_theme')).toBe('dark');
    expect(theme.theme()).toBe('dark');
  });

  it('test_following_the_system_leaves_no_override_on_the_document', () => {
    theme.set('dark');
    theme.set('system');

    expect(document.documentElement.hasAttribute('data-theme')).toBeFalse();
    expect(theme.theme()).toBe('system');
  });
});
