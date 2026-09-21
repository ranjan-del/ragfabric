import { DatePipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';

import { AdminService } from '../../../services/admin.service';
import { describeError } from '../../../services/http-error';
import { User } from '../../../models';
import {
  BadgeComponent,
  ButtonComponent,
  ConfirmDialogComponent,
  EmptyStateComponent,
  InputComponent,
  ModalComponent,
  SelectComponent,
  TableComponent,
  ToastService,
} from '../../../ui';

const PAGE_SIZE = 10;

/**
 * Users: create, change permissions, delete.
 *
 * Two behaviours here are deliberate rather than incidental. A failed load
 * renders as an error with the server's own message, never as an empty table,
 * because those two look identical and mean opposite things. And a delete
 * requires the operator to type the account's email, with the dialog spelling
 * out that the account's API keys stop working, because deleting a person is
 * not undoable and the surprising part is never the account itself.
 */
@Component({
  selector: 'app-console-users',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    DatePipe,
    BadgeComponent,
    ButtonComponent,
    ConfirmDialogComponent,
    EmptyStateComponent,
    InputComponent,
    ModalComponent,
    SelectComponent,
    TableComponent,
  ],
  template: `
    <section class="page-head spread">
      <div>
        <h1>Users</h1>
        <p>Create accounts, change roles, and remove people who have left.</p>
      </div>
      <span class="users-new">
        <ui-button [variant]="'primary'" (clicked)="openCreate()">New user</ui-button>
      </span>
    </section>

    @if (actionError()) {
      <p class="alert alert-error action-error" role="alert">{{ actionError() }}</p>
    }

    <div class="user-search toolbar">
      <ui-input [label]="'Search'" [placeholder]="'Filter by email'" [(value)]="search" />
    </div>

    <ui-table
      [loading]="loading()"
      [error]="error()"
      [empty]="filtered().length === 0"
      [density]="'comfortable'"
      (retried)="load()"
    >
      <tr head>
        <th scope="col">Email</th>
        <th scope="col">Role</th>
        <th scope="col">Status</th>
        <th scope="col">Created</th>
        <th scope="col"><span class="sr-only">Actions</span></th>
      </tr>
      <ng-container body>
        @for (user of visible(); track user.id) {
        <tr class="user-row">
          <td>{{ user.email }}</td>
          <td>
            <ui-badge [tone]="user.role === 'admin' ? 'brand' : 'neutral'">{{ user.role }}</ui-badge>
          </td>
          <td>
            <ui-badge [tone]="user.is_active ? 'success' : 'warning'">
              {{ user.is_active ? 'active' : 'disabled' }}
            </ui-badge>
          </td>
          <td>{{ user.created_at | date: 'mediumDate' }}</td>
          <td class="row-actions">
            <span class="user-edit">
              <ui-button [size]="'sm'" (clicked)="openEdit(user)">Permissions</ui-button>
            </span>
            <span class="user-delete">
              <ui-button [size]="'sm'" [variant]="'danger'" (clicked)="askDelete(user)">
                Delete
              </ui-button>
            </span>
          </td>
        </tr>
        }
      </ng-container>
      <div table-empty>
        <ui-empty-state
          [title]="search() ? 'No users match that search' : 'No users yet'"
          [message]="'Create an account to give someone access.'"
        />
      </div>
    </ui-table>

    @if (pageCount() > 1) {
      <nav class="pager" aria-label="Pagination">
        <ui-button [size]="'sm'" [disabled]="page() === 1" (clicked)="page.set(page() - 1)">
          Previous
        </ui-button>
        <span class="muted">Page {{ page() }} of {{ pageCount() }}</span>
        <ui-button
          [size]="'sm'"
          [disabled]="page() === pageCount()"
          (clicked)="page.set(page() + 1)"
        >
          Next
        </ui-button>
      </nav>
    }

    <ui-modal [open]="createOpen()" [title]="'New user'" (closed)="createOpen.set(false)">
      <div class="user-create stack gap">
        <ui-input [label]="'Email'" [type]="'email'" [required]="true" [(value)]="newEmail" />
        <ui-input
          [label]="'Password'"
          [type]="'password'"
          [required]="true"
          [hint]="'At least six characters.'"
          [(value)]="newPassword"
        />
        <ui-select
          [label]="'Role'"
          [options]="[
            { value: 'user', label: 'User' },
            { value: 'admin', label: 'Admin' },
          ]"
          [(value)]="newRole"
        />
        @if (formError()) {
          <p class="field-error" role="alert">{{ formError() }}</p>
        }
      </div>
      <div modal-actions class="row">
        <button type="button" class="btn btn-secondary" (click)="createOpen.set(false)">
          Cancel
        </button>
        <button
          type="button"
          class="btn btn-primary user-create-submit"
          [disabled]="saving()"
          (click)="create()"
        >
          Create user
        </button>
      </div>
    </ui-modal>

    <ui-modal [open]="editing() !== null" [title]="'Permissions'" (closed)="editing.set(null)">
      <div class="user-edit-form stack gap">
        <p class="muted">{{ editing()?.email }}</p>
        <ui-select
          [label]="'Role'"
          [options]="[
            { value: 'user', label: 'User' },
            { value: 'admin', label: 'Admin' },
          ]"
          [(value)]="editRole"
        />
        <ui-select
          [label]="'Status'"
          [options]="[
            { value: 'active', label: 'Active' },
            { value: 'disabled', label: 'Disabled' },
          ]"
          [(value)]="editStatus"
        />
        @if (formError()) {
          <p class="field-error" role="alert">{{ formError() }}</p>
        }
      </div>
      <div modal-actions class="row">
        <button type="button" class="btn btn-secondary" (click)="editing.set(null)">Cancel</button>
        <button
          type="button"
          class="btn btn-primary user-edit-submit"
          [disabled]="saving()"
          (click)="savePermissions()"
        >
          Save
        </button>
      </div>
    </ui-modal>

    <ui-confirm-dialog
      [open]="pendingDelete() !== null"
      [title]="'Delete user'"
      [message]="'Deleting an account cannot be undone.'"
      [consequences]="deleteConsequences()"
      [confirmWord]="pendingDelete()?.email ?? ''"
      [confirmLabel]="'Delete user'"
      (cancelled)="pendingDelete.set(null)"
      (confirmed)="confirmDelete()"
    />
  `,
})
export class UsersComponent implements OnInit {
  private readonly admin = inject(AdminService);
  private readonly toasts = inject(ToastService);

  readonly users = signal<User[]>([]);
  readonly loading = signal(false);
  readonly error = signal('');
  readonly actionError = signal('');
  readonly formError = signal('');
  readonly saving = signal(false);
  readonly search = signal('');
  readonly page = signal(1);

  readonly createOpen = signal(false);
  readonly newEmail = signal('');
  readonly newPassword = signal('');
  readonly newRole = signal('user');

  readonly editing = signal<User | null>(null);
  readonly editRole = signal('user');
  readonly editStatus = signal('active');

  readonly pendingDelete = signal<User | null>(null);

  readonly filtered = computed(() => {
    const needle = this.search().trim().toLowerCase();
    const users = this.users();
    return needle === '' ? users : users.filter((u) => u.email.toLowerCase().includes(needle));
  });

  readonly pageCount = computed(() => Math.max(1, Math.ceil(this.filtered().length / PAGE_SIZE)));

  readonly visible = computed(() => {
    const page = Math.min(this.page(), this.pageCount());
    const start = (page - 1) * PAGE_SIZE;
    return this.filtered().slice(start, start + PAGE_SIZE);
  });

  readonly deleteConsequences = computed(() => [
    'Their API keys are revoked immediately and stop authenticating.',
    'Their group memberships are removed, so any access granted through a group ends.',
    'Their collections and documents are kept, with the owner cleared.',
    'Their query and audit history is kept, with the account reference cleared.',
  ]);

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.error.set('');
    this.admin.listUsers().subscribe({
      next: (users) => {
        this.users.set(users);
        this.loading.set(false);
      },
      error: (err) => {
        this.loading.set(false);
        this.error.set(describeError(err, 'Could not load users.'));
      },
    });
  }

  openCreate(): void {
    this.formError.set('');
    this.newEmail.set('');
    this.newPassword.set('');
    this.newRole.set('user');
    this.createOpen.set(true);
  }

  create(): void {
    if (this.newEmail().trim() === '' || this.newPassword().length < 6) {
      this.formError.set('An email and a password of at least six characters are required.');
      return;
    }
    this.saving.set(true);
    this.admin
      .createUser({
        email: this.newEmail().trim(),
        password: this.newPassword(),
        role: this.newRole(),
        is_active: true,
      })
      .subscribe({
        next: (user) => {
          this.saving.set(false);
          this.createOpen.set(false);
          this.toasts.success(`Created ${user.email}.`);
          this.load();
        },
        error: (err) => {
          this.saving.set(false);
          this.formError.set(describeError(err, 'Could not create that user.'));
        },
      });
  }

  openEdit(user: User): void {
    this.formError.set('');
    this.editRole.set(user.role);
    this.editStatus.set(user.is_active ? 'active' : 'disabled');
    this.editing.set(user);
  }

  savePermissions(): void {
    const user = this.editing();
    if (user === null) {
      return;
    }
    this.saving.set(true);
    this.admin
      .setPermissions(user.id, {
        role: this.editRole(),
        is_active: this.editStatus() === 'active',
      })
      .subscribe({
        next: () => {
          this.saving.set(false);
          this.editing.set(null);
          this.toasts.success(`Updated ${user.email}.`);
          this.load();
        },
        error: (err) => {
          this.saving.set(false);
          this.formError.set(describeError(err, 'Could not save those permissions.'));
        },
      });
  }

  askDelete(user: User): void {
    this.actionError.set('');
    this.pendingDelete.set(user);
  }

  confirmDelete(): void {
    const user = this.pendingDelete();
    if (user === null) {
      return;
    }
    this.pendingDelete.set(null);
    this.admin.deleteUser(user.id).subscribe({
      next: () => {
        this.toasts.success(`Deleted ${user.email}.`);
        this.load();
      },
      error: (err) => {
        // The row stays exactly where it was: nothing was removed, so showing
        // it as removed would be a lie the next refresh would contradict.
        const message = describeError(err, 'Could not delete that user.');
        this.actionError.set(message);
        this.toasts.error(message);
      },
    });
  }
}
