import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';

import { AccessService } from '../../../services/access.service';
import { AdminService } from '../../../services/admin.service';
import { describeError } from '../../../services/http-error';
import { Group, User } from '../../../models';
import {
  ButtonComponent,
  ConfirmDialogComponent,
  EmptyStateComponent,
  InputComponent,
  ModalComponent,
  SelectComponent,
  TableComponent,
  ToastService,
} from '../../../ui';

/**
 * Groups: create, rename, delete, and edit membership in place.
 *
 * Membership is edited on the row rather than behind a dialog because adding
 * three people to a group is the most common thing anyone does here, and a
 * dialog per person turns that into nine clicks.
 *
 * The member picker only offers people who are not already in the group. An
 * "add" that silently does nothing because they were already a member reads
 * as a broken button.
 */
@Component({
  selector: 'app-console-groups',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
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
        <h1>Groups</h1>
        <p>Groups hold people. Grants give a group access to a collection.</p>
      </div>
      <span class="groups-new">
        <ui-button [variant]="'primary'" (clicked)="openCreate()">New group</ui-button>
      </span>
    </section>

    @if (actionError()) {
      <p class="alert alert-error action-error" role="alert">{{ actionError() }}</p>
    }

    <ui-table
      [loading]="loading()"
      [error]="error()"
      [empty]="groups().length === 0"
      (retried)="load()"
    >
      <tr head>
        <th scope="col">Group</th>
        <th scope="col">Members</th>
        <th scope="col">Add a member</th>
        <th scope="col"><span class="sr-only">Actions</span></th>
      </tr>
      <ng-container body>
        @for (group of groups(); track group.id) {
          <tr class="group-row">
            <td>
              <strong>{{ group.name }}</strong>
              @if (group.description) {
                <p class="muted">{{ group.description }}</p>
              }
            </td>
            <td>
              @if (membersOf(group.id).length === 0) {
                <span class="muted">No members</span>
              }
              <ul class="member-list">
                @for (member of membersOf(group.id); track member.id) {
                  <li>
                    <span>{{ member.email }}</span>
                    <span class="member-remove">
                      <ui-button [size]="'sm'" (clicked)="removeMember(group, member)">
                        Remove
                      </ui-button>
                    </span>
                  </li>
                }
              </ul>
            </td>
            <td class="member-add">
              <ui-select
                [label]="'Person'"
                [options]="candidatesFor(group.id)"
                [value]="choice()[group.id] ?? ''"
                (valueChange)="choose(group.id, $event)"
              />
              <button
                type="button"
                class="btn btn-secondary btn-sm member-add-submit"
                [disabled]="!choice()[group.id]"
                (click)="addMember(group)"
              >
                Add
              </button>
            </td>
            <td class="row-actions">
              <span class="group-rename">
                <ui-button [size]="'sm'" (clicked)="openEdit(group)">Rename</ui-button>
              </span>
              <span class="group-delete">
                <ui-button [size]="'sm'" [variant]="'danger'" (clicked)="askDelete(group)">
                  Delete
                </ui-button>
              </span>
            </td>
          </tr>
        }
      </ng-container>
      <div table-empty>
        <ui-empty-state
          [title]="'No groups yet'"
          [message]="'Create a group, then grant it access to a collection.'"
        />
      </div>
    </ui-table>

    <ui-modal [open]="createOpen()" [title]="'New group'" (closed)="createOpen.set(false)">
      <div class="group-create stack gap">
        <ui-input [label]="'Name'" [required]="true" [(value)]="newName" />
        <ui-input [label]="'Description'" [(value)]="newDescription" />
        @if (formError()) {
          <p class="field-error" role="alert">{{ formError() }}</p>
        }
      </div>
      <div modal-actions class="row">
        <button type="button" class="btn btn-secondary" (click)="createOpen.set(false)">
          Cancel
        </button>
        <button type="button" class="btn btn-primary group-create-submit" (click)="create()">
          Create group
        </button>
      </div>
    </ui-modal>

    <ui-modal [open]="editing() !== null" [title]="'Rename group'" (closed)="editing.set(null)">
      <div class="group-edit stack gap">
        <ui-input [label]="'Name'" [required]="true" [(value)]="editName" />
        <ui-input [label]="'Description'" [(value)]="editDescription" />
        @if (formError()) {
          <p class="field-error" role="alert">{{ formError() }}</p>
        }
      </div>
      <div modal-actions class="row">
        <button type="button" class="btn btn-secondary" (click)="editing.set(null)">Cancel</button>
        <button type="button" class="btn btn-primary group-edit-submit" (click)="saveEdit()">
          Save
        </button>
      </div>
    </ui-modal>

    <ui-confirm-dialog
      [open]="pendingDelete() !== null"
      [title]="'Delete group'"
      [message]="'Deleting a group cannot be undone.'"
      [consequences]="deleteConsequences()"
      [confirmWord]="pendingDelete()?.name ?? ''"
      [confirmLabel]="'Delete group'"
      (cancelled)="pendingDelete.set(null)"
      (confirmed)="confirmDelete()"
    />
  `,
})
export class GroupsComponent implements OnInit {
  private readonly access = inject(AccessService);
  private readonly admin = inject(AdminService);
  private readonly toasts = inject(ToastService);

  readonly groups = signal<Group[]>([]);
  readonly users = signal<User[]>([]);
  readonly members = signal<Record<number, User[]>>({});
  readonly choice = signal<Record<number, string>>({});

  readonly loading = signal(false);
  readonly error = signal('');
  readonly actionError = signal('');
  readonly formError = signal('');

  readonly createOpen = signal(false);
  readonly newName = signal('');
  readonly newDescription = signal('');

  readonly editing = signal<Group | null>(null);
  readonly editName = signal('');
  readonly editDescription = signal('');

  readonly pendingDelete = signal<Group | null>(null);

  readonly deleteConsequences = computed(() => [
    'Every grant this group holds is removed, so access through it ends immediately.',
    'Its members lose that access but their accounts are untouched.',
  ]);

  ngOnInit(): void {
    this.load();
    this.admin.listUsers().subscribe({ next: (users) => this.users.set(users) });
  }

  load(): void {
    this.loading.set(true);
    this.error.set('');
    this.access.listGroups().subscribe({
      next: (groups) => {
        this.loading.set(false);
        this.groups.set(groups);
        groups.forEach((group) => this.loadMembers(group.id));
      },
      error: (err) => {
        this.loading.set(false);
        this.error.set(describeError(err, 'Could not load groups.'));
      },
    });
  }

  membersOf(groupId: number): User[] {
    return this.members()[groupId] ?? [];
  }

  candidatesFor(groupId: number): { value: string; label: string }[] {
    const taken = new Set(this.membersOf(groupId).map((m) => m.id));
    const options = this.users()
      .filter((user) => !taken.has(user.id))
      .map((user) => ({ value: String(user.id), label: user.email }));
    return [{ value: '', label: 'Choose a person' }, ...options];
  }

  choose(groupId: number, value: string): void {
    this.choice.update((current) => ({ ...current, [groupId]: value }));
  }

  addMember(group: Group): void {
    const chosen = this.choice()[group.id];
    if (!chosen) {
      return;
    }
    this.actionError.set('');
    this.access.addMember(group.id, Number(chosen)).subscribe({
      next: () => {
        this.choose(group.id, '');
        this.loadMembers(group.id);
      },
      error: (err) => this.reportAction(err, 'Could not add that member.'),
    });
  }

  removeMember(group: Group, member: User): void {
    this.actionError.set('');
    this.access.removeMember(group.id, member.id).subscribe({
      next: () => this.loadMembers(group.id),
      error: (err) => this.reportAction(err, 'Could not remove that member.'),
    });
  }

  openCreate(): void {
    this.formError.set('');
    this.newName.set('');
    this.newDescription.set('');
    this.createOpen.set(true);
  }

  create(): void {
    if (this.newName().trim() === '') {
      this.formError.set('A group needs a name.');
      return;
    }
    this.access.createGroup(this.newName().trim(), this.newDescription()).subscribe({
      next: (group) => {
        this.createOpen.set(false);
        this.toasts.success(`Created ${group.name}.`);
        this.load();
      },
      error: (err) => this.formError.set(describeError(err, 'Could not create that group.')),
    });
  }

  openEdit(group: Group): void {
    this.formError.set('');
    this.editName.set(group.name);
    this.editDescription.set(group.description);
    this.editing.set(group);
  }

  saveEdit(): void {
    const group = this.editing();
    if (group === null) {
      return;
    }
    this.access
      .updateGroup(group.id, { name: this.editName().trim(), description: this.editDescription() })
      .subscribe({
        next: () => {
          this.editing.set(null);
          this.toasts.success('Group updated.');
          this.load();
        },
        error: (err) => this.formError.set(describeError(err, 'Could not rename that group.')),
      });
  }

  askDelete(group: Group): void {
    this.actionError.set('');
    this.pendingDelete.set(group);
  }

  confirmDelete(): void {
    const group = this.pendingDelete();
    if (group === null) {
      return;
    }
    this.pendingDelete.set(null);
    this.access.deleteGroup(group.id).subscribe({
      next: () => {
        this.toasts.success(`Deleted ${group.name}.`);
        this.load();
      },
      error: (err) => this.reportAction(err, 'Could not delete that group.'),
    });
  }

  private loadMembers(groupId: number): void {
    this.access.listMembers(groupId).subscribe({
      next: (users) => this.members.update((current) => ({ ...current, [groupId]: users })),
      error: (err) => this.reportAction(err, 'Could not load that group’s members.'),
    });
  }

  private reportAction(err: unknown, fallback: string): void {
    const message = describeError(err, fallback);
    this.actionError.set(message);
    this.toasts.error(message);
  }
}
