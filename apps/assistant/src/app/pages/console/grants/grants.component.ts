import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';

import { AccessService } from '../../../services/access.service';
import { AdminService } from '../../../services/admin.service';
import { CollectionService } from '../../../services/collection.service';
import { describeError } from '../../../services/http-error';
import { Collection, Grant, Group, User } from '../../../models';
import {
  BadgeComponent,
  ButtonComponent,
  ConfirmDialogComponent,
  EmptyStateComponent,
  SelectComponent,
  TableComponent,
  ToastService,
} from '../../../ui';

interface EffectiveRow {
  collection: string;
  permission: string;
  viaGroup: string;
}

/**
 * Grants: who can read what, in both directions, plus the answer.
 *
 * Listing grant rows is not enough. A grant joins a group to a collection,
 * and people reach it through membership, so "can Sam read the HR policies"
 * takes three tables to answer by hand. This screen answers it directly and
 * says which group supplies the access, because the follow up question is
 * always "why".
 *
 * Both directions are shown deliberately. By group answers "what does this
 * team have"; by collection answers "who can see this", which is the question
 * asked when something confidential has been uploaded to the wrong place.
 */
@Component({
  selector: 'app-console-grants',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    BadgeComponent,
    ButtonComponent,
    ConfirmDialogComponent,
    EmptyStateComponent,
    SelectComponent,
    TableComponent,
  ],
  template: `
    <section class="page-head">
      <h1>Grants</h1>
      <p>A grant gives a group read or write access to a collection.</p>
    </section>

    @if (actionError()) {
      <p class="alert alert-error action-error" role="alert">{{ actionError() }}</p>
    }

    <section class="card panel effective">
      <h2>Effective access</h2>
      <p class="muted">
        What one person can actually reach, resolved through the groups they belong to.
      </p>
      <ui-select
        [label]="'Person'"
        [options]="userOptions()"
        [value]="chosenUser()"
        (valueChange)="chosenUser.set($event)"
      />
      @if (chosenUser()) {
        @if (effective().length === 0) {
          <p class="muted">No access. This person is in no group that holds a grant.</p>
        } @else {
          <ul class="effective-list">
            @for (row of effective(); track row.collection + row.viaGroup) {
              <li class="effective-row">
                <strong>{{ row.collection }}</strong>
                <ui-badge [tone]="row.permission === 'write' ? 'warning' : 'success'">
                  {{ row.permission }}
                </ui-badge>
                <span class="muted">via {{ row.viaGroup }}</span>
              </li>
            }
          </ul>
        }
      }
    </section>

    <section class="card panel grant-create">
      <h2>Grant access</h2>
      <div class="row wrap">
        <ui-select
          [label]="'Group'"
          [options]="groupOptions()"
          [value]="newGroup()"
          (valueChange)="onFormChange('group', $event)"
        />
        <ui-select
          [label]="'Collection'"
          [options]="collectionOptions()"
          [value]="newCollection()"
          (valueChange)="onFormChange('collection', $event)"
        />
        <ui-select
          [label]="'Permission'"
          [options]="[
            { value: 'read', label: 'Read' },
            { value: 'write', label: 'Write' },
          ]"
          [value]="newPermission()"
          (valueChange)="onFormChange('permission', $event)"
        />
        <button type="button" class="btn btn-primary grant-create-submit" (click)="create()">
          Grant
        </button>
      </div>
      @if (grantError()) {
        <p class="field-error grant-error" role="alert">{{ grantError() }}</p>
      }
    </section>

    <nav class="view-toggle row" aria-label="Grant views">
      <button
        type="button"
        class="btn btn-sm view-group"
        [class.btn-primary]="view() === 'group'"
        (click)="view.set('group')"
      >
        By group
      </button>
      <button
        type="button"
        class="btn btn-sm view-collection"
        [class.btn-primary]="view() === 'collection'"
        (click)="view.set('collection')"
      >
        By collection
      </button>
    </nav>

    @if (view() === 'group') {
      <div class="by-group">
        <ui-table
          [loading]="loading()"
          [error]="error()"
          [empty]="groups().length === 0"
          (retried)="load()"
        >
          <tr head>
            <th scope="col">Group</th>
            <th scope="col">Collections</th>
          </tr>
          <ng-container body>
            @for (group of groups(); track group.id) {
              <tr class="group-grants">
                <td>
                  <strong>{{ group.name }}</strong>
                  <p class="muted">{{ memberCount(group.id) }} members</p>
                </td>
                <td>
                  @if (grantsForGroup(group.id).length === 0) {
                    <span class="muted">No access</span>
                  }
                  <ul class="grant-list">
                    @for (grant of grantsForGroup(group.id); track grant.id) {
                      <li>
                        <span>{{ collectionName(grant.collection_id) }}</span>
                        <ui-badge [tone]="grant.permission === 'write' ? 'warning' : 'success'">
                          {{ grant.permission }}
                        </ui-badge>
                        <span class="grant-revoke">
                          <ui-button [size]="'sm'" (clicked)="askRevoke(grant)">Revoke</ui-button>
                        </span>
                      </li>
                    }
                  </ul>
                </td>
              </tr>
            }
          </ng-container>
          <div table-empty>
            <ui-empty-state [title]="'No groups yet'" [message]="'Create a group first.'" />
          </div>
        </ui-table>
      </div>
    } @else {
      <div class="by-collection">
        <ui-table
          [loading]="loading()"
          [error]="error()"
          [empty]="collections().length === 0"
          (retried)="load()"
        >
          <tr head>
            <th scope="col">Collection</th>
            <th scope="col">Groups</th>
          </tr>
          <ng-container body>
            @for (collection of collections(); track collection.id) {
              <tr class="collection-grants">
                <td><strong>{{ collection.name }}</strong></td>
                <td>
                  @if (grantsForCollection(collection.id).length === 0) {
                    <span class="muted">Nobody, except an admin</span>
                  }
                  <ul class="grant-list">
                    @for (grant of grantsForCollection(collection.id); track grant.id) {
                      <li>
                        <span>{{ groupName(grant.group_id) }}</span>
                        <ui-badge [tone]="grant.permission === 'write' ? 'warning' : 'success'">
                          {{ grant.permission }}
                        </ui-badge>
                        <span class="grant-revoke">
                          <ui-button [size]="'sm'" (clicked)="askRevoke(grant)">Revoke</ui-button>
                        </span>
                      </li>
                    }
                  </ul>
                </td>
              </tr>
            }
          </ng-container>
          <div table-empty>
            <ui-empty-state [title]="'No collections yet'" [message]="'Create a collection first.'" />
          </div>
        </ui-table>
      </div>
    }

    <ui-confirm-dialog
      [open]="pendingRevoke() !== null"
      [title]="'Revoke access'"
      [message]="revokeMessage()"
      [consequences]="['Anyone who reached the collection only through this group loses it.']"
      [confirmLabel]="'Revoke'"
      (cancelled)="pendingRevoke.set(null)"
      (confirmed)="confirmRevoke()"
    />
  `,
})
export class GrantsComponent implements OnInit {
  private readonly access = inject(AccessService);
  private readonly admin = inject(AdminService);
  private readonly collectionsApi = inject(CollectionService);
  private readonly toasts = inject(ToastService);

  readonly groups = signal<Group[]>([]);
  readonly collections = signal<Collection[]>([]);
  readonly grants = signal<Grant[]>([]);
  readonly users = signal<User[]>([]);
  readonly members = signal<Record<number, User[]>>({});

  readonly loading = signal(false);
  readonly error = signal('');
  readonly actionError = signal('');
  readonly grantError = signal('');

  readonly view = signal<'group' | 'collection'>('group');
  readonly chosenUser = signal('');

  readonly newGroup = signal('');
  readonly newCollection = signal('');
  readonly newPermission = signal('read');

  readonly pendingRevoke = signal<Grant | null>(null);

  readonly userOptions = computed(() => [
    { value: '', label: 'Choose a person' },
    ...this.users().map((user) => ({ value: String(user.id), label: user.email })),
  ]);

  readonly groupOptions = computed(() => [
    { value: '', label: 'Choose a group' },
    ...this.groups().map((group) => ({ value: String(group.id), label: group.name })),
  ]);

  readonly collectionOptions = computed(() => [
    { value: '', label: 'Choose a collection' },
    ...this.collections().map((c) => ({ value: String(c.id), label: c.name })),
  ]);

  readonly effective = computed<EffectiveRow[]>(() => {
    const userId = Number(this.chosenUser());
    if (!userId) {
      return [];
    }
    const rows: EffectiveRow[] = [];
    for (const group of this.groups()) {
      const isMember = (this.members()[group.id] ?? []).some((m) => m.id === userId);
      if (!isMember) {
        continue;
      }
      for (const grant of this.grants().filter((g) => g.group_id === group.id)) {
        rows.push({
          collection: this.collectionName(grant.collection_id),
          permission: grant.permission,
          viaGroup: group.name,
        });
      }
    }
    return rows;
  });

  readonly revokeMessage = computed(() => {
    const grant = this.pendingRevoke();
    if (grant === null) {
      return '';
    }
    return `${this.groupName(grant.group_id)} will lose ${grant.permission} on ${this.collectionName(
      grant.collection_id,
    )}.`;
  });

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
        groups.forEach((group) =>
          this.access.listMembers(group.id).subscribe({
            next: (users) =>
              this.members.update((current) => ({ ...current, [group.id]: users })),
          }),
        );
      },
      error: (err) => {
        this.loading.set(false);
        this.error.set(describeError(err, 'Could not load groups.'));
      },
    });
    this.collectionsApi.list().subscribe({
      next: (collections) => this.collections.set(collections),
      error: (err) => this.error.set(describeError(err, 'Could not load collections.')),
    });
    this.access.listGrants().subscribe({
      next: (grants) => this.grants.set(grants),
      error: (err) => this.error.set(describeError(err, 'Could not load grants.')),
    });
  }

  memberCount(groupId: number): number {
    return (this.members()[groupId] ?? []).length;
  }

  grantsForGroup(groupId: number): Grant[] {
    return this.grants().filter((grant) => grant.group_id === groupId);
  }

  grantsForCollection(collectionId: number): Grant[] {
    return this.grants().filter((grant) => grant.collection_id === collectionId);
  }

  collectionName(id: number): string {
    return this.collections().find((c) => c.id === id)?.name ?? `collection ${id}`;
  }

  groupName(id: number): string {
    return this.groups().find((g) => g.id === id)?.name ?? `group ${id}`;
  }

  onFormChange(field: 'group' | 'collection' | 'permission', value: string): void {
    this.grantError.set('');
    if (field === 'group') {
      this.newGroup.set(value);
    } else if (field === 'collection') {
      this.newCollection.set(value);
    } else {
      this.newPermission.set(value);
    }
  }

  create(): void {
    const groupId = Number(this.newGroup());
    const collectionId = Number(this.newCollection());
    if (!groupId || !collectionId) {
      this.grantError.set('Choose a group and a collection.');
      return;
    }
    // The server upserts (group, collection), so posting a duplicate would
    // quietly succeed and look like a new grant. Saying so is more useful
    // than another identical row appearing.
    const existing = this.grants().find(
      (grant) => grant.group_id === groupId && grant.collection_id === collectionId,
    );
    if (existing) {
      this.grantError.set(
        `${this.groupName(groupId)} already has ${existing.permission} on ` +
          `${this.collectionName(collectionId)}. Revoke it first to change the permission.`,
      );
      return;
    }
    this.access
      .createGrant(groupId, collectionId, this.newPermission() as 'read' | 'write')
      .subscribe({
        next: () => {
          this.grantError.set('');
          this.toasts.success('Access granted.');
          this.refreshGrants();
        },
        error: (err) => this.grantError.set(describeError(err, 'Could not create that grant.')),
      });
  }

  askRevoke(grant: Grant): void {
    this.actionError.set('');
    this.pendingRevoke.set(grant);
  }

  confirmRevoke(): void {
    const grant = this.pendingRevoke();
    if (grant === null) {
      return;
    }
    this.pendingRevoke.set(null);
    this.access.deleteGrant(grant.id).subscribe({
      next: () => {
        this.toasts.success('Access revoked.');
        this.refreshGrants();
      },
      error: (err) => {
        const message = describeError(err, 'Could not revoke that grant.');
        this.actionError.set(message);
        this.toasts.error(message);
      },
    });
  }

  private refreshGrants(): void {
    this.access.listGrants().subscribe({
      next: (grants) => this.grants.set(grants),
      error: (err) => this.actionError.set(describeError(err, 'Could not reload grants.')),
    });
  }
}
