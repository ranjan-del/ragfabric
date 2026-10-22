import { DatePipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';

import { AccessService } from '../../../services/access.service';
import { AdminService } from '../../../services/admin.service';
import { CollectionService } from '../../../services/collection.service';
import { describeError } from '../../../services/http-error';
import { ApiKeyItem, Collection, User } from '../../../models';
import { parseRateLimit } from './rate-limit';
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

/**
 * API keys.
 *
 * The plaintext key exists in this component and nowhere else, for the life
 * of one panel. Everything below follows from that.
 *
 * It is shown once because it is stored hashed: there is nothing to show
 * later, and a "reveal" button would teach operators to expect one. The panel
 * cannot be dismissed by the backdrop, by Escape or by an X, only by ticking
 * "I have saved this key", because a panel that closes by accident loses a
 * secret that cannot be recovered.
 *
 * The secret is never written to localStorage or sessionStorage (they outlive
 * the page and are readable by anything else served from this origin), never
 * put in a URL (it would land in history and in every access log along the
 * way), and never logged. Acknowledging the panel clears it from the
 * component, which removes it from the DOM.
 */
@Component({
  selector: 'app-console-keys',
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
  styles: `
    .key-scopes { min-width: 8rem; max-width: 12rem; white-space: normal; }
    .key-rate { white-space: nowrap; }
  `,
  template: `
    <section class="page-head spread">
      <div>
        <h1>API keys</h1>
        <p>A key is a principal. It carries its own scopes and its own rate limit.</p>
      </div>
      <span class="keys-new">
        <ui-button [variant]="'primary'" (clicked)="openCreate()">New key</ui-button>
      </span>
    </section>

    @if (actionError()) {
      <p class="alert alert-error action-error" role="alert">{{ actionError() }}</p>
    }

    <ui-table
      [loading]="loading()"
      [error]="error()"
      [empty]="keys().length === 0"
      (retried)="load()"
    >
      <tr head>
        <th scope="col">Name</th>
        <th scope="col">Prefix</th>
        <th scope="col">Scopes</th>
        <th scope="col" title="Requests per minute">Rate limit</th>
        <th scope="col">Created</th>
        <th scope="col">Last used</th>
        <th scope="col">Expires</th>
        <th scope="col">Status</th>
        <th scope="col"><span class="sr-only">Actions</span></th>
      </tr>
      <ng-container body>
        @for (key of keys(); track key.id) {
          <tr class="key-row">
            <td>{{ key.name }}</td>
            <td><code>{{ key.key_prefix }}</code></td>
            <td class="key-scopes">
              <span class="muted">{{ scopeSummary(key) }}</span>
            </td>
            <td class="key-rate">{{ key.rate_limit_per_minute.toLocaleString('en-GB') }} / min</td>
            <td>{{ key.created_at | date: 'mediumDate' }}</td>
            <td>
              {{ key.last_used_at ? (key.last_used_at | date: 'medium') : 'Never' }}
            </td>
            <td>{{ key.expires_at ? (key.expires_at | date: 'mediumDate') : 'No expiry' }}</td>
            <td>
              <ui-badge [tone]="key.is_active ? 'success' : 'danger'">
                {{ key.is_active ? 'active' : 'revoked' }}
              </ui-badge>
            </td>
            <td class="row-actions">
              @if (key.is_active) {
                <span class="key-revoke">
                  <ui-button [size]="'sm'" [variant]="'danger'" (clicked)="askRevoke(key)">
                    Revoke
                  </ui-button>
                </span>
              }
            </td>
          </tr>
        }
      </ng-container>
      <div table-empty>
        <ui-empty-state
          [title]="'No API keys yet'"
          [message]="'Create one to let a service call the API without a password.'"
        />
      </div>
    </ui-table>

    <ui-modal [open]="createOpen()" [title]="'New API key'" (closed)="createOpen.set(false)">
      <div class="key-create stack gap">
        <ui-input [label]="'Name'" [required]="true" [(value)]="newName" />
        <ui-select
          [label]="'Acts as'"
          [hint]="'The key inherits this account\\'s access.'"
          [options]="userOptions()"
          [value]="newUser()"
          (valueChange)="newUser.set($event)"
        />
        <ui-select
          [label]="'Limit to a collection'"
          [options]="collectionOptions()"
          [value]="newCollection()"
          (valueChange)="newCollection.set($event)"
        />
        <ui-input
          class="key-rate-field"
          [label]="'Rate limit (requests per minute)'"
          [inputmode]="'numeric'"
          [hint]="rateLimitHint()"
          [error]="rateLimitError()"
          [(value)]="newRateLimit"
        />
        <p class="field-hint key-rate-note">
          A key's limit is set when it is created. The API cannot change it afterwards: to use a
          different limit, create a new key and revoke this one.
        </p>
        <ui-input
          [label]="'Expires on'"
          [type]="'date'"
          [hint]="'Leave empty for a key that does not expire.'"
          [(value)]="newExpiry"
        />
        @if (formError()) {
          <p class="field-error" role="alert">{{ formError() }}</p>
        }
      </div>
      <div modal-actions class="row">
        <button type="button" class="btn btn-secondary" (click)="createOpen.set(false)">
          Cancel
        </button>
        <button type="button" class="btn btn-primary key-create-submit" (click)="create()">
          Create key
        </button>
      </div>
    </ui-modal>

    @if (secret()) {
      <div class="modal-backdrop"></div>
      <div
        class="modal secret-panel"
        role="dialog"
        aria-modal="true"
        [attr.aria-labelledby]="secretTitleId"
      >
        <header class="modal-head">
          <h2 [id]="secretTitleId">Copy this key now</h2>
        </header>
        <div class="modal-body">
          <p>
            This is the only time this key is shown. It is stored hashed, so it cannot be shown
            again. If it is lost, revoke it and create another.
          </p>
          <p class="secret-value"><code>{{ secret() }}</code></p>
          <div class="row">
            <button type="button" class="btn btn-secondary key-copy" (click)="copy()">
              Copy to clipboard
            </button>
            @if (copied()) {
              <span class="muted" role="status">Copied</span>
            }
          </div>
          <label class="secret-ack">
            <input type="checkbox" [checked]="acknowledged()" (change)="onAcknowledge($event)" />
            I have saved this key somewhere safe
          </label>
        </div>
        <footer class="modal-foot">
          <button
            type="button"
            class="btn btn-primary secret-done"
            [disabled]="!acknowledged()"
            (click)="dismissSecret()"
          >
            Done
          </button>
        </footer>
      </div>
    }

    <ui-confirm-dialog
      [open]="pendingRevoke() !== null"
      [title]="'Revoke API key'"
      [message]="revokeMessage()"
      [consequences]="[
        'Every request using this key starts failing with 401 immediately.',
        'The key cannot be reactivated. Issue a new one instead.',
      ]"
      [confirmLabel]="'Revoke key'"
      (cancelled)="pendingRevoke.set(null)"
      (confirmed)="confirmRevoke()"
    />
  `,
})
export class KeysComponent implements OnInit {
  private readonly access = inject(AccessService);
  private readonly admin = inject(AdminService);
  private readonly collectionsApi = inject(CollectionService);
  private readonly toasts = inject(ToastService);

  readonly secretTitleId = 'api-key-secret-title';

  readonly keys = signal<ApiKeyItem[]>([]);
  readonly users = signal<User[]>([]);
  readonly collections = signal<Collection[]>([]);

  readonly loading = signal(false);
  readonly error = signal('');
  readonly actionError = signal('');
  readonly formError = signal('');

  readonly createOpen = signal(false);
  readonly newName = signal('');
  readonly newUser = signal('');
  readonly newCollection = signal('');
  readonly newExpiry = signal('');
  readonly newRateLimit = signal('');
  /** What the server gives a key that names no limit; null until it has said, or if it could not. */
  readonly defaultRateLimit = signal<number | null>(null);
  /** Shown once Create has been tried, so a half typed number is not scolded. */
  private readonly rateLimitChecked = signal(false);

  readonly rateLimitHint = computed(() => {
    const fallback = this.defaultRateLimit();
    return fallback === null
      ? 'Leave empty to use the server default.'
      : `Leave empty for the default of ${fallback.toLocaleString('en-GB')} per minute.`;
  });
  readonly rateLimitError = computed(() => {
    const parsed = parseRateLimit(this.newRateLimit());
    return this.rateLimitChecked() && !parsed.ok ? parsed.error : '';
  });

  /**
   * The plaintext key, held in memory only, for the life of the panel.
   * Cleared by dismissSecret(), which is what removes it from the DOM.
   */
  readonly secret = signal('');
  readonly acknowledged = signal(false);
  readonly copied = signal(false);

  readonly pendingRevoke = signal<ApiKeyItem | null>(null);

  readonly userOptions = computed(() => [
    { value: '', label: 'Choose an account' },
    ...this.users().map((user) => ({ value: String(user.id), label: user.email })),
  ]);

  readonly collectionOptions = computed(() => [
    { value: '', label: 'Every collection the account can reach' },
    ...this.collections().map((c) => ({ value: String(c.id), label: c.name })),
  ]);

  readonly revokeMessage = computed(() => {
    const key = this.pendingRevoke();
    return key === null ? '' : `Revoking ${key.name} (${key.key_prefix}) takes effect at once.`;
  });

  ngOnInit(): void {
    this.load();
    this.admin.listUsers().subscribe({ next: (users) => this.users.set(users) });
    this.collectionsApi.list().subscribe({ next: (list) => this.collections.set(list) });
    this.access.keyDefaults().subscribe({
      next: (defaults) => this.defaultRateLimit.set(defaults.rate_limit_per_minute),
      // An older server has no defaults endpoint: the hint then says "the server default", no number.
      error: () => this.defaultRateLimit.set(null),
    });
  }

  load(): void {
    this.loading.set(true);
    this.error.set('');
    this.access.listKeys().subscribe({
      next: (keys) => {
        this.loading.set(false);
        this.keys.set(keys);
      },
      error: (err) => {
        this.loading.set(false);
        this.error.set(describeError(err, 'Could not load API keys.'));
      },
    });
  }

  scopeSummary(key: ApiKeyItem): string {
    const collections =
      key.collection_ids.length === 0
        ? 'all collections'
        : key.collection_ids.map((id) => this.collectionName(id)).join(', ');
    const strategies = key.strategies.length === 0 ? 'all strategies' : key.strategies.join(', ');
    return `${collections} / ${strategies}`;
  }

  collectionName(id: number): string {
    return this.collections().find((c) => c.id === id)?.name ?? `collection ${id}`;
  }

  openCreate(): void {
    this.formError.set('');
    this.newName.set('');
    this.newUser.set('');
    this.newCollection.set('');
    this.newExpiry.set('');
    this.newRateLimit.set('');
    this.rateLimitChecked.set(false);
    this.createOpen.set(true);
  }

  create(): void {
    if (this.newName().trim() === '' || !this.newUser()) {
      this.formError.set('A key needs a name and an account to act as.');
      return;
    }
    this.rateLimitChecked.set(true);
    const limit = parseRateLimit(this.newRateLimit());
    if (!limit.ok) {
      return;
    }
    const collectionIds = this.newCollection() ? [Number(this.newCollection())] : [];
    this.access
      .createKey({
        name: this.newName().trim(),
        user_id: Number(this.newUser()),
        collection_ids: collectionIds,
        strategies: [],
        expires_at: this.newExpiry() ? `${this.newExpiry()}T00:00:00` : null,
        // Empty is left out so the server applies its own default.
        ...(limit.value === null ? {} : { rate_limit_per_minute: limit.value }),
      })
      .subscribe({
        next: (created) => {
          this.createOpen.set(false);
          this.acknowledged.set(false);
          this.copied.set(false);
          // Straight into the panel. Never stored, never logged, never
          // appended to the URL.
          this.secret.set(created.key);
        },
        error: (err) => this.formError.set(describeError(err, 'Could not create that key.')),
      });
  }

  copy(): void {
    const value = this.secret();
    if (!value) {
      return;
    }
    const clipboard = navigator.clipboard;
    if (clipboard && typeof clipboard.writeText === 'function') {
      clipboard.writeText(value).then(
        () => this.copied.set(true),
        () => this.copied.set(false),
      );
      return;
    }
    // No clipboard API (an insecure origin, typically). Say so rather than
    // reporting a copy that did not happen.
    this.copied.set(false);
    this.toasts.error('This browser would not let the page copy. Select the key and copy it.');
  }

  onAcknowledge(event: Event): void {
    this.acknowledged.set((event.target as HTMLInputElement).checked);
  }

  dismissSecret(): void {
    if (!this.acknowledged()) {
      return;
    }
    this.secret.set('');
    this.copied.set(false);
    this.acknowledged.set(false);
    this.toasts.success('Key created.');
    this.load();
  }

  askRevoke(key: ApiKeyItem): void {
    this.actionError.set('');
    this.pendingRevoke.set(key);
  }

  confirmRevoke(): void {
    const key = this.pendingRevoke();
    if (key === null) {
      return;
    }
    this.pendingRevoke.set(null);
    this.access.revokeKey(key.id).subscribe({
      next: () => {
        this.toasts.success(`Revoked ${key.name}.`);
        this.load();
      },
      error: (err) => {
        const message = describeError(err, 'Could not revoke that key.');
        this.actionError.set(message);
        this.toasts.error(message);
      },
    });
  }
}
