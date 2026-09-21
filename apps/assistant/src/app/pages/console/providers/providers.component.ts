import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';

import { ProviderUpdate, ProvidersService } from '../../../services/providers.service';
import { describeError } from '../../../services/http-error';
import { ProviderConfig, ProviderTestResult } from '../../../models';
import {
  ConfirmDialogComponent,
  InputComponent,
  SelectComponent,
  ToastService,
} from '../../../ui';

const LLM_PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'offline', label: 'Offline (no model calls)' },
];

const EMBEDDING_PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'offline', label: 'Offline (hashing embedder)' },
];

/**
 * Provider configuration.
 *
 * Two rules shape this screen.
 *
 * The secret is never here. The server reports which environment variable a
 * provider reads and whether it is set; this screen shows that and nothing
 * else. There is no field to paste a key into, because keys belong in the
 * deployment's environment, not in a config file written by a web form.
 *
 * Changing the embedding provider, model or dimension is treated as the most
 * destructive action in the console, because it is. ADR 0006 pins the
 * dimension of an index: every stored vector came from the model named here,
 * so changing it leaves an index of numbers that mean nothing until a
 * migration and a full re-index have run. The warning appears as soon as the
 * field changes, and saving it needs the word "reindex" typed out.
 */
@Component({
  selector: 'app-console-providers',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [ConfirmDialogComponent, InputComponent, SelectComponent],
  template: `
    <section class="page-head">
      <h1>Providers</h1>
      <p>Which model answers, which model embeds, and whether their keys are present.</p>
    </section>

    @if (error()) {
      <div class="card table-state table-error" role="alert">
        <p class="table-error-message">{{ error() }}</p>
        <p class="table-retry">
          <button type="button" class="btn btn-secondary btn-sm" (click)="load()">Try again</button>
        </p>
      </div>
    } @else if (loading()) {
      <div class="table-skeleton" aria-busy="true" aria-live="polite">
        <span class="sr-only">Loading provider configuration</span>
        <div class="skeleton-row"></div>
        <div class="skeleton-row"></div>
        <div class="skeleton-row"></div>
      </div>
    } @else if (config(); as current) {
      <section class="card panel">
        <h2>Answering model</h2>
        <p class="key-status llm-key-status" [attr.data-state]="keyState(current.llm.requires_key, current.llm.has_key)">
          @if (current.llm.requires_key) {
            {{ current.llm.key_env_var }} is {{ current.llm.has_key ? 'set' : 'not set' }}
            @if (!current.llm.has_key) {
              <span> in the server environment. Requests to this provider will fail.</span>
            }
          } @else {
            This provider needs no API key.
          }
        </p>
        <div class="row wrap">
          <span class="llm-provider">
            <ui-select
              [label]="'Provider'"
              [options]="llmProviders"
              [value]="llmProvider()"
              (valueChange)="llmProvider.set($event)"
            />
          </span>
          <span class="llm-model">
            <ui-input [label]="'Model'" [(value)]="llmModel" />
          </span>
          <span class="llm-base-url">
            <ui-input [label]="'Base URL'" [(value)]="llmBaseUrl" />
          </span>
          <button type="button" class="btn btn-secondary llm-test" (click)="test('llm')">
            Test connection
          </button>
        </div>
      </section>

      <section class="card panel">
        <h2>Embedding model</h2>
        <p
          class="key-status embeddings-key-status"
          [attr.data-state]="keyState(current.embeddings.requires_key, current.embeddings.has_key)"
        >
          @if (current.embeddings.requires_key) {
            {{ current.embeddings.key_env_var }} is
            {{ current.embeddings.has_key ? 'set' : 'not set' }}
          } @else {
            This provider needs no API key.
          }
        </p>
        <div class="row wrap">
          <span class="embeddings-provider">
            <ui-select
              [label]="'Provider'"
              [options]="embeddingProviders"
              [value]="embeddingsProvider()"
              (valueChange)="embeddingsProvider.set($event)"
            />
          </span>
          <span class="embeddings-model">
            <ui-input [label]="'Model'" [(value)]="embeddingsModel" />
          </span>
          <span class="embeddings-dim">
            <ui-input [label]="'Dimension'" [type]="'number'" [(value)]="embeddingsDim" />
          </span>
          <span class="embeddings-base-url">
            <ui-input [label]="'Base URL'" [(value)]="embeddingsBaseUrl" />
          </span>
          <button
            type="button"
            class="btn btn-secondary embeddings-test"
            (click)="test('embeddings')"
          >
            Test connection
          </button>
        </div>

        @if (embeddingsChanged()) {
          <div class="alert alert-error reindex-warning" role="alert">
            <strong>This invalidates every vector in the index.</strong>
            <p>
              The embedding dimension is pinned to the index (ADR 0006). Changing the provider,
              model or dimension means the stored vectors were produced by something else, so
              search will return nonsense until a migration and a full re-index have run. Nothing
              re-indexes itself: you have to run it.
            </p>
          </div>
        }
      </section>

      @if (testResult(); as result) {
        <p class="alert test-result" [attr.data-ok]="result.ok" role="status">
          @if (result.ok) {
            {{ result.target }} responded. Model: {{ result.model }}
          } @else {
            {{ result.target }} failed: {{ result.detail }}
          }
        </p>
      }

      @if (saveMessage()) {
        <p class="alert alert-info save-result" role="status">{{ saveMessage() }}</p>
      }

      @if (saveError()) {
        <p class="alert alert-error" role="alert">{{ saveError() }}</p>
      }

      <div class="row">
        <button
          type="button"
          class="btn btn-primary providers-save"
          [disabled]="!anythingChanged()"
          (click)="save()"
        >
          Save configuration
        </button>
        <button type="button" class="btn btn-secondary" [disabled]="!anythingChanged()" (click)="reset()">
          Discard changes
        </button>
      </div>
    }

    <ui-confirm-dialog
      [open]="confirmOpen()"
      [title]="'Change the embedding model'"
      [message]="
        'Every vector already stored was produced by the current embedding model. Changing it
         requires a migration and a full re-index before search works again.'
      "
      [consequences]="[
        'Existing vectors stay in the database and stop matching anything meaningful.',
        'A dimension change needs a schema migration as well as a re-index.',
        'Nothing re-indexes automatically. Run it yourself after saving.',
      ]"
      [confirmWord]="'reindex'"
      [confirmLabel]="'Save and accept the re-index'"
      (cancelled)="confirmOpen.set(false)"
      (confirmed)="commit()"
    />
  `,
})
export class ProvidersComponent implements OnInit {
  private readonly api = inject(ProvidersService);
  private readonly toasts = inject(ToastService);

  readonly llmProviders = LLM_PROVIDERS;
  readonly embeddingProviders = EMBEDDING_PROVIDERS;

  readonly config = signal<ProviderConfig | null>(null);
  readonly loading = signal(true);
  readonly error = signal('');
  readonly saveError = signal('');
  readonly saveMessage = signal('');
  readonly testResult = signal<ProviderTestResult | null>(null);
  readonly confirmOpen = signal(false);

  readonly llmProvider = signal('');
  readonly llmModel = signal('');
  readonly llmBaseUrl = signal('');

  readonly embeddingsProvider = signal('');
  readonly embeddingsModel = signal('');
  readonly embeddingsDim = signal('');
  readonly embeddingsBaseUrl = signal('');

  readonly llmChanged = computed(() => {
    const current = this.config();
    if (current === null) {
      return false;
    }
    return (
      this.llmProvider() !== current.llm.provider ||
      this.llmModel() !== (current.llm.model ?? '') ||
      this.llmBaseUrl() !== (current.llm.base_url ?? '')
    );
  });

  readonly embeddingsChanged = computed(() => {
    const current = this.config();
    if (current === null) {
      return false;
    }
    return (
      this.embeddingsProvider() !== current.embeddings.provider ||
      this.embeddingsModel() !== (current.embeddings.model ?? '') ||
      this.embeddingsDim() !== this.dimText(current.embeddings.dim) ||
      this.embeddingsBaseUrl() !== (current.embeddings.base_url ?? '')
    );
  });

  readonly anythingChanged = computed(() => this.llmChanged() || this.embeddingsChanged());

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.error.set('');
    this.api.read().subscribe({
      next: (config) => {
        this.loading.set(false);
        this.config.set(config);
        this.reset();
      },
      error: (err) => {
        this.loading.set(false);
        this.error.set(describeError(err, 'Could not load the provider configuration.'));
      },
    });
  }

  reset(): void {
    const current = this.config();
    if (current === null) {
      return;
    }
    this.llmProvider.set(current.llm.provider);
    this.llmModel.set(current.llm.model ?? '');
    this.llmBaseUrl.set(current.llm.base_url ?? '');
    this.embeddingsProvider.set(current.embeddings.provider);
    this.embeddingsModel.set(current.embeddings.model ?? '');
    this.embeddingsDim.set(this.dimText(current.embeddings.dim));
    this.embeddingsBaseUrl.set(current.embeddings.base_url ?? '');
    this.saveError.set('');
  }

  keyState(requiresKey: boolean, hasKey: boolean): string {
    if (!requiresKey) {
      return 'not-needed';
    }
    return hasKey ? 'present' : 'missing';
  }

  save(): void {
    if (!this.anythingChanged()) {
      return;
    }
    // An embedding change is the one action here that can destroy the value
    // of the index, so it does not save on a single click.
    if (this.embeddingsChanged()) {
      this.confirmOpen.set(true);
      return;
    }
    this.commit();
  }

  commit(): void {
    this.confirmOpen.set(false);
    const update: ProviderUpdate = {};
    if (this.llmChanged()) {
      update.llm = {
        provider: this.llmProvider(),
        model: this.llmModel().trim() === '' ? null : this.llmModel().trim(),
        base_url: this.llmBaseUrl().trim() === '' ? null : this.llmBaseUrl().trim(),
      };
    }
    if (this.embeddingsChanged()) {
      const dim = Number(this.embeddingsDim());
      update.embeddings = {
        provider: this.embeddingsProvider(),
        model: this.embeddingsModel().trim() === '' ? null : this.embeddingsModel().trim(),
        dim: Number.isFinite(dim) && dim > 0 ? dim : null,
        base_url:
          this.embeddingsBaseUrl().trim() === '' ? null : this.embeddingsBaseUrl().trim(),
      };
    }
    this.saveError.set('');
    this.saveMessage.set('');
    this.api.write(update).subscribe({
      next: (written) => {
        this.config.set({ llm: written.llm, embeddings: written.embeddings });
        this.reset();
        const parts = [`Saved: ${written.changed.join(', ') || 'no effective change'}.`];
        if (written.restart_required) {
          parts.push('Providers are built at startup, so restart the server to pick this up.');
        }
        if (written.requires_reindex) {
          parts.push('The embedding model changed. Run a migration and a full re-index.');
        }
        this.saveMessage.set(parts.join(' '));
        this.toasts.success('Provider configuration saved.');
      },
      error: (err) => {
        const message = describeError(err, 'Could not save the provider configuration.');
        this.saveError.set(message);
        this.toasts.error(message);
      },
    });
  }

  test(target: 'llm' | 'embeddings'): void {
    // Cleared first: a stale "ok" from the previous attempt must never be
    // read as the result of this one.
    this.testResult.set(null);
    this.api.test(target).subscribe({
      next: (result) => this.testResult.set(result),
      error: (err) =>
        this.testResult.set({
          target,
          ok: false,
          detail: describeError(err, 'The test call itself failed.'),
          model: null,
        }),
    });
  }

  private dimText(dim: number | null): string {
    return dim === null ? '' : String(dim);
  }
}
