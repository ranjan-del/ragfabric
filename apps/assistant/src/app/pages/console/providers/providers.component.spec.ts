// Console: Provider configuration (Task 16).
//
// Changing the embedding provider or model is the single most destructive
// action in the console. ADR 0006 pins the embedding dimension of an index,
// so every stored vector was produced by the model named here; changing it
// makes the whole index meaningless until a migration and a full re-index
// have run. The screen has to say that before the change, not after it.
//
// The connection test is the other half: it makes a real call and reports the
// real outcome. A screen that says "connection ok" without calling anything
// is worse than no screen, because it is believed.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProvidersComponent } from './providers.component';
import { ProviderConfig } from '../../../models';

const CONFIG: ProviderConfig = {
  llm: {
    provider: 'openai',
    model: 'gpt-5.4-mini',
    base_url: null,
    dim: null,
    key_env_var: 'OPENAI_API_KEY',
    requires_key: true,
    has_key: true,
  },
  embeddings: {
    provider: 'ollama',
    model: 'nomic-embed-text',
    base_url: 'http://localhost:11434/v1',
    dim: 768,
    key_env_var: null,
    requires_key: false,
    has_key: false,
  },
};

describe('ProvidersComponent', () => {
  let fixture: ComponentFixture<ProvidersComponent>;
  let http: HttpTestingController;

  function el<T extends HTMLElement>(selector: string): T {
    const found = fixture.nativeElement.querySelector(selector) as T | null;
    if (found === null) {
      throw new Error(`no element matched ${selector}`);
    }
    return found;
  }

  function type(selector: string, value: string): void {
    const input = el<HTMLInputElement>(selector);
    input.value = value;
    input.dispatchEvent(new Event('input'));
    fixture.detectChanges();
  }

  function pick(selector: string, value: string): void {
    const select = el<HTMLSelectElement>(selector);
    select.value = value;
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(ProvidersComponent);
    fixture.detectChanges();
  });

  function load(config: ProviderConfig = CONFIG): void {
    http.expectOne('/api/admin/providers').flush(config);
    fixture.detectChanges();
  }

  it('test_the_configuration_is_shown_and_the_secret_is_not', () => {
    load();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';

    expect(el<HTMLSelectElement>('.llm-provider select').value).toBe('openai');
    expect(el<HTMLInputElement>('.llm-model input').value).toBe('gpt-5.4-mini');
    expect(text).toContain('OPENAI_API_KEY');
    expect(el('.llm-key-status').textContent).toContain('set');
    expect((fixture.nativeElement as HTMLElement).innerHTML).not.toContain('sk-');
  });

  it('test_a_provider_whose_key_is_missing_is_flagged', () => {
    load({ ...CONFIG, llm: { ...CONFIG.llm, has_key: false } });

    expect(el('.llm-key-status').textContent).toContain('not set');
    expect(el('.llm-key-status').getAttribute('data-state')).toBe('missing');
  });

  it('test_a_failed_load_shows_an_error_not_an_empty_form', () => {
    http
      .expectOne('/api/admin/providers')
      .flush({ detail: 'nope' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    expect(el('[role="alert"]').textContent).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.llm-provider')).toBeNull();
  });

  it('test_changing_the_embedding_model_warns_about_reindexing', () => {
    load();
    expect(fixture.nativeElement.querySelector('.reindex-warning')).toBeNull();

    type('.embeddings-model input', 'text-embedding-3-large');

    const warning = el('.reindex-warning');
    expect(warning.getAttribute('role')).toBe('alert');
    const text = warning.textContent ?? '';
    expect(text).toContain('re-index');
    expect(text).toContain('dimension');
    expect(text).toContain('migration');
  });

  it('test_changing_the_embedding_provider_also_warns', () => {
    load();
    pick('.embeddings-provider select', 'openai');

    expect(fixture.nativeElement.querySelector('.reindex-warning')).not.toBeNull();
  });

  it('test_changing_only_the_llm_does_not_warn', () => {
    load();
    type('.llm-model input', 'gpt-5.4');

    expect(fixture.nativeElement.querySelector('.reindex-warning')).toBeNull();
  });

  it('test_an_embedding_change_cannot_be_saved_without_a_typed_confirmation', () => {
    load();
    type('.embeddings-model input', 'text-embedding-3-large');
    el<HTMLButtonElement>('.providers-save').click();
    fixture.detectChanges();

    http.expectNone((req) => req.method === 'PUT');
    const confirm = el<HTMLButtonElement>('.confirm-action');
    expect(confirm.disabled).toBeTrue();

    const word = el<HTMLInputElement>('.confirm-word input');
    word.value = 'reindex';
    word.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    el<HTMLButtonElement>('.confirm-action').click();
    fixture.detectChanges();

    const put = http.expectOne('/api/admin/providers');
    expect(put.request.method).toBe('PUT');
    expect(put.request.body.embeddings.model).toBe('text-embedding-3-large');
  });

  it('test_an_llm_only_change_saves_without_a_dialog', () => {
    load();
    type('.llm-model input', 'gpt-5.4');
    el<HTMLButtonElement>('.providers-save').click();
    fixture.detectChanges();

    const put = http.expectOne('/api/admin/providers');
    expect(put.request.method).toBe('PUT');
    expect(put.request.body.llm.model).toBe('gpt-5.4');
    expect(put.request.body.embeddings).toBeUndefined();
  });

  it('test_a_connection_test_reports_a_real_failure_as_a_failure', () => {
    load();
    el<HTMLButtonElement>('.llm-test').click();
    fixture.detectChanges();

    const posted = http.expectOne('/api/admin/providers/test');
    expect(posted.request.body).toEqual({ target: 'llm' });
    posted.flush({ target: 'llm', ok: false, detail: 'openai: OPENAI_API_KEY is not set', model: null });
    fixture.detectChanges();

    const result = el('.test-result');
    expect(result.textContent).toContain('OPENAI_API_KEY is not set');
    expect(result.getAttribute('data-ok')).toBe('false');
  });

  it('test_a_connection_test_reports_success_only_after_a_real_call', () => {
    load();
    expect(fixture.nativeElement.querySelector('.test-result')).toBeNull();

    el<HTMLButtonElement>('.embeddings-test').click();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.test-result')).toBeNull();

    http
      .expectOne('/api/admin/providers/test')
      .flush({ target: 'embeddings', ok: true, detail: '', model: 'nomic-embed-text' });
    fixture.detectChanges();

    const result = el('.test-result');
    expect(result.getAttribute('data-ok')).toBe('true');
    expect(result.textContent).toContain('nomic-embed-text');
  });

  it('test_a_saved_change_reports_that_a_restart_is_needed', () => {
    load();
    type('.llm-model input', 'gpt-5.4');
    el<HTMLButtonElement>('.providers-save').click();
    fixture.detectChanges();

    http.expectOne('/api/admin/providers').flush({
      ...CONFIG,
      llm: { ...CONFIG.llm, model: 'gpt-5.4' },
      requires_reindex: false,
      restart_required: true,
      changed: ['llm.model'],
    });
    fixture.detectChanges();

    expect(el('.save-result').textContent).toContain('restart');
  });
});
