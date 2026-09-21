// Console: API keys screen (Task 15).
//
// The security sensitive screen, so its specs are about what must NOT happen.
// The plaintext key exists in the browser for exactly one panel: it is shown
// once, acknowledged, and then it is gone from the DOM. It is never put in
// local storage, never in a URL, and never in a log, because every one of
// those outlives the panel and none of them is encrypted.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { KeysComponent } from './keys.component';
import { ApiKeyItem, User } from '../../../models';

const SECRET = 'rf_this_is_the_only_time_you_see_me_0123456789';

const USERS: User[] = [
  {
    id: 1,
    email: 'admin@example.com',
    role: 'admin',
    is_active: true,
    created_at: '2026-09-01T00:00:00Z',
  },
];

const KEYS: ApiKeyItem[] = [
  {
    id: 4,
    name: 'ci',
    key_prefix: 'rf_abc123def',
    principal_user_id: 1,
    collection_ids: [5],
    strategies: ['vectorless'],
    rate_limit_per_minute: 60,
    is_active: true,
    created_at: '2026-09-01T00:00:00Z',
    last_used_at: '2026-09-20T00:00:00Z',
    expires_at: null,
  },
];

describe('KeysComponent', () => {
  let fixture: ComponentFixture<KeysComponent>;
  let http: HttpTestingController;
  let writeText: jasmine.Spy;

  function el<T extends HTMLElement>(selector: string): T {
    const found = fixture.nativeElement.querySelector(selector) as T | null;
    if (found === null) {
      throw new Error(`no element matched ${selector}`);
    }
    return found;
  }

  function type(input: HTMLInputElement, value: string): void {
    input.value = value;
    input.dispatchEvent(new Event('input'));
    fixture.detectChanges();
  }

  function dom(): string {
    return (fixture.nativeElement as HTMLElement).innerHTML;
  }

  beforeEach(() => {
    writeText = jasmine.createSpy('writeText').and.returnValue(Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });
    localStorage.clear();

    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(KeysComponent);
    fixture.detectChanges();
  });

  afterEach(() => localStorage.clear());

  function load(keys: ApiKeyItem[] = KEYS): void {
    http.expectOne('/api/admin/keys').flush(keys);
    http.expectOne('/api/admin/users').flush(USERS);
    fixture.detectChanges();
  }

  function createKey(): void {
    el<HTMLButtonElement>('.keys-new button').click();
    fixture.detectChanges();
    type(el<HTMLInputElement>('.key-create input'), 'deploy-bot');
    const select = el<HTMLSelectElement>('.key-create select');
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    el<HTMLButtonElement>('.key-create-submit').click();
    fixture.detectChanges();

    http
      .expectOne((req) => req.url === '/api/admin/keys' && req.method === 'POST')
      .flush({ ...KEYS[0], id: 9, name: 'deploy-bot', key: SECRET });
    fixture.detectChanges();
  }

  it('test_the_list_shows_the_prefix_and_never_a_secret', () => {
    load();
    const row = el('.key-row').textContent ?? '';

    expect(row).toContain('rf_abc123def');
    expect(row).toContain('ci');
    expect(row).toContain('vectorless');
    expect(dom()).not.toContain(SECRET);
  });

  it('test_a_failed_request_shows_an_error_not_a_blank_table', () => {
    http
      .expectOne('/api/admin/keys')
      .flush({ detail: 'nope' }, { status: 500, statusText: 'Server Error' });
    http.expectOne('/api/admin/users').flush(USERS);
    fixture.detectChanges();

    expect(el('[role="alert"]').textContent).toBeTruthy();
    expect(fixture.nativeElement.querySelector('table')).toBeNull();
  });

  it('test_the_secret_is_shown_once_on_creation', () => {
    load();
    createKey();

    expect(el('.secret-value').textContent).toContain(SECRET);
    expect(el('.secret-panel').getAttribute('role')).toBe('dialog');
  });

  it('test_the_panel_cannot_be_closed_until_the_secret_is_acknowledged', () => {
    load();
    createKey();

    const done = el<HTMLButtonElement>('.secret-done');
    expect(done.disabled).toBeTrue();
    done.click();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.secret-panel')).not.toBeNull();

    const ack = el<HTMLInputElement>('.secret-ack input');
    ack.checked = true;
    ack.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(el<HTMLButtonElement>('.secret-done').disabled).toBeFalse();
  });

  it('test_the_secret_is_not_present_in_the_dom_after_the_panel_closes', () => {
    load();
    createKey();
    expect(dom()).toContain(SECRET);

    const ack = el<HTMLInputElement>('.secret-ack input');
    ack.checked = true;
    ack.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    el<HTMLButtonElement>('.secret-done').click();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.secret-panel')).toBeNull();
    expect(dom()).not.toContain(SECRET);
    expect((fixture.nativeElement as HTMLElement).textContent).not.toContain(SECRET);

    http.expectOne('/api/admin/keys').flush(KEYS);
    fixture.detectChanges();
    expect(dom()).not.toContain(SECRET);
  });

  it('test_the_secret_is_never_written_to_local_storage', () => {
    const setItem = spyOn(Storage.prototype, 'setItem').and.callThrough();
    load();
    createKey();

    const ack = el<HTMLInputElement>('.secret-ack input');
    ack.checked = true;
    ack.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    el<HTMLButtonElement>('.secret-done').click();
    fixture.detectChanges();

    for (const args of setItem.calls.allArgs()) {
      expect(String(args[1])).not.toContain(SECRET);
    }
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      expect(localStorage.getItem(key ?? '')).not.toContain(SECRET);
    }
    expect(sessionStorage.getItem('rf_key')).toBeNull();
  });

  it('test_copying_puts_the_secret_on_the_clipboard_and_nowhere_else', () => {
    load();
    createKey();
    el<HTMLButtonElement>('.key-copy').click();
    fixture.detectChanges();

    expect(writeText).toHaveBeenCalledWith(SECRET);
    expect(window.location.href).not.toContain(SECRET);
  });

  it('test_revoking_a_key_requires_confirmation', () => {
    load();
    el<HTMLButtonElement>('.key-revoke button').click();
    fixture.detectChanges();

    el<HTMLButtonElement>('.confirm-action').click();
    fixture.detectChanges();

    const revoked = http.expectOne('/api/admin/keys/4');
    expect(revoked.request.method).toBe('DELETE');
  });
});
