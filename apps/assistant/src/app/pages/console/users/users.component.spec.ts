// Console: Users screen (Task 13).
//
// The screen's job is that an operator can onboard and remove people without
// a database client. The two specs the plan names by hand are the two ways
// that goes wrong in practice: a delete that happens on one careless click,
// and a failed load that renders as an empty table, so the operator concludes
// there are no users rather than that the request failed.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UsersComponent } from './users.component';
import { User } from '../../../models';

const USERS: User[] = [
  {
    id: 1,
    email: 'admin@example.com',
    role: 'admin',
    is_active: true,
    created_at: '2026-09-01T00:00:00Z',
  },
  {
    id: 2,
    email: 'victim@example.com',
    role: 'user',
    is_active: true,
    created_at: '2026-09-02T00:00:00Z',
  },
];

describe('UsersComponent', () => {
  let fixture: ComponentFixture<UsersComponent>;
  let http: HttpTestingController;

  function el<T extends HTMLElement>(selector: string): T {
    const found = fixture.nativeElement.querySelector(selector) as T | null;
    if (found === null) {
      throw new Error(`no element matched ${selector}`);
    }
    return found;
  }

  function all(selector: string): HTMLElement[] {
    return Array.from(fixture.nativeElement.querySelectorAll(selector));
  }

  function type(input: HTMLInputElement, value: string): void {
    input.value = value;
    input.dispatchEvent(new Event('input'));
    fixture.detectChanges();
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(UsersComponent);
    fixture.detectChanges();
  });

  function load(users: User[] = USERS): void {
    http.expectOne('/api/admin/users').flush(users);
    fixture.detectChanges();
  }

  it('test_the_list_comes_from_the_api', () => {
    load();

    expect(all('.user-row').length).toBe(2);
    expect(fixture.nativeElement.textContent).toContain('victim@example.com');
  });

  it('test_a_failed_request_shows_an_error_not_a_blank_table', () => {
    http
      .expectOne('/api/admin/users')
      .flush({ detail: 'nope' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    const alert = el('[role="alert"]');
    expect(alert.textContent).toBeTruthy();
    expect(fixture.nativeElement.querySelector('table')).toBeNull();
    expect(all('.user-row').length).toBe(0);
  });

  it('test_an_empty_list_is_not_an_error', () => {
    load([]);

    expect(fixture.nativeElement.querySelector('[role="alert"]')).toBeNull();
    expect(fixture.nativeElement.textContent).toContain('No users');
  });

  it('test_searching_filters_the_rows', () => {
    load();
    type(el<HTMLInputElement>('.user-search input'), 'victim');

    expect(all('.user-row').length).toBe(1);
    expect(fixture.nativeElement.textContent).toContain('victim@example.com');
  });

  it('test_creating_a_user_posts_and_reloads', () => {
    load();
    el<HTMLButtonElement>('.users-new button').click();
    fixture.detectChanges();

    const inputs = all('.user-create input') as HTMLInputElement[];
    type(inputs[0], 'new@example.com');
    type(inputs[1], 'password123');
    el<HTMLButtonElement>('.user-create-submit').click();
    fixture.detectChanges();

    const posted = http.expectOne(
      (req) => req.url === '/api/admin/users' && req.method === 'POST',
    );
    expect(posted.request.body.email).toBe('new@example.com');
    posted.flush({ ...USERS[1], id: 3, email: 'new@example.com' });
    fixture.detectChanges();

    http.expectOne('/api/admin/users').flush(USERS);
  });

  it('test_delete_requires_confirmation', () => {
    load();
    const deleteButtons = all('.user-delete button') as HTMLButtonElement[];
    deleteButtons[deleteButtons.length - 1].click();
    fixture.detectChanges();

    const confirm = el<HTMLButtonElement>('.confirm-action');
    expect(confirm.disabled).toBeTrue();
    confirm.click();
    http.expectNone((req) => req.method === 'DELETE');

    type(el<HTMLInputElement>('.confirm-word input'), 'victim@example.com');
    expect(el<HTMLButtonElement>('.confirm-action').disabled).toBeFalse();
    el<HTMLButtonElement>('.confirm-action').click();
    fixture.detectChanges();

    const deleted = http.expectOne('/api/admin/users/2');
    expect(deleted.request.method).toBe('DELETE');
  });

  it('test_the_delete_dialog_says_what_else_it_removes', () => {
    load();
    const deleteButtons = all('.user-delete button') as HTMLButtonElement[];
    deleteButtons[deleteButtons.length - 1].click();
    fixture.detectChanges();

    const text = el('[role="dialog"]').textContent ?? '';
    expect(text).toContain('API keys');
    expect(text).toContain('cannot be undone');
  });

  it('test_a_failed_delete_leaves_the_row_and_reports_the_failure', () => {
    load();
    const deleteButtons = all('.user-delete button') as HTMLButtonElement[];
    deleteButtons[deleteButtons.length - 1].click();
    fixture.detectChanges();
    type(el<HTMLInputElement>('.confirm-word input'), 'victim@example.com');
    el<HTMLButtonElement>('.confirm-action').click();
    fixture.detectChanges();

    http
      .expectOne('/api/admin/users/2')
      .flush({ detail: 'You cannot delete your own account.' }, { status: 400, statusText: 'Bad' });
    fixture.detectChanges();

    expect(all('.user-row').length).toBe(2);
    expect(fixture.nativeElement.textContent).toContain('cannot delete your own account');
  });

  it('test_changing_a_role_puts_the_permission_change', () => {
    load();
    const editButtons = all('.user-edit button') as HTMLButtonElement[];
    editButtons[editButtons.length - 1].click();
    fixture.detectChanges();

    const select = el<HTMLSelectElement>('.user-edit-form select');
    select.value = 'admin';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    el<HTMLButtonElement>('.user-edit-submit').click();
    fixture.detectChanges();

    const put = http.expectOne('/api/admin/users/2/permissions');
    expect(put.request.method).toBe('PUT');
    expect(put.request.body.role).toBe('admin');
  });
});
