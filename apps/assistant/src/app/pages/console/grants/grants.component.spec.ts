// Console: Grants screen (Task 14).
//
// This is the screen that decides who can read what, so it has to answer the
// question an operator actually has, which is never "list the grant rows". It
// is "can this person read this collection, and why". That answer runs
// through group membership, so the screen resolves it rather than leaving the
// operator to join three tables in their head.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GrantsComponent } from './grants.component';
import { Collection, Grant, Group, User } from '../../../models';

const GROUPS: Group[] = [
  { id: 7, name: 'hr-team', description: '', created_at: '2026-09-01T00:00:00Z' },
  { id: 8, name: 'finance', description: '', created_at: '2026-09-01T00:00:00Z' },
];

const COLLECTIONS: Collection[] = [
  {
    id: 5,
    name: 'hr-policies',
    description: '',
    owner_id: 1,
    created_at: '2026-09-01T00:00:00Z',
    document_count: 3,
  },
  {
    id: 6,
    name: 'ledgers',
    description: '',
    owner_id: 1,
    created_at: '2026-09-01T00:00:00Z',
    document_count: 1,
  },
];

const GRANTS: Grant[] = [{ id: 1, group_id: 7, collection_id: 5, permission: 'read' }];

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
    email: 'member@example.com',
    role: 'user',
    is_active: true,
    created_at: '2026-09-02T00:00:00Z',
  },
];

describe('GrantsComponent', () => {
  let fixture: ComponentFixture<GrantsComponent>;
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

  function pick(select: HTMLSelectElement, value: string): void {
    select.value = value;
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(GrantsComponent);
    fixture.detectChanges();
  });

  function load(grants: Grant[] = GRANTS): void {
    http.expectOne('/api/admin/groups').flush(GROUPS);
    http.expectOne('/api/collections').flush(COLLECTIONS);
    http.expectOne('/api/admin/grants').flush(grants);
    http.expectOne('/api/admin/users').flush(USERS);
    fixture.detectChanges();
    http.expectOne('/api/admin/groups/7/members').flush([USERS[1]]);
    http.expectOne('/api/admin/groups/8/members').flush([]);
    fixture.detectChanges();
  }

  it('test_grants_are_shown_by_group_and_by_collection', () => {
    load();

    const byGroup = el('.by-group').textContent ?? '';
    expect(byGroup).toContain('hr-team');
    expect(byGroup).toContain('hr-policies');

    el<HTMLButtonElement>('.view-collection').click();
    fixture.detectChanges();
    const byCollection = el('.by-collection').textContent ?? '';
    expect(byCollection).toContain('hr-policies');
    expect(byCollection).toContain('hr-team');
  });

  it('test_effective_access_resolves_through_group_membership', () => {
    load();
    pick(el<HTMLSelectElement>('.effective select'), '2');

    const rows = all('.effective-row').map((row) => row.textContent ?? '');
    expect(rows.length).toBe(1);
    expect(rows[0]).toContain('hr-policies');
    expect(rows[0]).toContain('read');
    expect(rows[0]).toContain('hr-team');

    pick(el<HTMLSelectElement>('.effective select'), '1');
    expect(all('.effective-row').length).toBe(0);
    expect(el('.effective').textContent).toContain('No access');
  });

  it('test_creating_a_grant_posts_it', () => {
    load();
    const selects = all('.grant-create select') as HTMLSelectElement[];
    pick(selects[0], '8');
    pick(selects[1], '6');
    pick(selects[2], 'write');
    el<HTMLButtonElement>('.grant-create-submit').click();
    fixture.detectChanges();

    const posted = http.expectOne('/api/admin/grants');
    expect(posted.request.method).toBe('POST');
    expect(posted.request.body).toEqual({
      group_id: 8,
      collection_id: 6,
      permission: 'write',
    });
  });

  it('test_a_grant_that_already_exists_is_reported_not_duplicated', () => {
    load();
    const selects = all('.grant-create select') as HTMLSelectElement[];
    pick(selects[0], '7');
    pick(selects[1], '5');
    el<HTMLButtonElement>('.grant-create-submit').click();
    fixture.detectChanges();

    http.expectNone((req) => req.method === 'POST');
    expect(el('.grant-error').textContent).toContain('already has');
  });

  it('test_revoking_a_grant_deletes_it', () => {
    load();
    el<HTMLButtonElement>('.grant-revoke button').click();
    fixture.detectChanges();
    el<HTMLButtonElement>('.confirm-action').click();
    fixture.detectChanges();

    const deleted = http.expectOne('/api/admin/grants/1');
    expect(deleted.request.method).toBe('DELETE');
  });

  it('test_a_failed_request_shows_an_error_not_a_blank_table', () => {
    http
      .expectOne('/api/admin/groups')
      .flush({ detail: 'nope' }, { status: 500, statusText: 'Server Error' });
    http.expectOne('/api/collections').flush(COLLECTIONS);
    http.expectOne('/api/admin/grants').flush(GRANTS);
    http.expectOne('/api/admin/users').flush(USERS);
    fixture.detectChanges();

    expect(el('[role="alert"]').textContent).toBeTruthy();
    expect(fixture.nativeElement.querySelector('table')).toBeNull();
  });

  it('test_a_group_with_no_grants_says_so', () => {
    load([]);

    expect(el('.by-group').textContent).toContain('No access');
  });
});
