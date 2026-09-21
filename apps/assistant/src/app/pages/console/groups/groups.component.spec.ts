// Console: Groups screen (Task 13).
//
// A group is the unit access is granted to, so the screen has to make
// membership editable in place. Renaming matters more than it looks: without
// it, fixing a typo in a group's name means deleting the group and losing
// every grant and every member it carried.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GroupsComponent } from './groups.component';
import { Group, User } from '../../../models';

const GROUPS: Group[] = [
  { id: 7, name: 'hr-team', description: 'People ops', created_at: '2026-09-01T00:00:00Z' },
];

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

describe('GroupsComponent', () => {
  let fixture: ComponentFixture<GroupsComponent>;
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
    fixture = TestBed.createComponent(GroupsComponent);
    fixture.detectChanges();
  });

  function load(groups: Group[] = GROUPS, members: User[] = [USERS[1]]): void {
    http.expectOne('/api/admin/groups').flush(groups);
    http.expectOne('/api/admin/users').flush(USERS);
    fixture.detectChanges();
    for (const group of groups) {
      http.expectOne(`/api/admin/groups/${group.id}/members`).flush(members);
    }
    fixture.detectChanges();
  }

  it('test_groups_are_listed_with_their_members', () => {
    load();

    expect(all('.group-row').length).toBe(1);
    expect(fixture.nativeElement.textContent).toContain('hr-team');
    expect(fixture.nativeElement.textContent).toContain('member@example.com');
  });

  it('test_a_failed_request_shows_an_error_not_a_blank_table', () => {
    http
      .expectOne('/api/admin/groups')
      .flush({ detail: 'nope' }, { status: 503, statusText: 'Unavailable' });
    http.expectOne('/api/admin/users').flush(USERS);
    fixture.detectChanges();

    expect(el('[role="alert"]').textContent).toBeTruthy();
    expect(fixture.nativeElement.querySelector('table')).toBeNull();
  });

  it('test_creating_a_group_posts_and_reloads', () => {
    load();
    el<HTMLButtonElement>('.groups-new button').click();
    fixture.detectChanges();
    type(el<HTMLInputElement>('.group-create input'), 'finance');
    el<HTMLButtonElement>('.group-create-submit').click();
    fixture.detectChanges();

    const posted = http.expectOne(
      (req) => req.url === '/api/admin/groups' && req.method === 'POST',
    );
    expect(posted.request.body.name).toBe('finance');
  });

  it('test_renaming_a_group_puts_the_new_name', () => {
    load();
    el<HTMLButtonElement>('.group-rename button').click();
    fixture.detectChanges();
    type(el<HTMLInputElement>('.group-edit input'), 'people-team');
    el<HTMLButtonElement>('.group-edit-submit').click();
    fixture.detectChanges();

    const put = http.expectOne('/api/admin/groups/7');
    expect(put.request.method).toBe('PUT');
    expect(put.request.body.name).toBe('people-team');
  });

  it('test_delete_requires_confirmation', () => {
    load();
    el<HTMLButtonElement>('.group-delete button').click();
    fixture.detectChanges();

    expect(el<HTMLButtonElement>('.confirm-action').disabled).toBeTrue();
    el<HTMLButtonElement>('.confirm-action').click();
    http.expectNone((req) => req.method === 'DELETE');

    type(el<HTMLInputElement>('.confirm-word input'), 'hr-team');
    el<HTMLButtonElement>('.confirm-action').click();
    fixture.detectChanges();

    const deleted = http.expectOne('/api/admin/groups/7');
    expect(deleted.request.method).toBe('DELETE');
  });

  it('test_a_member_is_added_and_removed_inline', () => {
    load();
    const select = el<HTMLSelectElement>('.member-add select');
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    el<HTMLButtonElement>('.member-add-submit').click();
    fixture.detectChanges();

    const added = http.expectOne('/api/admin/groups/7/members');
    expect(added.request.method).toBe('POST');
    expect(added.request.body.user_id).toBe(1);
    added.flush({ detail: 'Member added.' });
    fixture.detectChanges();
    http.expectOne('/api/admin/groups/7/members').flush(USERS);
    fixture.detectChanges();

    el<HTMLButtonElement>('.member-remove button').click();
    fixture.detectChanges();
    const removed = http.expectOne('/api/admin/groups/7/members/1');
    expect(removed.request.method).toBe('DELETE');
  });

  it('test_a_member_already_in_the_group_is_not_offered_again', () => {
    load();
    const options = Array.from(el<HTMLSelectElement>('.member-add select').options).map(
      (o) => o.value,
    );

    expect(options).not.toContain('2');
    expect(options).toContain('1');
  });
});
