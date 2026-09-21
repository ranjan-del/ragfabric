// Collections screen, brought onto the console primitives (Task 14).
//
// Two gaps this closes. A collection could not be renamed, so fixing a typo
// meant deleting it, which cascades to every document inside. And a failed
// load silently produced the same empty page as a genuinely empty account,
// which is the exact failure the console's table shell exists to prevent.

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CollectionsComponent } from './collections.component';
import { Collection } from '../../models';

const COLLECTIONS: Collection[] = [
  {
    id: 5,
    name: 'hr-policies',
    description: 'Leave and benefits',
    owner_id: 1,
    created_at: '2026-09-01T00:00:00Z',
    document_count: 3,
  },
];

describe('CollectionsComponent', () => {
  let fixture: ComponentFixture<CollectionsComponent>;
  let http: HttpTestingController;

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

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(CollectionsComponent);
    fixture.detectChanges();
  });

  function load(collections: Collection[] = COLLECTIONS): void {
    http.expectOne('/api/collections').flush(collections);
    fixture.detectChanges();
  }

  it('test_collections_are_listed', () => {
    load();

    expect(fixture.nativeElement.querySelectorAll('.collection-card').length).toBe(1);
    expect(fixture.nativeElement.textContent).toContain('hr-policies');
  });

  it('test_a_failed_load_shows_an_error_not_an_empty_page', () => {
    http
      .expectOne('/api/collections')
      .flush({ detail: 'nope' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    expect(el('[role="alert"]').textContent).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.collection-card')).toBeNull();
    expect(fixture.nativeElement.textContent).not.toContain('No collections yet');
  });

  it('test_renaming_a_collection_puts_the_change', () => {
    load();
    el<HTMLButtonElement>('.collection-rename').click();
    fixture.detectChanges();

    const inputs = Array.from(
      fixture.nativeElement.querySelectorAll('.collection-edit input'),
    ) as HTMLInputElement[];
    type(inputs[0], 'people-policies');
    el<HTMLButtonElement>('.collection-edit-submit').click();
    fixture.detectChanges();

    const put = http.expectOne('/api/collections/5');
    expect(put.request.method).toBe('PUT');
    expect(put.request.body.name).toBe('people-policies');
  });

  it('test_deleting_a_collection_requires_typed_confirmation_and_says_what_goes_with_it', () => {
    load();
    el<HTMLButtonElement>('.collection-delete').click();
    fixture.detectChanges();

    expect(el('[role="dialog"]').textContent).toContain('3 document');
    const confirm = el<HTMLButtonElement>('.confirm-action');
    expect(confirm.disabled).toBeTrue();
    confirm.click();
    http.expectNone((req) => req.method === 'DELETE');

    type(el<HTMLInputElement>('.confirm-word input'), 'hr-policies');
    el<HTMLButtonElement>('.confirm-action').click();
    fixture.detectChanges();

    const deleted = http.expectOne('/api/collections/5');
    expect(deleted.request.method).toBe('DELETE');
  });
});
