// Tests for both route guards.
//
// adminGuard sends a non-admin to /dashboard rather than /login, which is the
// distinction worth pinning: being signed in as the wrong role is not the same
// as not being signed in, and bouncing a valid user to the login page reads as
// a broken session.
import { signal } from '@angular/core';
import { Observable, of, throwError } from 'rxjs';
import { TestBed } from '@angular/core/testing';
import {
  ActivatedRouteSnapshot,
  Router,
  RouterStateSnapshot,
  UrlTree,
  provideRouter,
} from '@angular/router';

import { adminGuard, authGuard } from './auth.guard';
import { AuthService } from '../services/auth.service';

function configure(auth: Partial<AuthService>) {
  TestBed.resetTestingModule();
  TestBed.configureTestingModule({
    providers: [provideRouter([]), { provide: AuthService, useValue: auth }],
  });
}

function run(guard: typeof authGuard) {
  return TestBed.runInInjectionContext(() =>
    guard({} as ActivatedRouteSnapshot, { url: '/documents' } as RouterStateSnapshot),
  );
}

function urlOf(result: unknown): string {
  return TestBed.inject(Router).serializeUrl(result as UrlTree);
}

describe('authGuard', () => {
  it('lets a signed-in user through', () => {
    configure({ token: 'token-1' } as Partial<AuthService>);
    expect(run(authGuard)).toBeTrue();
  });

  it('sends a signed-out user to /login', () => {
    configure({ token: null } as Partial<AuthService>);
    const result = run(authGuard);

    expect(result instanceof UrlTree).toBeTrue();
    expect(urlOf(result)).toBe('/login');
  });
});

describe('adminGuard', () => {
  it('lets an admin through', () => {
    configure({ isAdmin: signal(true) } as unknown as Partial<AuthService>);
    expect(run(adminGuard)).toBeTrue();
  });

  it('sends a signed-in non-admin to /dashboard, not to /login', () => {
    configure({ isAdmin: signal(false) } as unknown as Partial<AuthService>);
    const result = run(adminGuard);

    expect(result instanceof UrlTree).toBeTrue();
    expect(urlOf(result)).toBe('/dashboard');
  });

  it('waits for the profile when an admin route is opened directly', () => {
    configure({
      token: 'token-1',
      isAdmin: signal(false),
      user: signal(null),
      loadProfile: () => of({ id: 1, email: 'a@example.com', role: 'admin', is_active: true, created_at: '' }),
    } as unknown as Partial<AuthService>);
    let outcome: unknown;
    (run(adminGuard) as Observable<unknown>).subscribe((r) => (outcome = r));
    expect(outcome).toBeTrue();
  });

  it('sends a directly opened non-admin to /dashboard and a failed profile to /login', () => {
    configure({
      token: 'token-1',
      isAdmin: signal(false),
      user: signal(null),
      loadProfile: () => of({ id: 2, email: 'u@example.com', role: 'user', is_active: true, created_at: '' }),
    } as unknown as Partial<AuthService>);
    let outcome: unknown;
    (run(adminGuard) as Observable<unknown>).subscribe((r) => (outcome = r));
    expect(urlOf(outcome)).toBe('/dashboard');

    configure({
      token: 'token-1',
      isAdmin: signal(false),
      user: signal(null),
      loadProfile: () => throwError(() => new Error('401')),
    } as unknown as Partial<AuthService>);
    (run(adminGuard) as Observable<unknown>).subscribe((r) => (outcome = r));
    expect(urlOf(outcome)).toBe('/login');
  });
});
