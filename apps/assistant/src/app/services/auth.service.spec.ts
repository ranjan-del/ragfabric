// Tests for AuthService.
//
// Two things here are easy to get wrong and expensive when they are: the login
// call has to be form-encoded (the backend uses the OAuth2 password form, and a
// JSON body is rejected with a 422 that looks like bad credentials), and
// restore() must clear a token the server no longer accepts instead of leaving
// the app in a half-signed-in state.
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { AuthService } from './auth.service';

const API = '/api';
const TOKEN = { access_token: 'token-1', token_type: 'bearer' };
const USER = { id: 1, email: 'admin@example.com', role: 'admin', is_active: true };

describe('AuthService', () => {
  let service: AuthService;
  let backend: HttpTestingController;

  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(AuthService);
    backend = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    backend.verify();
    localStorage.clear();
  });

  it('starts signed out', () => {
    expect(service.isAuthenticated()).toBeFalse();
    expect(service.isAdmin()).toBeFalse();
    expect(service.token).toBeNull();
  });

  it('posts the credentials as an OAuth2 password form, not as JSON', () => {
    service.login('admin@example.com', 'adminpass123').subscribe();

    const req = backend.expectOne(`${API}/auth/login`);
    expect(req.request.headers.get('Content-Type')).toBe('application/x-www-form-urlencoded');
    expect(req.request.body).toContain('username=admin%40example.com');
    expect(req.request.body).toContain('password=adminpass123');

    req.flush(TOKEN);
    backend.expectOne(`${API}/auth/me`).flush(USER);
  });

  it('stores the token and loads the profile before reporting success', () => {
    const signedInAs: string[] = [];
    service.login('admin@example.com', 'pw').subscribe((u) => signedInAs.push(u.email));

    backend.expectOne(`${API}/auth/login`).flush(TOKEN);
    // Not signed in yet: the token exists but the profile has not arrived, and
    // the caller navigates on the profile, not on the token.
    expect(service.user()).toBeNull();

    backend.expectOne(`${API}/auth/me`).flush(USER);

    expect(signedInAs).toEqual(['admin@example.com']);
    expect(service.token).toBe('token-1');
    expect(service.isAuthenticated()).toBeTrue();
    expect(service.isAdmin()).toBeTrue();
  });

  it('reports a failure when the profile cannot be loaded after login', () => {
    let failed = false;
    service.login('admin@example.com', 'pw').subscribe({ error: () => (failed = true) });

    backend.expectOne(`${API}/auth/login`).flush(TOKEN);
    backend.expectOne(`${API}/auth/me`).flush({}, { status: 500, statusText: 'Server Error' });

    expect(failed).toBeTrue();
    expect(service.user()).toBeNull();
  });

  it('does not sign anyone in when the credentials are rejected', () => {
    let status = 0;
    service.login('admin@example.com', 'wrong').subscribe({ error: (e) => (status = e.status) });
    backend
      .expectOne(`${API}/auth/login`)
      .flush({ detail: 'Incorrect email or password' }, { status: 401, statusText: 'Unauthorized' });

    expect(status).toBe(401);
    expect(service.token).toBeNull();
    expect(service.isAuthenticated()).toBeFalse();
  });

  it('distinguishes a plain user from an admin', () => {
    service.loadProfile().subscribe();
    backend.expectOne(`${API}/auth/me`).flush({ ...USER, role: 'user' });

    expect(service.isAuthenticated()).toBeTrue();
    expect(service.isAdmin()).toBeFalse();
  });

  describe('restore', () => {
    it('does nothing without a stored token', () => {
      service.restore();
      // verify() in afterEach fails if a profile request was made.
      expect(service.isAuthenticated()).toBeFalse();
    });

    it('reloads the profile from a stored token', () => {
      localStorage.setItem('rag_token', 'token-from-last-time');
      service.restore();
      backend.expectOne(`${API}/auth/me`).flush(USER);

      expect(service.isAuthenticated()).toBeTrue();
    });

    it('clears a token the server no longer accepts', () => {
      localStorage.setItem('rag_token', 'expired');
      service.restore();
      backend.expectOne(`${API}/auth/me`).flush({}, { status: 401, statusText: 'Unauthorized' });

      expect(service.token).toBeNull();
      expect(service.isAuthenticated()).toBeFalse();
    });
  });

  it('clears the token and the user on logout', () => {
    service.loadProfile().subscribe();
    backend.expectOne(`${API}/auth/me`).flush(USER);
    localStorage.setItem('rag_token', 'token-1');

    service.logout();

    expect(service.token).toBeNull();
    expect(service.user()).toBeNull();
  });
});
