// Tests for the auth interceptor.
import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';

import { authInterceptor } from './auth.interceptor';

const API = '/api';

describe('authInterceptor', () => {
  let http: HttpClient;
  let backend: HttpTestingController;
  let router: jasmine.SpyObj<Router>;

  beforeEach(() => {
    localStorage.clear();
    router = jasmine.createSpyObj<Router>('Router', ['navigate']);
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
        { provide: Router, useValue: router },
      ],
    });
    http = TestBed.inject(HttpClient);
    backend = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    backend.verify();
    localStorage.clear();
  });

  it('attaches the bearer token when one is stored', () => {
    localStorage.setItem('rag_token', 'token-1');
    http.get(`${API}/documents`).subscribe();
    const req = backend.expectOne(`${API}/documents`);

    expect(req.request.headers.get('Authorization')).toBe('Bearer token-1');
    req.flush([]);
  });

  it('sends no Authorization header when signed out', () => {
    http.get(`${API}/documents`).subscribe({ error: () => {} });
    const req = backend.expectOne(`${API}/documents`);

    expect(req.request.headers.has('Authorization')).toBeFalse();
    req.flush({}, { status: 401, statusText: 'Unauthorized' });
  });

  it('ends the session and redirects when a request comes back 401', () => {
    localStorage.setItem('rag_token', 'expired');
    http.get(`${API}/documents`).subscribe({ error: () => {} });
    backend.expectOne(`${API}/documents`).flush({}, { status: 401, statusText: 'Unauthorized' });

    expect(localStorage.getItem('rag_token')).toBeNull();
    expect(router.navigate).toHaveBeenCalledWith(['/login']);
  });

  it('leaves a failed login on the login page', () => {
    // A 401 from /auth/login means "wrong password", not "session expired".
    // Redirecting would send the user to the page they are already looking at
    // and would wipe the form's error state on the way.
    http.post(`${API}/auth/login`, 'username=a&password=b').subscribe({ error: () => {} });
    backend
      .expectOne(`${API}/auth/login`)
      .flush({ detail: 'Incorrect email or password' }, { status: 401, statusText: 'Unauthorized' });

    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('leaves a 403 alone, because the token is valid and the role is not', () => {
    localStorage.setItem('rag_token', 'token-1');
    let status = 0;
    http.get(`${API}/admin/users`).subscribe({ error: (e) => (status = e.status) });
    backend
      .expectOne(`${API}/admin/users`)
      .flush({ detail: 'Forbidden' }, { status: 403, statusText: 'Forbidden' });

    expect(status).toBe(403);
    expect(localStorage.getItem('rag_token')).toBe('token-1');
    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('passes a server error through without signing the user out', () => {
    localStorage.setItem('rag_token', 'token-1');
    let status = 0;
    http.get(`${API}/search`).subscribe({ error: (e) => (status = e.status) });
    backend.expectOne(`${API}/search`).flush({}, { status: 500, statusText: 'Server Error' });

    expect(status).toBe(500);
    expect(localStorage.getItem('rag_token')).toBe('token-1');
  });
});
