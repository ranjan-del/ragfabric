import { Injectable, computed, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

import { ApiService } from './api.service';
import { Token, User } from '../models';

const TOKEN_KEY = 'rag_token';

/**
 * Authentication state + token storage. The current user is exposed as a signal
 * so the shell and guards can react to sign-in / sign-out without extra wiring.
 */
@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly _user = signal<User | null>(null);
  readonly user = this._user.asReadonly();
  readonly isAuthenticated = computed(() => this._user() !== null);
  readonly isAdmin = computed(() => this._user()?.role === 'admin');

  constructor(private http: HttpClient, private api: ApiService) {}

  get token(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  /** Log in with email + password (OAuth2 password form) and load the profile. */
  login(email: string, password: string): Observable<User> {
    const body = new URLSearchParams();
    body.set('username', email);
    body.set('password', password);
    return new Observable<User>((subscriber) => {
      this.http
        .post<Token>(`${this.api.baseUrl}/auth/login`, body.toString(), {
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        })
        .subscribe({
          next: (token) => {
            localStorage.setItem(TOKEN_KEY, token.access_token);
            this.loadProfile().subscribe({
              next: (user) => {
                subscriber.next(user);
                subscriber.complete();
              },
              error: (err) => subscriber.error(err),
            });
          },
          error: (err) => subscriber.error(err),
        });
    });
  }

  register(email: string, password: string): Observable<User> {
    return this.http.post<User>(`${this.api.baseUrl}/auth/register`, {
      email,
      password,
    });
  }

  loadProfile(): Observable<User> {
    return this.http
      .get<User>(`${this.api.baseUrl}/auth/me`)
      .pipe(tap((user) => this._user.set(user)));
  }

  /** Restore the session on app start if a token is present. */
  restore(): void {
    if (this.token) {
      this.loadProfile().subscribe({ error: () => this.logout() });
    }
  }

  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    this._user.set(null);
  }
}
