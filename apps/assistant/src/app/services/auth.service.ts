import { Injectable, computed, inject, signal } from '@angular/core';
import { requests } from '@ragfabric/sdk';
import { Observable, tap } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';
import { User } from '../models';

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

  private readonly sdk = inject(SdkHttp);

  get token(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  /** Log in with email + password (OAuth2 password form) and load the profile. */
  login(email: string, password: string): Observable<User> {
    return new Observable<User>((subscriber) => {
      this.sdk
        .send(requests.auth.login(email, password))
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
    return this.sdk.send(requests.auth.register(email, password));
  }

  loadProfile(): Observable<User> {
    return this.sdk
      .send(requests.auth.me())
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
