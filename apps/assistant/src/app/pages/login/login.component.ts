import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss',
})
export class LoginComponent {
  mode = signal<'login' | 'register'>('login');
  email = '';
  password = '';
  loading = signal(false);
  error = signal<string | null>(null);

  constructor(private auth: AuthService, private router: Router) {}

  switchMode(mode: 'login' | 'register'): void {
    this.mode.set(mode);
    this.error.set(null);
  }

  submit(): void {
    if (!this.email || !this.password) {
      this.error.set('Enter an email and password.');
      return;
    }
    this.loading.set(true);
    this.error.set(null);

    if (this.mode() === 'register') {
      this.auth.register(this.email, this.password).subscribe({
        next: () => this.doLogin(),
        error: (err) => this.fail(err, 'Registration failed.'),
      });
    } else {
      this.doLogin();
    }
  }

  private doLogin(): void {
    this.auth.login(this.email, this.password).subscribe({
      next: () => {
        this.loading.set(false);
        this.router.navigate(['/dashboard']);
      },
      error: (err) => this.fail(err, 'Invalid email or password.'),
    });
  }

  private fail(err: unknown, fallback: string): void {
    this.loading.set(false);
    const detail = (err as { error?: { detail?: string } })?.error?.detail;
    this.error.set(detail || fallback);
  }
}
