import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';

import { AdminService } from '../../services/admin.service';
import { AuthService } from '../../services/auth.service';
import { User } from '../../models';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './admin.component.html',
  styleUrl: './admin.component.scss',
})
export class AdminComponent implements OnInit {
  users = signal<User[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  constructor(private admin: AdminService, public auth: AuthService) {}

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.admin.listUsers().subscribe({
      next: (data) => {
        this.users.set(data);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  toggleRole(user: User): void {
    const role = user.role === 'admin' ? 'user' : 'admin';
    this.admin.setPermissions(user.id, { role }).subscribe({
      next: () => this.refresh(),
      error: (err) =>
        this.error.set(
          (err as { error?: { detail?: string } })?.error?.detail ||
            'Update failed.'
        ),
    });
  }

  toggleActive(user: User): void {
    this.admin.setPermissions(user.id, { is_active: !user.is_active }).subscribe({
      next: () => this.refresh(),
      error: (err) =>
        this.error.set(
          (err as { error?: { detail?: string } })?.error?.detail ||
            'Update failed.'
        ),
    });
  }
}
