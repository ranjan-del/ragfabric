import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive, Router } from '@angular/router';

import { AuthService } from './services/auth.service';

interface NavItem {
  label: string;
  path: string;
  icon: string;
  adminOnly?: boolean;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements OnInit {
  readonly navItems: NavItem[] = [
    { label: 'Dashboard', path: '/dashboard', icon: '▚' },
    { label: 'Ask', path: '/search', icon: '✦' },
    { label: 'Documents', path: '/documents', icon: '▤' },
    { label: 'Collections', path: '/collections', icon: '◫' },
    { label: 'Analytics', path: '/analytics', icon: '◔' },
    { label: 'Admin', path: '/admin', icon: '⚙', adminOnly: true },
  ];

  readonly menuOpen = signal(false);

  constructor(public auth: AuthService, private router: Router) {}

  ngOnInit(): void {
    this.auth.restore();
  }

  logout(): void {
    this.auth.logout();
    this.router.navigate(['/login']);
  }

  toggleMenu(): void {
    this.menuOpen.update((v) => !v);
  }

  closeMenu(): void {
    this.menuOpen.set(false);
  }
}
