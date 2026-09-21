import { Component, OnInit, signal, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive, Router } from '@angular/router';

import { AuthService } from './services/auth.service';
import { ThemeService } from './ui/theme.service';
import { ToastsComponent } from './ui/toasts.component';

interface NavItem {
  label: string;
  path: string;
  icon: string;
  adminOnly?: boolean;
}

@Component({
    selector: 'app-root',
    imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive, ToastsComponent],
    templateUrl: './app.component.html',
    changeDetection: ChangeDetectionStrategy.Eager,
    styleUrl: './app.component.scss'
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

  constructor(
    public auth: AuthService,
    public theme: ThemeService,
    private router: Router,
  ) {}

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
