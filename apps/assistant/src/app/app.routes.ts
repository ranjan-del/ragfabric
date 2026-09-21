import { Routes } from '@angular/router';

import { authGuard, adminGuard } from './guards/auth.guard';

export const routes: Routes = [
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  {
    path: 'login',
    loadComponent: () =>
      import('./pages/login/login.component').then((m) => m.LoginComponent),
  },
  {
    path: 'dashboard',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/dashboard/dashboard.component').then(
        (m) => m.DashboardComponent
      ),
  },
  {
    path: 'documents',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/documents/documents.component').then(
        (m) => m.DocumentsComponent
      ),
  },
  {
    path: 'search',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/search/search.component').then((m) => m.SearchComponent),
  },
  {
    path: 'collections',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/collections/collections.component').then(
        (m) => m.CollectionsComponent
      ),
  },
  {
    path: 'analytics',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/analytics/analytics.component').then(
        (m) => m.AnalyticsComponent
      ),
  },
  {
    path: 'admin',
    canActivate: [authGuard, adminGuard],
    loadComponent: () =>
      import('./pages/admin/admin.component').then((m) => m.AdminComponent),
  },
  {
    path: 'console',
    canActivate: [authGuard, adminGuard],
    loadComponent: () =>
      import('./pages/console/console.component').then((m) => m.ConsoleComponent),
    children: [
      { path: '', redirectTo: 'users', pathMatch: 'full' },
      {
        path: 'users',
        loadComponent: () =>
          import('./pages/console/users/users.component').then((m) => m.UsersComponent),
      },
      {
        path: 'groups',
        loadComponent: () =>
          import('./pages/console/groups/groups.component').then((m) => m.GroupsComponent),
      },
      {
        path: 'grants',
        loadComponent: () =>
          import('./pages/console/grants/grants.component').then((m) => m.GrantsComponent),
      },
      {
        path: 'keys',
        loadComponent: () =>
          import('./pages/console/keys/keys.component').then((m) => m.KeysComponent),
      },
    ],
  },
  { path: '**', redirectTo: 'dashboard' },
];
