import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

interface ConsoleTab {
  label: string;
  path: string;
}

/**
 * The console shell: one place for everything that used to require a database
 * client. The tabs are a flat list rather than a nested menu because there are
 * five of them and there will not be twenty.
 */
@Component({
  selector: 'app-console',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, RouterLinkActive, RouterOutlet],
  template: `
    <nav class="console-tabs" aria-label="Console sections">
      @for (tab of tabs; track tab.path) {
        <a [routerLink]="tab.path" routerLinkActive="active" class="console-tab">{{ tab.label }}</a>
      }
    </nav>
    <router-outlet />
  `,
})
export class ConsoleComponent {
  readonly tabs: ConsoleTab[] = [
    { label: 'Users', path: 'users' },
    { label: 'Groups', path: 'groups' },
    { label: 'Grants', path: 'grants' },
    { label: 'API keys', path: 'keys' },
    { label: 'Providers', path: 'providers' },
  ];
}
