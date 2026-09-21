import { Injectable, signal } from '@angular/core';

export type Theme = 'light' | 'dark' | 'system';

const THEME_KEY = 'rag_theme';

/**
 * Light, dark, or whatever the operating system says.
 *
 * "system" is the default and is represented by the ABSENCE of a data-theme
 * attribute, so the prefers-color-scheme media query in the stylesheet is
 * what decides. An explicit choice writes the attribute, which wins over the
 * media query, which is the whole point of having a toggle.
 */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly _theme = signal<Theme>('system');
  readonly theme = this._theme.asReadonly();

  constructor() {
    this.restore();
  }

  restore(): void {
    const saved = this.read();
    if (saved === 'light' || saved === 'dark') {
      this.set(saved);
    }
  }

  set(theme: Theme): void {
    this._theme.set(theme);
    const root = document.documentElement;
    if (theme === 'system') {
      root.removeAttribute('data-theme');
      this.forget();
      return;
    }
    root.setAttribute('data-theme', theme);
    this.remember(theme);
  }

  toggle(): void {
    this.set(this._theme() === 'dark' ? 'light' : 'dark');
  }

  // localStorage throws in a private window with site data blocked, and a
  // theme preference is never worth failing a page load over.
  private read(): string | null {
    try {
      return localStorage.getItem(THEME_KEY);
    } catch {
      return null;
    }
  }

  private remember(theme: Theme): void {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Ignored on purpose: see read().
    }
  }

  private forget(): void {
    try {
      localStorage.removeItem(THEME_KEY);
    } catch {
      // Ignored on purpose: see read().
    }
  }
}
