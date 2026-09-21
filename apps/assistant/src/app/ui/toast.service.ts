import { Injectable, signal } from '@angular/core';

export type ToastTone = 'success' | 'error' | 'info';

export interface Toast {
  id: number;
  tone: ToastTone;
  message: string;
}

/** How long a toast stays up before it dismisses itself, in milliseconds. */
const LIFETIME = 6000;

/**
 * Transient feedback for actions that succeeded or failed.
 *
 * An error toast does not expire on its own. A success message the operator
 * missed costs nothing; a failure they missed means they believe something
 * happened that did not.
 */
@Injectable({ providedIn: 'root' })
export class ToastService {
  private nextId = 1;
  private readonly _toasts = signal<Toast[]>([]);
  readonly toasts = this._toasts.asReadonly();

  success(message: string): void {
    this.push('success', message, LIFETIME);
  }

  info(message: string): void {
    this.push('info', message, LIFETIME);
  }

  error(message: string): void {
    this.push('error', message, null);
  }

  dismiss(id: number): void {
    this._toasts.update((list) => list.filter((toast) => toast.id !== id));
  }

  clear(): void {
    this._toasts.set([]);
  }

  private push(tone: ToastTone, message: string, lifetime: number | null): void {
    const id = this.nextId++;
    this._toasts.update((list) => [...list, { id, tone, message }]);
    if (lifetime !== null) {
      setTimeout(() => this.dismiss(id), lifetime);
    }
  }
}
