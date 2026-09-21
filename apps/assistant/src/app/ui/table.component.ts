import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

export type TableDensity = 'comfortable' | 'compact';

/**
 * The console's table shell.
 *
 * Tables are the main surface here, so the four states a table can be in are
 * decided once, in this component, instead of being re-invented per screen:
 *
 *   error    an alert with the real message and a way to try again
 *   loading  a skeleton with the shape of the rows to come
 *   empty    an explanation that nothing exists yet
 *   ready    the rows
 *
 * The order matters. A request that failed while loading must read as failed,
 * not as still loading, and never as empty. Collapsing these into one blank
 * table is precisely how a broken console passes for a working one with no
 * data in it.
 */
@Component({
  selector: 'ui-table',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (error()) {
      <div class="table-state table-error" role="alert">
        <p class="table-error-message">{{ error() }}</p>
        <p class="table-retry">
          <button type="button" class="btn btn-secondary btn-sm" (click)="retried.emit()">
            Try again
          </button>
        </p>
      </div>
    } @else if (loading()) {
      <div class="table-skeleton" aria-busy="true" aria-live="polite">
        <span class="sr-only">Loading</span>
        @for (row of skeletonRows; track row) {
          <div class="skeleton-row"></div>
        }
      </div>
    } @else if (empty()) {
      <div class="table-state">
        <ng-content select="[table-empty]" />
      </div>
    } @else {
      <div class="table-wrap">
        <table [class]="tableClasses()">
          <thead>
            <ng-content select="[head]" />
          </thead>
          <tbody>
            <ng-content select="[body]" />
          </tbody>
        </table>
      </div>
    }
  `,
})
export class TableComponent {
  readonly loading = input(false);
  readonly error = input('');
  readonly empty = input(false);
  readonly density = input<TableDensity>('comfortable');
  readonly retried = output<void>();

  readonly skeletonRows = [0, 1, 2, 3, 4];
  readonly tableClasses = computed(() => `data density-${this.density()}`);
}
