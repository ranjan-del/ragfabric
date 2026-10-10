import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

export interface BarRow {
  label: string;
  value: number | null | undefined;
  /** Shown after the value, e.g. "3 unpriced". */
  note?: string;
}

/**
 * Horizontal bars in plain HTML and CSS (decision D19: no chart library).
 *
 * A null value draws no bar and says "n/a": an unmeasured number is never
 * drawn as a zero length bar that looks measured (ADR 0004). The label and
 * the value are text, so the chart reads the same without its colours.
 */
@Component({
  selector: 'ui-bar-chart',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <ul class="bars" [attr.aria-label]="label()">
      @for (row of scaled(); track row.label) {
        <li class="row" data-test="bar-row">
          <span class="label">{{ row.label }}</span>
          <span class="track">
            @if (row.width !== null) {<span class="bar" [style.width.%]="row.width"></span>}
          </span>
          <span class="value">{{ row.text }}@if (row.note) {<span class="muted"> {{ row.note }}</span>}</span>
        </li>
      }
    </ul>
  `,
  styles: `
    .bars { list-style: none; margin: 0; padding: 0; display: grid; gap: var(--space-2); }
    .row { display: grid; grid-template-columns: minmax(6rem, 9rem) 1fr minmax(5rem, auto); gap: var(--space-2); align-items: center; font-size: 0.85rem; }
    .label { overflow-wrap: anywhere; }
    .track { height: 10px; background: var(--surface-2); border-radius: 5px; overflow: hidden; }
    .bar { display: block; height: 100%; background: var(--brand); border-radius: 5px; }
    .value { text-align: right; font-variant-numeric: tabular-nums; }
  `,
})
export class BarChartComponent {
  readonly rows = input<readonly BarRow[]>([]);
  readonly label = input('');
  readonly format = input<(value: number) => string>((v) => String(v));

  readonly scaled = computed(() => {
    const rows = this.rows();
    const max = Math.max(0, ...rows.map((r) => r.value ?? 0));
    return rows.map((r) => ({
      label: r.label,
      note: r.note,
      width: r.value == null ? null : max === 0 ? 0 : (r.value / max) * 100,
      text: r.value == null ? 'n/a' : this.format()(r.value),
    }));
  });
}
