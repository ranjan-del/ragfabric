import { TraceSpan } from '@ragfabric/sdk';

export interface WaterfallBar {
  span: TraceSpan;
  /** Percent of the run's width. */
  left: number;
  width: number;
}

function isSpan(value: unknown): value is TraceSpan {
  const v = value as Partial<TraceSpan> | null;
  return (
    typeof v === 'object' && v !== null && typeof v.name === 'string' &&
    typeof v.started_ms === 'number' && typeof v.duration_ms === 'number'
  );
}

/**
 * Place each stored span on one time axis, in start order.
 *
 * The axis runs to the later of the run's latency and the last span's end, so
 * a span is never drawn past the edge. A span that is not shaped like one is
 * left out rather than guessed at; zero length spans get a sliver of width so
 * they stay visible.
 */
export function waterfall(trace: readonly unknown[], latencyMs: number): WaterfallBar[] {
  const spans = trace.filter(isSpan).sort((a, b) => a.started_ms - b.started_ms);
  const end = Math.max(latencyMs, ...spans.map((s) => s.started_ms + s.duration_ms), 1);
  return spans.map((span) => ({
    span,
    left: (span.started_ms / end) * 100,
    width: Math.max((span.duration_ms / end) * 100, 0.5),
  }));
}
