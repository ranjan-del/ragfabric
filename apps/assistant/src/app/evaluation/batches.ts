import { EvalMetrics, EvalRun, QualityPoint } from '@ragfabric/sdk';

/** Batch labels, most recent first (by the latest run started in each). */
export function batchNames(runs: readonly EvalRun[]): string[] {
  const latest = new Map<string, string>();
  for (const run of runs) {
    const started = run.started_at ?? '';
    if ((latest.get(run.batch) ?? '') <= started) {
      latest.set(run.batch, started);
    }
  }
  return [...latest.entries()].sort((a, b) => b[1].localeCompare(a[1])).map(([name]) => name);
}

export function runsOf(runs: readonly EvalRun[], batch: string): EvalRun[] {
  return runs.filter((r) => r.batch === batch).sort((a, b) => a.target.localeCompare(b.target));
}

/** The metric columns, in the order the report prints them. */
export const METRICS: readonly { key: keyof EvalMetrics; label: string }[] = [
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'hit', label: 'Hit rate' },
  { key: 'reciprocal_rank', label: 'MRR' },
  { key: 'correctness', label: 'Correctness' },
  { key: 'faithfulness', label: 'Faithfulness' },
  { key: 'context_relevance', label: 'Context relevance' },
  { key: 'citation_correct', label: 'Citation correctness' },
];

/**
 * An SVG polyline for a quality trend on a 0 to 1 scale. A null point breaks
 * the line rather than dropping it to zero.
 */
export function trendPath(points: readonly QualityPoint[], width: number, height: number): string {
  if (points.length === 0) {
    return '';
  }
  const step = points.length === 1 ? 0 : width / (points.length - 1);
  let path = '';
  let drawing = false;
  points.forEach((point, i) => {
    if (point.correctness == null) {
      drawing = false;
      return;
    }
    const x = (i * step).toFixed(1);
    const y = (height - point.correctness * height).toFixed(1);
    path += `${drawing ? 'L' : 'M'}${x},${y} `;
    drawing = true;
  });
  return path.trim();
}
