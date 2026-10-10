import { Citation, Highlight, Span } from '@ragfabric/sdk';

export type AnswerSegment =
  | { kind: 'text'; text: string }
  | { kind: 'marker'; marker: string; citation: Citation };

/**
 * Split an answer into plain text and citation markers.
 *
 * A marker is clickable only when the server sent a citation with that
 * marker; anything else that looks like `[n]` stays text, so a click never
 * opens a source the answer does not have.
 */
export function answerSegments(text: string, citations: readonly Citation[]): AnswerSegment[] {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const out: AnswerSegment[] = [];
  let plain = '';
  let last = 0;
  for (const match of text.matchAll(/\[\d+\]/g)) {
    const citation = byMarker.get(match[0]);
    if (citation === undefined) {
      continue;
    }
    plain += text.slice(last, match.index);
    if (plain !== '') {
      out.push({ kind: 'text', text: plain });
      plain = '';
    }
    out.push({ kind: 'marker', marker: match[0], citation });
    last = (match.index ?? 0) + match[0].length;
  }
  plain += text.slice(last);
  if (plain !== '') {
    out.push({ kind: 'text', text: plain });
  }
  return out;
}

export interface SnippetSegment {
  text: string;
  /** Inside a query term the server highlighted. */
  highlight: boolean;
  /** Inside the sentence the answer quoted from this chunk. */
  support: boolean;
}

/** Cut a snippet at every highlight and supporting span boundary. Offsets are the server's. */
export function snippetSegments(
  snippet: string,
  highlights: readonly Highlight[] = [],
  support: Span | null | undefined = null,
): SnippetSegment[] {
  const clamp = (n: number) => Math.max(0, Math.min(snippet.length, n));
  const ranges = highlights.map((h) => [clamp(h.start), clamp(h.end)] as const);
  const supportRange = support ? ([clamp(support.start), clamp(support.end)] as const) : null;
  const cuts = new Set<number>([0, snippet.length]);
  for (const [start, end] of ranges) {
    cuts.add(start);
    cuts.add(end);
  }
  if (supportRange) {
    cuts.add(supportRange[0]);
    cuts.add(supportRange[1]);
  }
  const points = [...cuts].sort((a, b) => a - b);
  const out: SnippetSegment[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const start = points[i] as number;
    const end = points[i + 1] as number;
    if (end <= start) {
      continue;
    }
    const segment = {
      text: snippet.slice(start, end),
      highlight: ranges.some(([s, e]) => start >= s && end <= e),
      support: supportRange !== null && start >= supportRange[0] && end <= supportRange[1],
    };
    const previous = out[out.length - 1];
    if (previous && previous.highlight === segment.highlight && previous.support === segment.support) {
      previous.text += segment.text;
    } else {
      out.push(segment);
    }
  }
  return out;
}
