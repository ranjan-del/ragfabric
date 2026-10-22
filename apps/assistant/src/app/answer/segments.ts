import { Citation, Highlight, Span } from '@ragfabric/sdk';

import { WalkedEdge } from './graph';

export type AnswerSegment =
  | { kind: 'text'; text: string }
  | { kind: 'marker'; marker: string; citation: Citation }
  | { kind: 'edge'; marker: string; edge: WalkedEdge };

/** `[3]` passage markers and `[E 3]` edge markers. The server reads `[E3]` and `[e 3]` the same way. */
const MARKER = /\[(?:(\d+)|E\s*(\d+))\]/gi;

/**
 * Split an answer into plain text and citation markers.
 *
 * A marker is clickable only when the server sent a citation with that
 * marker, or (for `[E k]`) the walked sub graph has a k-th edge; anything else
 * that looks like a marker stays text, so a click never opens a source or an
 * edge the answer does not have.
 */
export function answerSegments(
  text: string,
  citations: readonly Citation[],
  edges: readonly WalkedEdge[] = [],
): AnswerSegment[] {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const byK = new Map(edges.map((e) => [e.k, e]));
  const out: AnswerSegment[] = [];
  let plain = '';
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const citation = match[1] === undefined ? undefined : byMarker.get(match[0]);
    const edge = match[2] === undefined ? undefined : byK.get(Number(match[2]));
    if (citation === undefined && edge === undefined) {
      continue;
    }
    plain += text.slice(last, match.index);
    if (plain !== '') {
      out.push({ kind: 'text', text: plain });
      plain = '';
    }
    if (citation !== undefined) {
      out.push({ kind: 'marker', marker: match[0], citation });
    } else if (edge !== undefined) {
      out.push({ kind: 'edge', marker: match[0], edge });
    }
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

export interface ExpandedSnippet {
  text: string;
  highlights: Highlight[];
  support: Span | null;
}

const ELLIPSIS = '...';

/** Whether the server cut this snippet out of a longer chunk (it marks each cut with "..."). */
export function isTruncated(snippet: string): boolean {
  return snippet.startsWith(ELLIPSIS) || snippet.endsWith(ELLIPSIS);
}

/**
 * Widen a cut snippet using the original file's text.
 *
 * The server's offsets are relative to the snippet it sent, so the snippet is
 * located in `full` and every offset moves by the same amount. Returns null
 * when the snippet is not in the file (a converted format, or a changed file),
 * so the caller can say so instead of showing text that is not the source.
 */
export function expandSnippet(
  snippet: string,
  full: string,
  highlights: readonly Highlight[] = [],
  support: Span | null | undefined = null,
  radius = 600,
): ExpandedSnippet | null {
  const lead = snippet.startsWith(ELLIPSIS) ? ELLIPSIS.length : 0;
  const trail = snippet.endsWith(ELLIPSIS) ? ELLIPSIS.length : 0;
  const body = snippet.slice(lead, snippet.length - trail);
  const at = body === '' ? -1 : full.indexOf(body);
  if (at < 0) {
    return null;
  }
  const from = Math.max(0, at - radius);
  const to = Math.min(full.length, at + body.length + radius);
  const before = from > 0 ? ELLIPSIS : '';
  const after = to < full.length ? ELLIPSIS : '';
  // snippet offset -> expanded offset
  const shift = before.length + (at - from) - lead;
  return {
    text: before + full.slice(from, to) + after,
    highlights: highlights.map((h) => ({ ...h, start: h.start + shift, end: h.end + shift })),
    support: support ? { ...support, start: support.start + shift, end: support.end + shift } : null,
  };
}
