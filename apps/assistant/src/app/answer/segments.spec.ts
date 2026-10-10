// Splitting an answer into text and clickable markers, and marking a snippet.

import { Citation } from '@ragfabric/sdk';

import { answerSegments, snippetSegments } from './segments';

const C1: Citation = { marker: '[1]', snippet: 'a', used: true };
const C2: Citation = { marker: '[2]', snippet: 'b', used: true };

describe('answerSegments', () => {
  it('test_markers_become_citation_segments_in_order', () => {
    const parts = answerSegments('Leave is 24 days [1][2]. Ask HR.', [C1, C2]);

    expect(parts.map((p) => (p.kind === 'text' ? p.text : p.marker))).toEqual([
      'Leave is 24 days ',
      '[1]',
      '[2]',
      '. Ask HR.',
    ]);
    expect(parts[1]).toEqual({ kind: 'marker', marker: '[1]', citation: C1 });
  });

  it('test_a_marker_with_no_citation_stays_plain_text', () => {
    const parts = answerSegments('See [3].', [C1]);
    expect(parts).toEqual([{ kind: 'text', text: 'See [3].' }]);
  });
});

describe('snippetSegments', () => {
  it('test_marks_query_terms_and_the_quoted_sentence', () => {
    const snippet = 'Staff get 24 days. Carry over is 5.';
    const parts = snippetSegments(snippet, [{ term: 'days', start: 13, end: 17 }], {
      text: 'Staff get 24 days.',
      start: 0,
      end: 18,
    });

    expect(parts.map((p) => p.text).join('')).toBe(snippet);
    expect(parts).toEqual([
      { text: 'Staff get 24 ', highlight: false, support: true },
      { text: 'days', highlight: true, support: true },
      { text: '.', highlight: false, support: true },
      { text: ' Carry over is 5.', highlight: false, support: false },
    ]);
  });

  it('test_out_of_range_spans_are_clamped_never_thrown', () => {
    const parts = snippetSegments('abc', [{ term: 'x', start: 2, end: 99 }], null);
    expect(parts.map((p) => p.text).join('')).toBe('abc');
  });
});
