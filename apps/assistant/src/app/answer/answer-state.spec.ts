// One streamed answer's state, shared by Ask and every Compare column (Task 6).

import { AskEvent } from '@ragfabric/sdk';

import { AnswerState } from './answer-state';

const RETRIEVAL: AskEvent = {
  event: 'retrieval',
  data: {
    chunks: 2,
    strategy: 'vectorless',
    trace: [],
    sub_questions: [],
    subgraph: null,
    router: null,
    fallback_from: null,
  },
};

describe('AnswerState', () => {
  it('test_tokens_append_and_done_finishes', () => {
    const state = new AnswerState();
    state.start();
    expect(state.status()).toBe('running');

    state.apply(RETRIEVAL);
    state.apply({ event: 'token', data: { text: 'Leave is ' } });
    state.apply({ event: 'token', data: { text: '24 days [1].' } });
    state.apply({ event: 'citations', data: { citations: [{ marker: '[1]', snippet: 's', used: true }] } });
    state.apply({ event: 'done', data: { run_id: 7, latency_ms: 40, usage: { llm_calls: 1 } } });

    expect(state.text()).toBe('Leave is 24 days [1].');
    expect(state.retrieval()?.strategy).toBe('vectorless');
    expect(state.citations().length).toBe(1);
    expect(state.done()?.run_id).toBe(7);
    expect(state.status()).toBe('done');
  });

  it('test_superseded_replaces_the_drawn_text_and_keeps_the_reason', () => {
    const state = new AnswerState();
    state.start();
    state.apply({ event: 'token', data: { text: 'An unsupported claim.' } });
    state.apply({
      event: 'superseded',
      data: {
        text: 'Leave is 24 days [1].',
        reason: 'unsupported claims removed',
        dropped_claims: [{ text: 'An unsupported claim.', reason: 'no_support' }],
      },
    });

    expect(state.text()).toBe('Leave is 24 days [1].');
    expect(state.superseded()?.reason).toBe('unsupported claims removed');
    expect(state.superseded()?.dropped_claims?.length).toBe(1);
  });

  it('test_a_stream_that_ends_without_done_is_failed_not_final', () => {
    const state = new AnswerState();
    state.start();
    state.apply({ event: 'token', data: { text: 'Half an ans' } });
    state.complete();

    expect(state.status()).toBe('failed');
    expect(state.error()).toContain('ended before the answer finished');
    expect(state.text()).toBe('Half an ans');
  });

  it('test_fail_records_the_message_and_start_clears_everything', () => {
    const state = new AnswerState();
    state.start();
    state.apply({ event: 'token', data: { text: 'x' } });
    state.fail('Agentic needs a model.');
    expect(state.status()).toBe('failed');
    expect(state.error()).toBe('Agentic needs a model.');

    state.start();
    expect(state.text()).toBe('');
    expect(state.error()).toBeNull();
    expect(state.superseded()).toBeNull();
  });
});
