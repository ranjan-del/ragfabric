import { RATE_LIMIT_MAX, parseRateLimit } from './rate-limit';

describe('parseRateLimit', () => {
  it('test_empty_or_blank_means_the_default_and_is_not_a_number', () => {
    expect(parseRateLimit('')).toEqual({ ok: true, value: null });
    expect(parseRateLimit('   ')).toEqual({ ok: true, value: null });
  });

  it('test_whole_numbers_from_1_to_the_server_maximum_are_accepted', () => {
    expect(parseRateLimit('1')).toEqual({ ok: true, value: 1 });
    expect(parseRateLimit(' 250 ')).toEqual({ ok: true, value: 250 });
    expect(parseRateLimit(String(RATE_LIMIT_MAX))).toEqual({ ok: true, value: RATE_LIMIT_MAX });
  });

  it('test_zero_negative_fractional_exponent_and_text_are_refused', () => {
    for (const bad of ['0', '00', '-1', '1.5', '1e3', '+4', 'ten', String(RATE_LIMIT_MAX + 1)]) {
      expect(parseRateLimit(bad).ok).withContext(bad).toBeFalse();
    }
  });
});
