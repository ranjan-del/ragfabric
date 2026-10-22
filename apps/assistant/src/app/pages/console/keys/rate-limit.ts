/** The server accepts 1 to 100000 requests per minute for a key (`ApiKeyCreate`). */
export const RATE_LIMIT_MAX = 100000;

export type RateLimitInput = { ok: true; value: number | null } | { ok: false; error: string };

/**
 * Read the rate limit field. Empty means "use the server default" and is sent
 * as no value at all, never as 0 or as a number the page made up. Anything but
 * a whole number from 1 to RATE_LIMIT_MAX is refused with a reason.
 */
export function parseRateLimit(text: string): RateLimitInput {
  const trimmed = text.trim();
  if (trimmed === '') {
    return { ok: true, value: null };
  }
  if (!/^\d+$/.test(trimmed)) {
    return { ok: false, error: `Enter a whole number from 1 to ${RATE_LIMIT_MAX.toLocaleString('en-GB')}, or leave it empty for the default.` };
  }
  const value = Number(trimmed);
  if (value < 1 || value > RATE_LIMIT_MAX) {
    return { ok: false, error: `The limit must be from 1 to ${RATE_LIMIT_MAX.toLocaleString('en-GB')} requests per minute.` };
  }
  return { ok: true, value };
}
