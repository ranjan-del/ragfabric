// Errors raised by the SDK.

/**
 * Turn a FastAPI error body into one readable line, or null when it carries none.
 *
 * The server's own `detail` is preferred over anything invented by a client,
 * because it is the only part of a failed response that knows what went
 * wrong. Validation errors arrive as a list of `{loc, msg, type}`; their
 * messages are joined.
 */
export function describeDetail(body: unknown): string | null {
  if (typeof body !== 'object' || body === null) {
    return typeof body === 'string' && body.trim() !== '' ? body.trim() : null;
  }
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string' && detail.trim() !== '') {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (item as { msg?: unknown } | null)?.msg)
      .filter((msg): msg is string => typeof msg === 'string');
    return messages.length > 0 ? messages.join('. ') : null;
  }
  return null;
}

/**
 * A request the server answered with an error status, or could not be sent.
 *
 * `status` is 0 when the server was never reached. `detail` is the server's
 * own explanation when it gave one, otherwise a line built from the status.
 * `body` is the parsed error body, kept for callers that need more than the
 * one line.
 */
export class RagFabricError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly body: unknown;

  constructor(status: number, body: unknown, statusText = '') {
    const detail =
      describeDetail(body) ??
      (status === 0
        ? 'Could not reach the server.'
        : `Request failed: ${status} ${statusText}`.trim());
    super(detail);
    this.name = 'RagFabricError';
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}
