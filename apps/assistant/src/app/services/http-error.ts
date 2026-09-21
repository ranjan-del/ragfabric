import { HttpErrorResponse } from '@angular/common/http';

/**
 * Turn a failed request into something an operator can act on.
 *
 * The server's own `detail` is preferred over anything invented here, because
 * it is the only part of the response that knows what actually went wrong.
 * "Something went wrong" is the last resort, not the first answer.
 */
export function describeError(error: unknown, fallback = 'Something went wrong.'): string {
  if (!(error instanceof HttpErrorResponse)) {
    return fallback;
  }
  const detail = (error.error as { detail?: unknown } | null)?.detail;
  if (typeof detail === 'string' && detail.trim() !== '') {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    // FastAPI validation errors: a list of {loc, msg, type}.
    const messages = detail
      .map((item) => (item as { msg?: unknown }).msg)
      .filter((msg): msg is string => typeof msg === 'string');
    if (messages.length > 0) {
      return messages.join('. ');
    }
  }
  if (error.status === 0) {
    return 'Could not reach the server.';
  }
  if (error.status === 403) {
    return 'You do not have permission to do that.';
  }
  return `Request failed: ${error.status} ${error.statusText}`.trim();
}
