import { RagFabricError, describeDetail } from '@ragfabric/sdk';

/**
 * Turn a failed request into something an operator can act on.
 *
 * The server's own `detail` is preferred over anything invented here, because
 * it is the only part of the response that knows what actually went wrong.
 * "Something went wrong" is the last resort, not the first answer. Every
 * request goes through @ragfabric/sdk, so a failed one is a RagFabricError.
 */
export function describeError(error: unknown, fallback = 'Something went wrong.'): string {
  if (!(error instanceof RagFabricError)) {
    return fallback;
  }
  const detail = describeDetail(error.body);
  if (detail !== null) {
    return detail;
  }
  if (error.status === 0) {
    return 'Could not reach the server.';
  }
  if (error.status === 403) {
    return 'You do not have permission to do that.';
  }
  return error.detail;
}
