// describeError over the SDK's RagFabricError (Phase 9, Task 5).

import { RagFabricError } from '@ragfabric/sdk';

import { describeError } from './http-error';

describe('describeError', () => {
  it('test_prefers_the_servers_own_detail', () => {
    expect(describeError(new RagFabricError(409, { detail: 'Name taken.' }))).toBe('Name taken.');
  });

  it('test_joins_validation_messages', () => {
    const error = new RagFabricError(422, { detail: [{ msg: 'too short' }, { msg: 'bad' }] });
    expect(describeError(error)).toBe('too short. bad');
  });

  it('test_explains_an_unreachable_server_and_a_forbidden_call', () => {
    expect(describeError(new RagFabricError(0, null))).toBe('Could not reach the server.');
    expect(describeError(new RagFabricError(403, null))).toBe('You do not have permission to do that.');
  });

  it('test_falls_back_for_anything_that_is_not_an_api_error', () => {
    expect(describeError(new Error('boom'), 'Upload failed.')).toBe('Upload failed.');
  });
});
