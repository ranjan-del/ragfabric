import { Injectable, inject } from '@angular/core';
import { requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';
import { ApiKeyCreated, ApiKeyItem, Grant, Group, User } from '../models';

/**
 * Groups, membership, grants and API keys: everything under /api/admin that
 * decides who can read what.
 *
 * One service rather than four because every screen in the console's access
 * section needs at least two of them at once (a grant is meaningless without
 * its group, membership is meaningless without its users), and splitting them
 * would only mean each screen injecting the same set anyway.
 */
@Injectable({ providedIn: 'root' })
export class AccessService {
  private readonly sdk = inject(SdkHttp);

  listGroups(): Observable<Group[]> {
    return this.sdk.send(requests.admin.groups.list());
  }

  createGroup(name: string, description: string): Observable<Group> {
    return this.sdk.send(requests.admin.groups.create(name, description));
  }

  updateGroup(id: number, changes: { name?: string; description?: string }): Observable<Group> {
    return this.sdk.send(requests.admin.groups.update(id, changes));
  }

  deleteGroup(id: number): Observable<unknown> {
    return this.sdk.send(requests.admin.groups.delete(id));
  }

  listMembers(groupId: number): Observable<User[]> {
    return this.sdk.send(requests.admin.groups.members(groupId));
  }

  addMember(groupId: number, userId: number): Observable<unknown> {
    return this.sdk.send(requests.admin.groups.addMember(groupId, userId));
  }

  removeMember(groupId: number, userId: number): Observable<unknown> {
    return this.sdk.send(requests.admin.groups.removeMember(groupId, userId));
  }

  listGrants(): Observable<Grant[]> {
    return this.sdk.send(requests.admin.grants.list());
  }

  createGrant(
    groupId: number,
    collectionId: number,
    permission: 'read' | 'write',
  ): Observable<Grant> {
    return this.sdk.send(requests.admin.grants.create(groupId, collectionId, permission));
  }

  deleteGrant(id: number): Observable<unknown> {
    return this.sdk.send(requests.admin.grants.delete(id));
  }

  listKeys(): Observable<ApiKeyItem[]> {
    return this.sdk.send(requests.admin.keys.list());
  }

  /**
   * Create an API key. The plaintext secret is in this response and in no
   * other response, ever: the server stores only its hash. Callers must hand
   * it straight to the one time panel and keep it out of any store.
   */
  createKey(payload: {
    name: string;
    user_id: number;
    collection_ids: number[];
    strategies: string[];
    expires_at: string | null;
  }): Observable<ApiKeyCreated> {
    return this.sdk.send(requests.admin.keys.create(payload));
  }

  revokeKey(id: number): Observable<unknown> {
    return this.sdk.send(requests.admin.keys.revoke(id));
  }
}
