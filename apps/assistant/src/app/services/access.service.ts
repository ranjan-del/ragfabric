import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { ApiService } from './api.service';
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
  private readonly http = inject(HttpClient);
  private readonly api = inject(ApiService);

  listGroups(): Observable<Group[]> {
    return this.http.get<Group[]>(`${this.api.baseUrl}/admin/groups`);
  }

  createGroup(name: string, description: string): Observable<Group> {
    return this.http.post<Group>(`${this.api.baseUrl}/admin/groups`, { name, description });
  }

  updateGroup(id: number, changes: { name?: string; description?: string }): Observable<Group> {
    return this.http.put<Group>(`${this.api.baseUrl}/admin/groups/${id}`, changes);
  }

  deleteGroup(id: number): Observable<unknown> {
    return this.http.delete(`${this.api.baseUrl}/admin/groups/${id}`);
  }

  listMembers(groupId: number): Observable<User[]> {
    return this.http.get<User[]>(`${this.api.baseUrl}/admin/groups/${groupId}/members`);
  }

  addMember(groupId: number, userId: number): Observable<unknown> {
    return this.http.post(`${this.api.baseUrl}/admin/groups/${groupId}/members`, {
      user_id: userId,
    });
  }

  removeMember(groupId: number, userId: number): Observable<unknown> {
    return this.http.delete(`${this.api.baseUrl}/admin/groups/${groupId}/members/${userId}`);
  }

  listGrants(): Observable<Grant[]> {
    return this.http.get<Grant[]>(`${this.api.baseUrl}/admin/grants`);
  }

  createGrant(
    groupId: number,
    collectionId: number,
    permission: 'read' | 'write',
  ): Observable<Grant> {
    return this.http.post<Grant>(`${this.api.baseUrl}/admin/grants`, {
      group_id: groupId,
      collection_id: collectionId,
      permission,
    });
  }

  deleteGrant(id: number): Observable<unknown> {
    return this.http.delete(`${this.api.baseUrl}/admin/grants/${id}`);
  }

  listKeys(): Observable<ApiKeyItem[]> {
    return this.http.get<ApiKeyItem[]>(`${this.api.baseUrl}/admin/keys`);
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
    return this.http.post<ApiKeyCreated>(`${this.api.baseUrl}/admin/keys`, payload);
  }

  revokeKey(id: number): Observable<unknown> {
    return this.http.delete(`${this.api.baseUrl}/admin/keys/${id}`);
  }
}
