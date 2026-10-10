import { Injectable, inject } from '@angular/core';
import { requests } from '@ragfabric/sdk';
import { Observable } from 'rxjs';

import { SdkHttp } from '../sdk/sdk-http.service';
import { User } from '../models';

@Injectable({ providedIn: 'root' })
export class AdminService {
  private readonly sdk = inject(SdkHttp);

  listUsers(): Observable<User[]> {
    return this.sdk.send(requests.admin.users.list());
  }

  createUser(payload: {
    email: string;
    password: string;
    role: string;
    is_active: boolean;
  }): Observable<User> {
    return this.sdk.send(requests.admin.users.create(payload));
  }

  deleteUser(userId: number): Observable<unknown> {
    return this.sdk.send(requests.admin.users.delete(userId));
  }

  setPermissions(userId: number, changes: { role?: string; is_active?: boolean }): Observable<User> {
    return this.sdk.send(requests.admin.users.setPermissions(userId, changes));
  }
}
