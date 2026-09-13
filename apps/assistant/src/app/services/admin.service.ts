import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiService } from './api.service';
import { User } from '../models';

@Injectable({ providedIn: 'root' })
export class AdminService {
  constructor(private http: HttpClient, private api: ApiService) {}

  listUsers(): Observable<User[]> {
    return this.http.get<User[]>(`${this.api.baseUrl}/admin/users`);
  }

  setPermissions(
    userId: number,
    changes: { role?: string; is_active?: boolean }
  ): Observable<User> {
    return this.http.put<User>(
      `${this.api.baseUrl}/admin/users/${userId}/permissions`,
      changes
    );
  }
}
