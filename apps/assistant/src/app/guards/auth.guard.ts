import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { catchError, map, of } from 'rxjs';

import { AuthService } from '../services/auth.service';

/** Allow the route only when a token is present; otherwise send to /login. */
export const authGuard: CanActivateFn = () => {
  const auth = inject(AuthService);
  const router = inject(Router);
  if (auth.token) {
    return true;
  }
  return router.createUrlTree(['/login']);
};

/**
 * Allow the route only for admins; regular users are sent to the dashboard.
 *
 * An admin route opened directly (a reload, a pasted link) runs before the
 * profile has loaded, when nobody looks like an admin yet. With a token and no
 * profile, the guard waits for the profile instead of sending an admin away.
 */
export const adminGuard: CanActivateFn = () => {
  const auth = inject(AuthService);
  const router = inject(Router);
  const dashboard = router.createUrlTree(['/dashboard']);
  if (auth.isAdmin()) {
    return true;
  }
  if (auth.token && auth.user?.() == null) {
    return auth.loadProfile().pipe(
      map((user) => (user.role === 'admin' ? true : dashboard)),
      catchError(() => of(router.createUrlTree(['/login']))),
    );
  }
  return dashboard;
};
