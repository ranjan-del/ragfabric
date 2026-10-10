// The sign-in page must never print credentials.
//
// It used to show "Demo admin: admin@example.com / adminpass123" on every
// deployment: wrong whenever the admin password had been changed, and a working
// admin password published to anyone who opened the page whenever it had not.
// quickstart writes a random admin password, so the hint was usually wrong too.
// The development defaults live in .env.example, and the server logs a warning
// at startup while they are in use; the page says nothing.
import { provideHttpClient, withXhr } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { LoginComponent } from './login.component';

describe('LoginComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [LoginComponent],
      providers: [provideHttpClient(withXhr()), provideHttpClientTesting(), provideRouter([])],
    });
  });

  it('shows no credentials in sign-in or register mode', () => {
    const fixture = TestBed.createComponent(LoginComponent);
    fixture.detectChanges();
    const page = (): string => (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(page()).not.toContain('adminpass123');
    expect(page()).not.toContain('admin@example.com');
    expect(page()).not.toContain('Demo admin');

    fixture.componentInstance.switchMode('register');
    fixture.detectChanges();
    expect(page()).not.toContain('adminpass123');
  });
});
