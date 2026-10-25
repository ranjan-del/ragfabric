import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { AnalyticsOverview, UsageStats } from '../../models';
import { AnalyticsService } from '../../services/analytics.service';
import { AuthService } from '../../services/auth.service';
import { DashboardComponent } from './dashboard.component';

// Found with Playwright in the dark theme: the "Ask your knowledge base" card
// faded to a hard coded white, so its light grey dark-theme text sat on white,
// about 2:1 at the right hand end. Every colour the card's background passes
// through must hold the text to WCAG AA (4.5:1) in both themes.

function colours(value: string): Array<[number, number, number]> {
  return Array.from(value.matchAll(/rgba?\(([^)]+)\)/g)).map((m) => {
    const [r, g, b] = (m[1] as string).split(/[ ,/]+/).filter(Boolean).map(Number);
    return [r as number, g as number, b as number];
  });
}

function luminance([r, g, b]: [number, number, number]): number {
  const f = (v: number) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}

function contrast(a: [number, number, number], b: [number, number, number]): number {
  const [x, y] = [luminance(a), luminance(b)];
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

describe('DashboardComponent call to action card', () => {
  afterEach(() => document.documentElement.removeAttribute('data-theme'));

  for (const theme of ['light', 'dark']) {
    it(`keeps its heading and text at least 4.5 to 1 across its background in ${theme}`, async () => {
      document.documentElement.setAttribute('data-theme', theme);
      await TestBed.configureTestingModule({
        imports: [DashboardComponent],
        providers: [
          provideRouter([]),
          {
            provide: AnalyticsService,
            useValue: {
              overview: () => of({ documents: 1, ready_documents: 1, collections: 0, chunks: 2, queries: 0 } as unknown as AnalyticsOverview),
              usage: () => of({ recent_queries: [], top_documents: [] } as unknown as UsageStats),
            },
          },
          { provide: AuthService, useValue: { user: () => null } },
        ],
      }).compileComponents();
      const fixture = TestBed.createComponent(DashboardComponent);
      document.body.appendChild(fixture.nativeElement);
      fixture.detectChanges();
      try {
        const card = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>('.cta')!;
        const stops = colours(getComputedStyle(card).backgroundImage);
        expect(stops.length).toBeGreaterThan(1);
        for (const selector of ['h3', 'p']) {
          const text = colours(getComputedStyle(card.querySelector(selector)!).color)[0]!;
          for (const stop of stops) {
            expect(contrast(text, stop)).withContext(`${selector} on rgb(${stop})`).toBeGreaterThanOrEqual(4.5);
          }
        }
      } finally {
        (fixture.nativeElement as HTMLElement).remove();
      }
    });
  }
});
