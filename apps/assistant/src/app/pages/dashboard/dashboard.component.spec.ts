import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { AnalyticsOverview, UsageStats } from '../../models';
import { AnalyticsService } from '../../services/analytics.service';
import { AuthService } from '../../services/auth.service';
import { DashboardComponent } from './dashboard.component';

// Found with Playwright at 390 px: one long recent question made the "Recent
// questions" card 728 px wide. The body clips horizontal overflow, so the card
// ran off the screen and every row's confidence badge was cut away unseen.

const LONG = 'Who does Asha Rao report to, and who does that person report to in turn, and since when?';

const overview = {
  documents: 3,
  ready_documents: 3,
  collections: 0,
  chunks: 6,
  queries: 31,
} as unknown as AnalyticsOverview;

const usage = {
  recent_queries: [{ question: LONG, confidence: 0.42 }],
  top_documents: [{ document_id: 1, filename: `${'very-long-file-name-'.repeat(6)}.md`, chunks: 2 }],
} as unknown as UsageStats;

describe('DashboardComponent on a narrow screen', () => {
  let fixture: ComponentFixture<DashboardComponent>;
  let host: HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [
        provideRouter([]),
        { provide: AnalyticsService, useValue: { overview: () => of(overview), usage: () => of(usage) } },
        { provide: AuthService, useValue: { user: () => ({ email: 'admin@example.com' }) } },
      ],
    }).compileComponents();
    fixture = TestBed.createComponent(DashboardComponent);
    host = fixture.nativeElement as HTMLElement;
    host.style.display = 'block';
    host.style.width = '350px';
    document.body.appendChild(host);
    fixture.detectChanges();
  });

  afterEach(() => host.remove());

  it('keeps every card inside the page width, however long a question is', () => {
    const limit = host.getBoundingClientRect().right + 1;
    for (const card of Array.from(host.querySelectorAll<HTMLElement>('.card'))) {
      expect(card.getBoundingClientRect().right).withContext(card.className).toBeLessThanOrEqual(limit);
    }
  });

  it('keeps the confidence badge of a long question on screen', () => {
    const badge = host.querySelector<HTMLElement>('.list .badge-brand');
    expect(badge).not.toBeNull();
    expect(badge!.getBoundingClientRect().right).toBeLessThanOrEqual(host.getBoundingClientRect().right + 1);
  });

  it('still shortens the long question with an ellipsis rather than wrapping it', () => {
    const text = host.querySelector<HTMLElement>('.q-text')!;
    expect(getComputedStyle(text).textOverflow).toBe('ellipsis');
    expect(text.scrollWidth).toBeGreaterThan(text.clientWidth);
  });
});
