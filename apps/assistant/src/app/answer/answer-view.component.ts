import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { Citation } from '@ragfabric/sdk';

import { AnswerState } from './answer-state';
import { answerSegments } from './segments';

/**
 * A streamed answer with clickable citation markers and source chips.
 *
 * Each `[n]` in the text and each chip is a button: clicking one emits the
 * citation, and the page opens the source viewer on it.
 */
@Component({
  selector: 'app-answer-view',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @let s = state();
    @if (s.superseded(); as sup) {
      <div class="alert alert-warning superseded" role="status" data-test="superseded">
        The streamed answer was corrected by the server ({{ sup.reason }}). What you see is the
        answer it recorded.
        @if ((sup.dropped_claims?.length ?? 0) + (sup.dropped_relationship_claims?.length ?? 0) > 0) {
          <details>
            <summary>Removed claims</summary>
            <ul>
              @for (claim of sup.dropped_claims ?? []; track $index) {
                <li>{{ claim.text }} <span class="muted">({{ claim.reason }})</span></li>
              }
              @for (claim of sup.dropped_relationship_claims ?? []; track $index) {
                <li>{{ claim.text }} <span class="muted">({{ claim.reason }})</span></li>
              }
            </ul>
          </details>
        }
      </div>
    }
    <p class="answer-text" data-test="answer" aria-live="polite">
      @for (part of segments(); track $index) {
        @if (part.kind === 'text') {<span>{{ part.text }}</span>} @else {
          <button
            type="button"
            class="marker"
            [attr.aria-label]="'Open source ' + part.marker"
            (click)="cite.emit(part.citation)"
          >{{ part.marker }}</button>
        }
      }
      @if (s.status() === 'running') {<span class="caret" aria-hidden="true"></span>}
    </p>
    @if (s.status() === 'failed') {
      <p class="alert alert-error" role="alert" data-test="error">{{ s.error() }}</p>
    }
    @if (s.citations().length > 0) {
      <div class="chips" aria-label="Sources cited">
        @for (c of cited(); track c.marker) {
          <button type="button" class="chip" (click)="cite.emit(c)" [attr.data-test]="'chip-' + c.marker">
            <span class="chip-marker">{{ c.marker }}</span>
            {{ c.filename ?? 'unknown source' }}@if (c.page != null) { p{{ c.page }}}
          </button>
        }
      </div>
      @if (uncited().length > 0) {
        <details class="uncited">
          <summary>{{ uncited().length }} more retrieved, not cited</summary>
          <div class="chips">
            @for (c of uncited(); track c.marker) {
              <button type="button" class="chip unused" (click)="cite.emit(c)" [attr.data-test]="'chip-' + c.marker">
                <span class="chip-marker">{{ c.marker }}</span>
                {{ c.filename ?? 'unknown source' }}@if (c.page != null) { p{{ c.page }}}
              </button>
            }
          </div>
        </details>
      }
    }
  `,
  styles: `
    .answer-text { white-space: pre-wrap; line-height: 1.6; margin: 0; }
    .marker { border: 0; background: var(--brand-soft); color: var(--brand-700); border-radius: 6px;
      padding: 0 0.3rem; margin: 0 0.1rem; font: inherit; font-size: 0.85em; cursor: pointer; }
    .marker:focus-visible, .chip:focus-visible { outline: none; box-shadow: var(--focus-ring); }
    .caret { display: inline-block; width: 0.5ch; height: 1em; background: var(--text-muted);
      vertical-align: text-bottom; animation: blink 1s steps(2) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .chips { display: flex; flex-wrap: wrap; gap: var(--space-2); margin-top: var(--space-3); }
    .chip { border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
      border-radius: 999px; padding: 0.2rem 0.7rem; font: inherit; font-size: 0.85rem; cursor: pointer; }
    .chip.unused { opacity: 0.75; }
    .chip-marker { font-weight: 600; margin-right: 0.25rem; }
    .superseded { margin-bottom: var(--space-3); }
    .uncited { margin-top: var(--space-2); font-size: 0.85rem; color: var(--text-muted); }
    .uncited summary { cursor: pointer; }
  `,
})
export class AnswerViewComponent {
  readonly state = input.required<AnswerState>();
  readonly cite = output<Citation>();

  readonly segments = computed(() => answerSegments(this.state().text(), this.state().citations()));
  /** The sources the answer cites, then (folded away) the ones only retrieved. */
  readonly cited = computed(() => this.state().citations().filter((c) => c.used));
  readonly uncited = computed(() => this.state().citations().filter((c) => !c.used));
}
