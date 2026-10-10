import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { RetrievalEvent } from '@ragfabric/sdk';

import { percent } from './format';

/**
 * Why this strategy answered.
 *
 * Under AUTO it shows the router's decision as the server reported it: the
 * strategy, whether a rule or the classifier decided, the classifier's
 * confidence (a rule has none, and none is invented), the reasoning and the
 * router's own estimates. A fallback is always called out, because the
 * strategy that answered is then not the one that was chosen.
 */
@Component({
  selector: 'app-router-card',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @let r = retrieval();
    <section class="router-card card" aria-label="Routing decision">
      @if (r === null) {
        <p class="muted">Retrieving...</p>
      } @else {
        <div class="router-head">
          <span class="muted">Answered by</span>
          <strong class="strategy" data-test="strategy">{{ r.strategy }}</strong>
          @if (mode() === 'manual') {
            <span class="badge badge-neutral">chosen by you</span>
          } @else if (r.router) {
            <span class="badge badge-brand" data-test="source">{{ sourceLabel() }}</span>
          }
        </div>
        @if (r.fallback_from) {
          <p class="fallback alert alert-warning" data-test="fallback">
            The router chose <strong>{{ r.fallback_from }}</strong>, which found nothing, so
            <strong>{{ r.strategy }}</strong> answered instead.
          </p>
        }
        @if (mode() === 'auto') {
          @if (r.router; as d) {
            <dl class="router-facts">
              <div>
                <dt>Confidence</dt>
                <dd data-test="confidence">{{ confidence() }}</dd>
              </div>
              <div><dt>Query type</dt><dd>{{ d.query_type }}</dd></div>
              <div><dt>Complexity</dt><dd>{{ d.estimated_complexity }}</dd></div>
              <div><dt>Expected cost</dt><dd>{{ d.expected_cost_level }}</dd></div>
              <div><dt>Expected latency</dt><dd>{{ d.expected_latency_level }}</dd></div>
            </dl>
            <p class="reasoning" data-test="reasoning">{{ d.reasoning }}</p>
          } @else {
            <p class="muted" data-test="no-decision">The server did not report a routing decision.</p>
          }
        }
        <p class="muted small">{{ r.chunks }} passage(s) retrieved.</p>
      }
    </section>
  `,
  styles: `
    .router-card { padding: var(--space-4); display: grid; gap: var(--space-2); }
    .router-head { display: flex; gap: var(--space-2); align-items: center; flex-wrap: wrap; }
    .strategy { text-transform: capitalize; }
    .router-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr)); gap: var(--space-2); margin: 0; }
    .router-facts dt { font-size: 0.75rem; color: var(--text-muted); }
    .router-facts dd { margin: 0; text-transform: capitalize; }
    .reasoning { margin: 0; }
    .fallback { margin: 0; }
    .small { font-size: 0.8rem; margin: 0; }
  `,
})
export class RouterCardComponent {
  readonly retrieval = input<RetrievalEvent['data'] | null>(null);
  readonly mode = input<'auto' | 'manual'>('auto');

  /** `signals` is a rule; `signals_fallback` is a rule used because the classifier failed. */
  readonly sourceLabel = computed(() => {
    const source = this.retrieval()?.router?.source;
    if (source === 'classifier') {
      return 'classifier';
    }
    return source === 'signals_fallback' ? 'rule (classifier unavailable)' : 'rule';
  });

  readonly confidence = computed(() => {
    const router = this.retrieval()?.router;
    if (!router) {
      return '';
    }
    return router.confidence == null
      ? 'Rule based, no confidence measured'
      : `${percent(router.confidence)} (as reported by the classifier, uncalibrated)`;
  });
}
