import { ChangeDetectionStrategy, Component, OnInit, computed, inject, input, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { RagFabricError, Run } from '@ragfabric/sdk';

import { count, ms, percent, score, usd } from '../../answer/format';
import { describeError } from '../../services/http-error';
import { RunDetailsService } from '../../services/run-details.service';
import { RunService } from '../../services/run.service';
import { waterfall } from './waterfall';

/**
 * Trace: everything stored about one run, and what this session saw stream.
 *
 * The stored run (GET /api/runs/{id}) gives the spans, the latency split, the
 * counts, the estimated cost, the routing and the sources. The walked sub
 * graph and the agent's sub questions are not stored with a run (decision
 * D17), so they appear only for runs streamed in this browser session, and
 * the page says so otherwise.
 */
@Component({
  selector: 'app-trace',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink],
  template: `
    <div class="page-head">
      <h1>Trace @if (run(); as r) {<span class="muted">run {{ r.id }}</span>}</h1>
      <p>How one answer was produced: each timed step, what it cost, and where the passages came from.</p>
    </div>

    @if (error()) {
      <div class="alert alert-error" role="alert" data-test="error">{{ error() }}</div>
    } @else if (run(); as r) {
      <section class="card block">
        <p class="question" data-test="question">{{ r.question }}</p>
        <dl class="facts">
          <div><dt>Mode</dt><dd>{{ r.mode }}</dd></div>
          <div><dt>Requested</dt><dd>{{ r.requested_strategy ?? 'n/a' }}</dd></div>
          <div><dt>Answered by</dt><dd data-test="selected">{{ r.selected_strategy }}</dd></div>
          <div><dt>Router confidence</dt><dd data-test="router-confidence">{{ routerConfidence() }}</dd></div>
          <div><dt>LLM</dt><dd>{{ r.llm_model ?? 'n/a' }}</dd></div>
          <div><dt>Embeddings</dt><dd>{{ r.embedding_model ?? 'n/a' }}</dd></div>
        </dl>
        @if (r.fallback_from) {
          <p class="alert alert-warning" data-test="fallback">{{ r.fallback_from }} found nothing; {{ r.selected_strategy }} answered instead.</p>
        }
        @if (r.router_reasoning) {
          <p data-test="router-reasoning"><span class="muted">Router reasoning:</span> {{ r.router_reasoning }}</p>
        }
      </section>

      <section class="card block" aria-label="Cost and latency">
        <h2>Cost and latency</h2>
        <dl class="facts">
          <div><dt>Total</dt><dd data-test="latency">{{ ms(r.latency_ms) }}</dd></div>
          <div><dt>Retrieval</dt><dd>{{ ms(r.retrieval_latency_ms) }}</dd></div>
          <div><dt>Generation</dt><dd>{{ ms(r.generation_latency_ms) }}</dd></div>
          <div><dt>LLM calls</dt><dd>{{ count(r.llm_calls) }}</dd></div>
          <div><dt>Retrieval calls</dt><dd>{{ count(r.retrieval_calls) }}</dd></div>
          <div><dt>Tokens in / out</dt><dd>{{ count(r.input_tokens) }} / {{ count(r.output_tokens) }}</dd></div>
          <div><dt>Cost (estimate)</dt><dd data-test="cost">{{ usd(r.estimated_cost_usd) }}</dd></div>
        </dl>
        <div class="split" aria-hidden="true">
          <span class="split-retrieval" [style.flex-grow]="r.retrieval_latency_ms"></span>
          <span class="split-generation" [style.flex-grow]="r.generation_latency_ms"></span>
        </div>
      </section>

      <section class="card block" aria-label="Steps">
        <h2>{{ r.selected_strategy === 'agentic' ? 'Agent steps' : 'Steps' }}</h2>
        @if (bars().length === 0) {
          <p class="muted">No spans were stored for this run.</p>
        }
        <ol class="waterfall">
          @for (bar of bars(); track $index) {
            <li class="span-row" data-test="span">
              <div class="span-label">
                <strong>{{ bar.span.name }}</strong>
                <span class="muted">{{ ms(bar.span.started_ms) }} + {{ ms(bar.span.duration_ms) }}</span>
              </div>
              <div class="track"><span class="bar" [style.left.%]="bar.left" [style.width.%]="bar.width"></span></div>
              @if (attributes(bar.span.attributes).length > 0) {
                <div class="attrs">
                  @for (a of attributes(bar.span.attributes); track a[0]) {<code>{{ a[0] }}={{ a[1] }}</code>}
                </div>
              }
            </li>
          }
        </ol>
      </section>

      <section class="card block" aria-label="Graph path and sub questions">
        <h2>Graph path and sub questions</h2>
        @if (details(); as d) {
          @if (d.retrieval?.subgraph; as g) {
            <p class="muted">{{ g.nodes.length }} entities, {{ g.edges.length }} relationships walked@if (g.truncated) {, cut by the node budget}.</p>
            @if (g.empty_reason) {<p data-test="empty-reason">Nothing walked: {{ g.empty_reason }}</p>}
            <ul class="edges" data-test="graph-path">
              @for (e of g.edges; track e.id) {
                <li>{{ nodeName(g.nodes, e.source_id) }} <code>{{ e.walked_as }}</code> {{ nodeName(g.nodes, e.target_id) }}
                  <span class="muted">(confidence {{ score(e.confidence) }})</span></li>
              }
            </ul>
          }
          @if ((d.retrieval?.sub_questions?.length ?? 0) > 0) {
            <ul class="subq" data-test="sub-questions">
              @for (q of d.retrieval!.sub_questions; track $index) {
                <li><strong>{{ q.status }}</strong>: {{ q.text }}@if (q.reason) { <span class="muted">({{ q.reason }})</span>}</li>
              }
            </ul>
          }
          @if (!d.retrieval?.subgraph && (d.retrieval?.sub_questions?.length ?? 0) === 0) {
            <p class="muted">This strategy walks no graph and asks no sub questions.</p>
          }
        } @else {
          <p class="muted" data-test="not-stored">
            The walked sub graph and the agent's sub questions are not stored with a run. They are shown
            here only for runs asked in this browser session.
          </p>
        }
      </section>

      <section class="card block" aria-label="Sources">
        <h2>Sources</h2>
        <table class="sources">
          <thead><tr><th>Rank</th><th>Document</th><th>Chunk</th><th>Page</th><th>Score</th><th>Cited</th></tr></thead>
          <tbody>
            @for (s of r.sources ?? []; track s.rank) {
              <tr data-test="source">
                <td>{{ s.rank }}</td><td>{{ s.document_id ?? 'n/a' }}</td><td>{{ s.chunk_id ?? 'n/a' }}</td>
                <td>{{ s.page ?? 'n/a' }}</td><td>{{ score(s.score, 3) }}</td><td>{{ s.cited ? 'yes' : 'no' }}</td>
              </tr>
            }
          </tbody>
        </table>
      </section>

      @if (r.answer) {
        <section class="card block" aria-label="Recorded answer">
          <h2>Recorded answer</h2>
          <p class="answer">{{ r.answer }}</p>
        </section>
      }
      <a class="btn btn-sm" routerLink="/ask">Ask another question</a>
    } @else {
      <div class="empty-state"><span class="spinner spinner-brand"></span></div>
    }
  `,
  styles: `
    :host { display: grid; gap: var(--space-4); }
    .block { padding: var(--space-4); display: grid; gap: var(--space-3); }
    h2 { margin: 0; font-size: 1rem; }
    .question { font-size: 1.05rem; margin: 0; }
    .facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: var(--space-2); margin: 0; }
    .facts dt { font-size: 0.72rem; color: var(--text-muted); }
    .facts dd { margin: 0; overflow-wrap: anywhere; }
    .split { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--surface-2); }
    .split-retrieval { background: var(--accent); }
    .split-generation { background: var(--brand); }
    .waterfall { list-style: none; padding: 0; margin: 0; display: grid; gap: var(--space-2); }
    .span-label { display: flex; justify-content: space-between; gap: var(--space-2); font-size: 0.85rem; }
    .track { position: relative; height: 10px; background: var(--surface-2); border-radius: 5px; }
    .bar { position: absolute; top: 0; bottom: 0; background: var(--brand); border-radius: 5px; }
    .attrs { display: flex; flex-wrap: wrap; gap: var(--space-1); font-size: 0.75rem; }
    .sources { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .sources th, .sources td { text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border); }
    .answer { white-space: pre-wrap; margin: 0; }
    .edges, .subq { margin: 0; padding-left: 1.2rem; }
  `,
})
export class TraceComponent implements OnInit {
  private readonly runs = inject(RunService);
  private readonly runDetails = inject(RunDetailsService);

  /** Bound from the route parameter (`withComponentInputBinding`). */
  readonly runId = input.required<string>();
  readonly run = signal<Run | null>(null);
  readonly error = signal<string | null>(null);

  readonly ms = ms;
  readonly count = count;
  readonly usd = usd;
  readonly score = score;

  readonly bars = computed(() => {
    const r = this.run();
    return r ? waterfall(r.trace, r.latency_ms) : [];
  });
  readonly details = computed(() => {
    const r = this.run();
    return r ? this.runDetails.get(r.id) : null;
  });
  readonly routerConfidence = computed(() => {
    const r = this.run();
    if (!r || r.mode !== 'auto') {
      return 'n/a (not routed)';
    }
    return r.router_confidence == null ? 'rule based, none measured' : percent(r.router_confidence);
  });

  ngOnInit(): void {
    const id = Number(this.runId());
    if (!Number.isInteger(id) || id <= 0) {
      this.error.set('Run not found.');
      return;
    }
    this.runs.get(id).subscribe({
      next: (run) => this.run.set(run),
      error: (err: unknown) =>
        this.error.set(
          err instanceof RagFabricError && err.status === 404
            ? 'Run not found. It may belong to another user.'
            : describeError(err, 'The run could not be loaded.'),
        ),
    });
  }

  attributes(attrs: Record<string, unknown> | undefined): [string, string][] {
    return Object.entries(attrs ?? {}).map(([k, v]) => [k, String(v)]);
  }

  nodeName(nodes: readonly { id: number; name: string }[], id: number): string {
    return nodes.find((n) => n.id === id)?.name ?? `#${id}`;
  }
}
