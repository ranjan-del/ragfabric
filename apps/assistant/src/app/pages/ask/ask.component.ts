import { ChangeDetectionStrategy, Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { AskOptions, Citation, STRATEGIES, StrategyName } from '@ragfabric/sdk';
import { Subscription } from 'rxjs';

import { AnswerState } from '../../answer/answer-state';
import { AnswerViewComponent } from '../../answer/answer-view.component';
import { count, ms } from '../../answer/format';
import { RouterCardComponent } from '../../answer/router-card.component';
import { SourceViewerComponent } from '../../answer/source-viewer.component';
import { Collection } from '../../models';
import { AskService } from '../../services/ask.service';
import { CollectionService } from '../../services/collection.service';

/**
 * Ask: one question, one streamed answer, and why that strategy answered it.
 *
 * AUTO sends `strategy: "auto"` and shows the router's decision; MANUAL sends
 * the strategy picked here. Citations are buttons that open the source viewer.
 */
@Component({
  selector: 'app-ask',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, RouterLink, AnswerViewComponent, RouterCardComponent, SourceViewerComponent],
  template: `
    <div class="page-head">
      <h1>Ask</h1>
      <p>Ask in plain language. The answer streams in with its sources and the reason this strategy answered it.</p>
    </div>

    <form class="card ask-form" (ngSubmit)="ask()">
      <label class="sr-only" for="question">Question</label>
      <textarea
        id="question"
        name="question"
        rows="2"
        [(ngModel)]="question"
        placeholder="e.g. How did casual leave change between 2025 and 2026?"
        (keydown.enter)="submitOnEnter($event)"
      ></textarea>
      <div class="controls">
        <fieldset class="mode" aria-label="Mode">
          <label><input type="radio" name="mode" value="auto" [(ngModel)]="mode" /> AUTO</label>
          <label><input type="radio" name="mode" value="manual" [(ngModel)]="mode" /> MANUAL</label>
        </fieldset>
        @if (mode === 'manual') {
          <label class="control">
            <span>Strategy</span>
            <select name="strategy" [(ngModel)]="strategy" data-test="strategy-select">
              @for (s of strategies; track s) {<option [value]="s">{{ s }}</option>}
            </select>
          </label>
        }
        <label class="control">
          <span>Collection</span>
          <select name="collection" [(ngModel)]="collectionId">
            <option [ngValue]="null">All I can read</option>
            @for (c of collections(); track c.id) {<option [ngValue]="c.id">{{ c.name }}</option>}
          </select>
        </label>
        <label class="control narrow">
          <span>Passages</span>
          <input type="number" name="topK" min="1" max="50" [(ngModel)]="topK" />
        </label>
        <button class="btn btn-primary" type="submit" [disabled]="state.status() === 'running' || !question.trim()">
          {{ state.status() === 'running' ? 'Answering...' : 'Ask' }}
        </button>
        @if (state.status() === 'running') {
          <button class="btn btn-ghost" type="button" (click)="stop()">Stop</button>
        }
      </div>
    </form>

    @if (state.status() !== 'idle') {
      <div class="result">
        <app-router-card [retrieval]="state.retrieval()" [mode]="mode" />
        <section class="card answer" aria-label="Answer">
          <app-answer-view [state]="state" (cite)="open($event)" />
          @if (state.done(); as d) {
            <dl class="meta" data-test="meta">
              <div><dt>Latency</dt><dd>{{ ms(d.latency_ms) }}</dd></div>
              <div><dt>LLM calls</dt><dd>{{ count(d.usage.llm_calls) }}</dd></div>
              <div><dt>Retrieval calls</dt><dd>{{ count(d.usage.retrieval_calls) }}</dd></div>
              <div><dt>Embedding calls</dt><dd>{{ count(d.usage.embedding_calls) }}</dd></div>
              <div><dt>Tokens in / out</dt><dd>{{ count(d.usage.input_tokens) }} / {{ count(d.usage.output_tokens) }}</dd></div>
            </dl>
            <a class="btn btn-sm" [routerLink]="['/trace', d.run_id]" data-test="trace-link">Open the trace</a>
          }
        </section>
      </div>
    }

    <app-source-viewer [citation]="viewing()" (closed)="viewing.set(null)" />
  `,
  styles: `
    .ask-form { display: grid; gap: var(--space-3); padding: var(--space-4); }
    textarea { width: 100%; resize: vertical; font: inherit; padding: var(--space-3);
      border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface); color: var(--text); }
    .controls { display: flex; flex-wrap: wrap; gap: var(--space-3); align-items: end; }
    .mode { display: flex; gap: var(--space-3); border: 0; padding: 0; margin: 0; }
    .mode label { display: inline-flex; align-items: center; gap: var(--space-1); white-space: nowrap; }
    .mode input { width: auto; margin: 0; }
    .answer > a { justify-self: start; }
    .control { display: grid; gap: var(--space-1); font-size: 0.8rem; color: var(--text-muted); }
    .control select, .control input { font: inherit; color: var(--text); background: var(--surface);
      border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.4rem 0.6rem; }
    .narrow input { width: 5rem; }
    .result { display: grid; gap: var(--space-4); margin-top: var(--space-4); }
    .answer { padding: var(--space-4); display: grid; gap: var(--space-3); }
    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr)); gap: var(--space-2); margin: 0; }
    .meta dt { font-size: 0.75rem; color: var(--text-muted); }
    .meta dd { margin: 0; }
  `,
})
export class AskComponent implements OnInit, OnDestroy {
  private readonly asker = inject(AskService);
  private readonly collectionService = inject(CollectionService);
  private subscription: Subscription | null = null;

  readonly strategies = STRATEGIES;
  readonly state = new AnswerState();
  readonly collections = signal<Collection[]>([]);
  readonly viewing = signal<Citation | null>(null);
  readonly ms = ms;
  readonly count = count;

  question = '';
  mode: 'auto' | 'manual' = 'auto';
  strategy: StrategyName = 'traditional';
  collectionId: number | null = null;
  topK = 8;

  ngOnInit(): void {
    this.collectionService.list().subscribe({ next: (list) => this.collections.set(list) });
  }

  ngOnDestroy(): void {
    this.stop();
  }

  ask(): void {
    const question = this.question.trim();
    if (!question || this.state.status() === 'running') {
      return;
    }
    this.stop();
    const options: AskOptions = {
      strategy: this.mode === 'auto' ? 'auto' : this.strategy,
      top_k: this.topK,
      collection_id: this.collectionId ?? undefined,
    };
    this.subscription = this.asker.run(this.state, question, options);
  }

  stop(): void {
    this.subscription?.unsubscribe();
    this.subscription = null;
    if (this.state.status() === 'running') {
      this.state.fail('Stopped before the answer finished.');
    }
  }

  open(citation: Citation): void {
    this.viewing.set(citation);
  }

  submitOnEnter(event: Event): void {
    const key = event as KeyboardEvent;
    if (!key.shiftKey) {
      key.preventDefault();
      this.ask();
    }
  }
}
