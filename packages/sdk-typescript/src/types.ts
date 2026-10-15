// Public type names over the generated schema.
//
// Everything here is either an alias of a type generated from the server's
// OpenAPI specification (src/generated/schema.ts), or, where the specification
// cannot describe a shape, typed by hand and says why. Hand typed shapes are
// the drift surface, so they are kept few and each is pinned by a test.

import type * as S from './generated/schema.js';

export type { paths } from './generated/schema.js';

export type Answer = S.AnswerResponse;
export type AskRequest = S.AskRequest;
export type Citation = S.Citation;
export type Highlight = S.Highlight;
export type Span = S.Span;
export type SourceDocument = S.SourceDocument;
export type Usage = S.Usage;
export type RouterDecision = S.RouterDecisionOut;
export type Subgraph = S.SubgraphOut;
export type GraphNode = S.GraphNodeOut;
export type GraphEdge = S.GraphEdgeOut;
export type SubQuestionReport = S.SubQuestionReportOut;
export type DroppedClaim = S.DroppedClaimOut;
export type TraceSpan = S.TraceSpanOut;
export type Run = S.RunOut;
export type RunSource = S.SourceOut;
export type IngestionRun = S.IngestionRunOut;
export type SearchRequest = S.SearchRequest;
export type SearchResults = S.SearchResults;
export type SearchResultItem = S.SearchResultItem;
export type User = S.UserOut;
export type Token = S.Token;
export type DocumentItem = S.DocumentOut;
export type DocumentList = S.DocumentList;
export type Collection = S.CollectionOut;
export type CollectionDetail = S.CollectionDetail;
export type CollectionCreate = S.CollectionCreate;
export type CollectionUpdate = S.CollectionUpdate;
export type Group = S.GroupOut;
export type GroupUpdate = S.GroupUpdate;
export type Grant = S.GrantOut;
export type ApiKeyCreate = S.ApiKeyCreate;
export type ApiKeyItem = S.ApiKeyOut;
export type ApiKeyCreated = S.ApiKeyCreated;
export type Override = S.OverrideOut;
export type OverrideCreate = S.OverrideCreate;
export type AdminUserCreate = S.AdminUserCreate;
export type PermissionUpdate = S.PermissionUpdate;
export type ProviderStatus = S.ProviderStatus;
export type ProviderConfig = S.ProviderConfigOut;
export type ProviderConfigUpdate = S.ProviderConfigUpdate;
export type ProviderConfigWritten = S.ProviderConfigWritten;
export type ProviderTestResult = S.ProviderTestResult;

/** The four retrieval strategies, in the order Compare shows them. */
export type StrategyName = 'traditional' | 'vectorless' | 'agentic' | 'graph';
/** What a caller may request: a strategy, or `auto` to let the router choose. */
export type RequestedStrategy = StrategyName | 'auto';
export const STRATEGIES: readonly StrategyName[] = ['traditional', 'vectorless', 'agentic', 'graph'];

// --- Analytics --------------------------------------------------------------
// Hand typed: GET /api/analytics/overview and /usage declare no response model,
// so the specification types both as unknown.

export interface AnalyticsOverview {
  documents: number;
  collections: number;
  chunks: number;
  users: number;
  queries: number;
  ready_documents: number;
  indexed_vectors: number;
}

export interface UsageStats {
  recent_queries: { question: string; confidence: number; created_at: string }[];
  top_documents: { document_id: number; filename: string; chunks: number }[];
  most_cited_documents: { document_id: number; filename: string; citations: number }[];
}

// --- The /api/ask event stream -----------------------------------------------
// Hand typed (decision D8): OpenAPI cannot describe a server sent events
// stream. The envelope is typed here; every payload reuses generated types.
// test/events.test.ts pins ASK_EVENT_NAMES to the names the server sends.

export const ASK_EVENT_NAMES = ['retrieval', 'token', 'superseded', 'citations', 'done'] as const;

export interface RetrievalEvent {
  event: 'retrieval';
  data: {
    chunks: number;
    strategy: string;
    trace: TraceSpan[];
    sub_questions: SubQuestionReport[];
    subgraph: Subgraph | null;
    router: RouterDecision | null;
    fallback_from: string | null;
  };
}

export interface TokenEvent {
  event: 'token';
  data: { text: string };
}

/**
 * The streamed text failed the citation contract and was repaired. Replace
 * whatever was drawn from the token events with `data.text`: that is the
 * answer the server recorded.
 */
export interface SupersededEvent {
  event: 'superseded';
  data: {
    text: string;
    reason: string;
    dropped_claims?: DroppedClaim[];
    dropped_relationship_claims?: DroppedClaim[];
  };
}

export interface CitationsEvent {
  event: 'citations';
  data: { citations: Citation[] };
}

export interface DoneEvent {
  event: 'done';
  data: { run_id: number; latency_ms: number; usage: Usage };
}

/** An event this SDK does not know, passed through rather than dropped. */
export interface UnknownEvent {
  event: 'unknown';
  name: string;
  data: unknown;
}

export type AskEvent =
  | RetrievalEvent
  | TokenEvent
  | SupersededEvent
  | CitationsEvent
  | DoneEvent
  | UnknownEvent;
