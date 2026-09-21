// Shared API types mirroring the FastAPI response schemas.

export interface User {
  id: number;
  email: string;
  role: 'admin' | 'user';
  is_active: boolean;
  created_at: string;
}

export interface Token {
  access_token: string;
  token_type: string;
}

export interface DocumentItem {
  id: number;
  filename: string;
  format: string;
  content_type: string;
  status: 'processing' | 'ready' | 'failed';
  collection_id: number | null;
  owner_id: number | null;
  version: number;
  num_chunks: number;
  error: string;
  created_at: string;
}

export interface DocumentList {
  items: DocumentItem[];
  total: number;
}

export interface Collection {
  id: number;
  name: string;
  description: string;
  owner_id: number | null;
  created_at: string;
  document_count: number;
}

export interface CollectionDetail extends Collection {
  documents: DocumentItem[];
}

/** A character range within a string carried elsewhere in the same payload. */
export interface Span {
  text: string;
  start: number;
  end: number;
}

export interface Highlight {
  term: string;
  start: number;
  end: number;
}

export interface Citation {
  marker: string;
  chunk_id: number | null;
  document_id: number | null;
  filename: string | null;
  page: number | null;
  score: number;
  snippet: string;
  /** True when the answer text actually carries this marker. */
  used: boolean;
  /** Query-term spans, relative to `snippet`. */
  highlights: Highlight[];
  /** The sentence the answer quoted from this chunk, relative to `snippet`. */
  supporting_span: Span | null;
}

export interface SourceDocument {
  document_id: number | null;
  filename: string | null;
  page: number | null;
  collection_id: number | null;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  confidence: number;
  citations: Citation[];
  /** Query-term spans, relative to `answer`. */
  highlights: Highlight[];
  source_document: SourceDocument | null;
}

export interface SearchResultItem {
  chunk_id: number | null;
  document_id: number | null;
  filename: string | null;
  format: string | null;
  page: number | null;
  chunk_index: number | null;
  score: number;
  /** Hybrid mode only: the two component scores behind the fused ranking. */
  lexical_score: number | null;
  hybrid_score: number | null;
  text: string;
}

export interface SearchResults {
  query: string;
  mode: string;
  results: SearchResultItem[];
}

export interface AnalyticsOverview {
  documents: number;
  collections: number;
  chunks: number;
  users: number;
  queries: number;
  ready_documents: number;
  /** Live size of the in-memory index. Should equal `chunks`; if it does not,
   *  the index has drifted from the database and needs a rebuild. */
  indexed_vectors: number;
}

export interface UsageStats {
  recent_queries: { question: string; confidence: number; created_at: string }[];
  top_documents: { document_id: number; filename: string; chunks: number }[];
  /** Documents ranked by how often answers actually cited them. */
  most_cited_documents: {
    document_id: number;
    filename: string;
    citations: number;
  }[];
}

// --- Console v1 (Phase 4) -------------------------------------------------

export interface Group {
  id: number;
  name: string;
  description: string;
  created_at: string;
}

export interface Grant {
  id: number;
  group_id: number;
  collection_id: number;
  permission: 'read' | 'write';
}

export interface ApiKeyItem {
  id: number;
  name: string;
  key_prefix: string;
  principal_user_id: number | null;
  collection_ids: number[];
  strategies: string[];
  rate_limit_per_minute: number;
  is_active: boolean;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
}

/** Only ever returned once, by the create call. Never stored anywhere. */
export interface ApiKeyCreated extends ApiKeyItem {
  key: string;
}

export interface ProviderStatus {
  provider: string;
  model: string | null;
  base_url: string | null;
  dim: number | null;
  key_env_var: string | null;
  requires_key: boolean;
  /** Whether the key is present in the server's environment. Never the key. */
  has_key: boolean;
}

export interface ProviderConfig {
  llm: ProviderStatus;
  embeddings: ProviderStatus;
}

export interface ProviderConfigWritten extends ProviderConfig {
  requires_reindex: boolean;
  restart_required: boolean;
  changed: string[];
}

export interface ProviderTestResult {
  target: string;
  ok: boolean;
  detail: string;
  model: string | null;
}
