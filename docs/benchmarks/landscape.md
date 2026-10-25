# RAG landscape

> Status: desk research as at 25 October 2026, from each project's or vendor's own documentation and
> from published papers. **Nothing on this page was measured by RagFabric.** A vendor figure is the
> vendor's claim, on its own data and setup, and is labelled as such. Where a product publishes no
> quality figure, the page says so rather than estimating one (ADR 0004). Measured comparisons are
> in [comparison.md](comparison.md).

## Feature comparison

| Product | Type | Retrieval offered | Citations and grounding | Evaluation built in | Access control | Self-host | Published quality figures |
|---|---|---|---|---|---|---|---|
| RagFabric | Open source | Vector, BM25 with full text, cross encoder rerank, graph, agentic, router with fallbacks | Every sentence must carry a marker, checked in code; repair or refuse | Hit, MRR, correctness, faithfulness, citation, cost | Groups, grants and per-document overrides applied inside every store query, with a leak test | Yes | Own subsets only ([comparison.md](comparison.md)) |
| LlamaIndex | Open source (MIT) | Vector, rerank, graph indexes, agents and workflows | Not enforced | Faithfulness, relevancy, correctness, hit rate, MRR | Metadata filters | Yes | None published |
| LangChain / LangGraph | Open source; LangSmith paid | Any retriever; an agentic RAG tutorial grades documents and rewrites the question | Not enforced | Through LangSmith | Not built in | Yes | None published |
| Haystack | Open source; enterprise platform paid | Retrievers, rankers, agents | Not enforced | MRR, MAP, NDCG, recall, faithfulness, context relevance | Enterprise platform | Yes | None published |
| RAGFlow | Open source (Apache 2.0) | Weighted vector and full text, rerank, GraphRAG, RAPTOR, multi-round agentic | Traceable citations | Retrieval testing page | Not stated | Yes (Docker) | None published |
| Microsoft GraphRAG | Open source (MIT) | Graph with community summaries | Not stated | No | No | Yes | Comprehensiveness win rate of 72% to 83% against naive RAG, judged by an LLM (paper) |
| LightRAG | Open source (MIT) | Naive, local, global, hybrid and mix modes; rerank | Yes | RAGAS integration | No | Yes | Win rates against naive RAG in its paper, judged by an LLM |
| Dify | Source available | Vector, full text, weighted hybrid, rerank, agents | Not stated | Through external tools | SSO and RBAC in the enterprise edition | Yes | None published |
| R2R | Open source (MIT) | Hybrid with reciprocal rank fusion, knowledge graph, research agent | Yes | No | Users and collections | Yes | None published |
| Kotaemon | Open source (Apache 2.0) | Hybrid, rerank, GraphRAG and LightRAG, ReAct, ReWOO | Highlighted in a PDF viewer | No | Private and public collections | Yes | None published |
| PageIndex | Open source (MIT) and cloud | No vectors; the model searches a tree index of each document | Page level | No | No | Yes | FinanceBench 98.7% (vendor figure) |
| OpenAI file search | Managed, paid | Hybrid with weighted fusion, ranker, score threshold, query rewrite | File citation annotations | No | Attribute filters | No | None published |
| Google Vertex AI RAG Engine | Managed, paid | Hybrid, ranking API | Check grounding API: support score and claim level citations | Not found | Access lists from source connectors (preview) | No | None published |
| Azure AI Search agentic retrieval | Managed, paid by tokens | LLM query planning, subqueries run in parallel, each semantically reranked | Source references and an activity log | No | Document level access lists (preview) | No | Up to 40% better answer relevance than single-query RAG (vendor figure) |
| AWS Bedrock Knowledge Bases | Managed, paid | Managed hybrid, managed rerank, agentic retrieval | Citations to source chunks | Citation precision and coverage, faithfulness | Access list awareness (managed tier) | No | None published |
| Cohere | Managed or private deployment, paid | Rerank, tool use RAG | Citations tied to spans of the answer | No | No | Private deployment | None published |
| Vectara | Managed, VPC or on premises, paid | Hybrid, rerank, agents | Citations, hallucination scoring (HHEM) | Yes | Enforces each source's permissions | Yes (VPC, on premises) | Customer case study only |
| Pinecone Assistant | Managed, paid | Not documented | File references | Correctness, completeness, alignment | Metadata filters | No | None published |

Verba (Weaviate) was archived on 8 June 2026 and is not compared.

## Where RagFabric is ahead

- **A router across strategies.** None of the open source tools reviewed picks a retrieval strategy
  per question, records why, and falls back to another. Azure and Bedrock offer agentic retrieval
  only as managed services.
- **A citation contract in code.** Answers without valid markers are repaired or refused. No open
  source framework reviewed enforces this; Google's check grounding and Bedrock's citation
  evaluation are the nearest managed equivalents.
- **Permissions inside the store query, with a leak test.** Among the open source tools this is
  matched only in part, by R2R collections and Kotaemon's private collections.
- **One harness for retrieval, answers, citations and cost**, comparable to Haystack and LlamaIndex
  evaluation, and more than RAGFlow, Dify, R2R or Kotaemon document.
- **A bounded agent** with hard caps and recorded stop reasons.
- **PostgreSQL is the only store**, and it runs fully offline with local models.

## Where others are ahead

- **Connectors.** RagFabric has a watched folder and Google Drive; the managed services connect
  many enterprise sources and carry their permissions over.
- **Identity.** RagFabric has JWT and API keys, with OIDC planned. Glean, Vectara, Azure, Google
  and Bedrock enforce the source system's own permissions.
- **Document parsing.** RagFabric has no OCR; RAGFlow handles scanned documents.
- **Hybrid search by default.** OpenAI, Dify and RAGFlow fuse vector and keyword results by
  default. RagFabric fuses them only when the router is unsure, although in its own MultiHop-RAG
  run vectorless retrieval found the right document more often than vector retrieval.
- **Agentic speed and quality.** Azure plans subqueries and runs them in parallel, each reranked.
  RagFabric's agent retrieves one sub-question after another and, on a small local model, scores
  below its own single shot strategy ([agentic-rag.md](../agentic-rag.md#measured-behaviour-and-next-steps)).
- **Index enrichment.** RagFabric has no contextual chunk enrichment, RAPTOR or GraphRAG community
  summaries (the last is on the roadmap).

## Agentic techniques in the literature

Gains as reported by each paper, on its own models and datasets; none was reproduced here.

| Technique | What it does | Reported result |
|---|---|---|
| Query decomposition with rerank | Splits the question, retrieves per part, reranks the pooled results | MRR@10 up 36.7% and F1 up 11.6% on MultiHop-RAG and HotpotQA |
| Parallel subqueries | Runs planned subqueries at the same time, each reranked | Up to 40% better answer relevance (Azure, vendor figure) |
| IRCoT | Alternates retrieval with reasoning steps | Up to 21 points better retrieval and 15 points better QA on multi-hop sets |
| FLARE | Retrieves again when generation confidence is low | Superior or competitive on four long-form knowledge tasks (abstract; figures in the paper) |
| Self-RAG | A model trained to decide when to retrieve and to critique itself | Significantly better than retrieval augmented baselines on open-domain QA, reasoning and fact verification (abstract; figures in the paper) |
| CRAG | Grades retrieved passages and corrects or widens a weak search | Significant improvement over RAG approaches on four datasets (abstract; figures in the paper) |
| Adaptive-RAG | A small classifier picks no retrieval, single step or multi step | Close to multi step accuracy at less than half the steps |

For scale, the MultiHop-RAG paper's best retrieval was Hits@10 0.7467 (voyage-02 with
bge-reranker-large), and GPT-4 reached 0.56 accuracy from retrieved chunks against 0.89 with the
ground truth evidence. RagFabric's numbers on a 100 document subset are not comparable with these.

## What this suggests for RagFabric

1. Make hybrid fusion the default for Traditional and for the agent's tools, then rerank after
   fusion, and find out why the cross encoder raised hit rate but lowered correctness.
2. Rework the agent: parallel sub-question retrieval, a ranked and capped evidence pool, and
   escalation from single shot only when the evidence check fails.
3. Add a semantic grounding check per claim beside the citation contract, for small models whose
   citations are weak.
4. Add OIDC and SAML, carry source permissions over, and add more enterprise connectors.
5. Add contextual chunk enrichment and OCR at ingestion.

## Sources

- MultiHop-RAG: <https://arxiv.org/abs/2401.15391>
- Question decomposition: <https://arxiv.org/abs/2507.00355>
- IRCoT: <https://aclanthology.org/2023.acl-long.557>
- FLARE: <https://arxiv.org/abs/2305.06983>
- Self-RAG: <https://arxiv.org/abs/2310.11511>
- CRAG: <https://arxiv.org/abs/2401.15884>
- Adaptive-RAG: <https://arxiv.org/abs/2403.14403>
- GraphRAG: <https://arxiv.org/abs/2404.16130>, <https://github.com/microsoft/graphrag>
- LightRAG: <https://github.com/HKUDS/LightRAG>
- RAGFlow: <https://github.com/infiniflow/ragflow>
- R2R: <https://github.com/SciPhi-AI/R2R>; Kotaemon: <https://github.com/Cinnamon/kotaemon>;
  PageIndex: <https://github.com/VectifyAI/PageIndex>; Verba: <https://github.com/weaviate/Verba>
- LlamaIndex evaluation: <https://developers.llamaindex.ai/python/framework/module_guides/evaluating/>
- Haystack evaluators: <https://docs.haystack.deepset.ai/docs/evaluators>
- LangGraph agentic RAG: <https://docs.langchain.com/oss/python/langgraph/agentic-rag>
- Dify: <https://github.com/langgenius/dify>
- OpenAI file search: <https://developers.openai.com/api/docs/guides/retrieval>
- Azure AI Search agentic retrieval: <https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview>
- AWS Bedrock Knowledge Bases: <https://docs.aws.amazon.com/bedrock/latest/userguide/kb-how-retrieval.html>
- Google Vertex AI RAG Engine: <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview>,
  check grounding: <https://docs.cloud.google.com/generative-ai-app-builder/docs/check-grounding>
- Cohere RAG: <https://docs.cohere.com/docs/retrieval-augmented-generation-rag>
- Vectara: <https://www.vectara.com/>
- Pinecone Assistant: <https://docs.pinecone.io/guides/assistant/overview>
