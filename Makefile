# make eval                          all strategies and the router, the shipped questions
# make eval STRATEGY=graph           one target (traditional+rerank=llm works too)
# make eval QUESTIONS=./my.json COLLECTION=handbook   your own questions over your own collection
# Writes docs/benchmarks/latest.md from the stored run (ADR 0004).

STRATEGY ?=
QUESTIONS ?=
COLLECTION ?=
REPORT ?= docs/benchmarks/latest.md

.PHONY: eval
eval:
	uv run ragfabric eval corpus
	uv run ragfabric eval run $(if $(STRATEGY),--strategy $(STRATEGY)) $(if $(QUESTIONS),--questions $(QUESTIONS)) $(if $(COLLECTION),--collection $(COLLECTION)) --report $(REPORT)
