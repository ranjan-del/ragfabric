# Access leak test

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).

Two collections from `benchmarks/leak_corpus`: `leak-public` (open) and `leak-secret` (granted to one
group). A principal outside that group asks eight questions answered only in `leak-secret`, scoped to
both collections. Per target: probes that retrieved any forbidden chunk, forbidden chunks retrieved
(any is a leak), and result slots left unfilled although the open collection could fill them (a
filter that leaks nothing can still starve the answer). LlamaIndex and LangChain have no access model
of their own, so each runs as built by default and with its documented metadata filter (D17).

Reproduce: `uv run --with-requirements benchmarks/requirements-competitors.txt python benchmarks/leak_test.py --out $RFB_BENCH_DIR/leak-test.json`.

<!-- summarise kind="leak" file="leak-test.json" targets="ragfabric:traditional,ragfabric:vectorless,ragfabric:auto,ragfabric:agentic,llamaindex:default,llamaindex:metadata-filter,langchain:default,langchain:metadata-filter" -->
| target | probes_leaking | forbidden_chunks | unfilled_slots |
|---|---|---|---|
| ragfabric:traditional | TBD | TBD | TBD |
| ragfabric:vectorless | TBD | TBD | TBD |
| ragfabric:auto | TBD | TBD | TBD |
| ragfabric:agentic | TBD | TBD | TBD |
| llamaindex:default | TBD | TBD | TBD |
| llamaindex:metadata-filter | TBD | TBD | TBD |
| langchain:default | TBD | TBD | TBD |
| langchain:metadata-filter | TBD | TBD | TBD |
<!-- /summarise -->
