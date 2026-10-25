# How RagFabric works, in plain words

> A guide for someone meeting RagFabric for the first time. It explains what the tool does, what
> each part is for, what the benchmark numbers mean, and what to expect from it today. The detailed
> pages are linked from each section.

## What it does

RagFabric answers questions from your own documents, and shows where each part of the answer came
from. You upload documents, ask a question in plain language, and get an answer with numbered
references such as `[1]` that open the exact passage it used.

It does this in three steps, the same steps every RAG (retrieval augmented generation) system takes:

| Step | What happens | Example |
|---|---|---|
| 1. Find | Search the documents and pick the few passages most likely to hold the answer | "How many days of leave carry forward?" finds the leave policy paragraph |
| 2. Answer | A language model reads only those passages and writes the answer | "Up to 10 days carry forward." |
| 3. Show the source | Each statement carries a marker pointing at the passage it came from | "Up to 10 days carry forward [1]", where [1] opens the policy paragraph |

What makes RagFabric different is that it has four ways of doing step 1, a router that picks one per
question, a rule that refuses an answer it cannot cite, and a built-in way to measure all of this on
your own documents.

## What happens when you add a document

| Stage | What RagFabric does |
|---|---|
| Read | Extracts the text from PDF, DOCX, PPTX, TXT, CSV or Markdown, keeping page numbers |
| Clean | Repairs hyphenation, removes repeated headers and footers, detects headings |
| Split | Cuts the text into overlapping chunks (600 characters by default), so a passage can be found and cited precisely |
| Index | Writes each chunk to a vector index (meaning), a keyword index (exact words) and, if enabled, a knowledge graph (people, teams and how they relate) |
| Protect | Records which collection the document belongs to, so every later search only sees what the asker may see |

## What happens when you ask

| Stage | What RagFabric does |
|---|---|
| Route | In `auto` mode the router reads the question and picks a strategy, with a short reason you can see |
| Retrieve | The chosen strategy finds the passages (details below) |
| Answer | The model writes an answer from those passages only |
| Check | The citation contract checks every marker points at a real passage and that quoted figures appear in it; if not, the model is asked to repair the answer, and if it still cannot, RagFabric says it could not find an answer rather than guess |
| Record | The run is stored with its sources, timings, model calls, tokens and estimated cost, and can be opened on the Trace page |

## The four strategies

| Strategy | How it finds passages | Good at | Weak at |
|---|---|---|---|
| Traditional | Compares the meaning of the question with the meaning of each chunk (vector search) | Questions phrased in ordinary language | Exact codes, names and numbers |
| Vectorless | Ranks chunks by the words they share with the question (BM25 and full text search) | Names, IDs, codes, exact phrases | Synonyms and paraphrase |
| Graph | Follows links between people, teams and documents in a knowledge graph | "Who reports to whom", relationship questions | Everything else; it needs a graph built at upload time |
| Agentic | Breaks the question into parts, searches for each, checks what it found and searches again if something is missing | Questions with several parts or comparisons | Speed; on small local models it currently scores below Traditional ([why](agentic-rag.md#measured-behaviour-and-next-steps)) |

`auto` is not a fifth strategy: it is the router choosing one of the four for each question. See
[routing.md](routing.md).

## Who can see what

Documents live in collections. Users belong to groups, and groups are granted collections; single
documents can be overridden. The filter is applied inside every search query, so a passage the asker
may not see is never retrieved, never sent to the model and never cited. API keys carry their own
rate limits. See [Access control](concepts/access-control.md).

## The pages in the app

| Page | What it is for |
|---|---|
| Ask | Ask a question in `auto` or with a chosen strategy; see the router's choice, the streamed answer and clickable citations |
| Compare | Ask one question with all four strategies side by side, with each answer, its sources, time and cost |
| Trace | Open one run and see every step it took, how long each took and what it cost |
| Evaluation | See the latest measured batch and how each strategy scored by question type |
| Documents and collections | Upload, organise and remove documents |
| Console | Users, groups, grants, API keys and rate limits (admins only) |

## What the benchmark numbers mean

Every score runs from 0 to 1, so 0.90 means 90%.

| Number | Plain meaning | Which step it checks |
|---|---|---|
| hit | Did the right document appear among the top 5 passages? | Find |
| mrr | How near the top the right document was; 1.0 means first every time | Find |
| recall | What share of all the relevant documents were found | Find |
| correctness | Is the answer right? A separate model marks it against the known answer | Answer |
| citation | When the answer says `[1]`, does passage 1 really hold the evidence? | Show the source |
| refusals | How many times it said it could not find an answer (a count, not a share) | Answer |
| p50 | Half of the questions were answered within this time | Speed |

A refusal is not always bad: when the documents do not hold the answer, refusing is correct. A wrong
citation is always bad, because it makes a right answer impossible to check.

## What to expect today

From the measured runs in [the benchmark comparison](benchmarks/comparison.md) and the
[model sweep](benchmarks/results/model-sweep.md), each one run on one laptop on a small sample:

| Area | What the runs show |
|---|---|
| Finding documents | Strong. Vector search found a relevant document for most questions on the BEIR samples, in tens of milliseconds |
| Answers with a strong cloud model | Accurate and well cited on the shipped questions |
| Answers with a small local model | Fair answers and weak citations on hard multi-document questions |
| Against LlamaIndex and LangChain | Same retrieval (they search RagFabric's own chunks). LangChain scored higher on correctness with a 3B model because it never refuses; neither framework gives citations to check |
| The router | Helped on questions it had not seen before |
| Graph and agentic | Not ready for daily use on small models: graph refuses often, agentic is slow |

In short: the retrieval engine and the platform work; answer quality depends mostly on the model you
connect. Use a strong model for answers people will rely on, Traditional, Vectorless and `auto` for
everyday questions, and treat Graph and Agentic as experimental for now.

## Which strategy to use

| Your question looks like | Use |
|---|---|
| An everyday question in plain words | Traditional, or `auto` |
| An ID, a code, an exact name or phrase | Vectorless |
| Not sure, or a mix of questions | `auto` |
| Who relates to whom, across documents | Graph, once a graph is built, and check the answer |
| A comparison or a question with several parts | Try Agentic on a strong model; on a small model Traditional is better today |

To find out for your own documents, write a few questions with known answers and run
`ragfabric eval run`; the Evaluation page then shows which strategy wins on your data. See
[evaluation.md](evaluation.md).

## Where to go next

| To | Read |
|---|---|
| Install and try it | [Quick start](getting-started.md) |
| Understand the design | [architecture.md](architecture.md) |
| Tune it | [configuration.md](configuration.md) and `ragfabric.example.yaml` |
| See the measured results | [benchmarks/comparison.md](benchmarks/comparison.md) |
| See how it compares with other tools | [benchmarks/landscape.md](benchmarks/landscape.md) |
| Fix a problem | [troubleshooting.md](troubleshooting.md) and [failure-cases.md](failure-cases.md) |
