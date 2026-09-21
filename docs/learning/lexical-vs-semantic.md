# Lexical versus semantic search, learning notes

> Status: Phase 4 (shipped). A plain-language companion to
> [concepts/lexical-vs-vector.md](../concepts/lexical-vs-vector.md). That document is grounded in
> RagFabric's implementation: the formula, the schema, the SQL, the real numbers. This one is about
> the idea, and assumes no knowledge of the codebase.

## The one sentence version

**Lexical search finds what you named. Semantic search finds what you meant.** Most of what follows
is working out what that implies.

## Two ways of deciding a document is relevant

Imagine a filing cabinet and two clerks.

The first clerk has never read anything in the cabinet, but has a complete index of which words
appear in which folders, and how many folders contain each word. Ask for "ERR_QUOTA_4419" and this
clerk is instant and exactly right, because that string sits in one folder and nowhere else. Ask
for "why does the export keep giving up" and the clerk looks for folders containing "export",
"keep" and "giving", finds the wrong ones or none, and cannot do better, because it does not know
what any of the words mean.

The second clerk read the whole cabinet before you arrived, along with a very large number of
other documents, and remembers roughly what each folder is about. Ask "why does the export keep
giving up" and it hands you the folder about retry limits without hesitation, because it understands
those to be the same subject. Ask for "ERR_QUOTA_4419" and it says, honestly, that the string means
nothing to it, and offers you some folders about errors and quotas, one of which may be the right one
by luck.

Lexical search is the first clerk. Semantic (vector) search is the second. Neither is a
better version of the other.

## What "semantic" actually rests on

Semantic search works by turning text into a list of numbers, a vector, using a model trained on a
very large amount of text. The training is what makes it work, and the training is also its limit.

The model is good at a phrasing difference it has seen many times before: "car" and "automobile",
"how long before it stops" and "what is the timeout". It is poor at anything it never saw. An
internal error code, a customer's part number, a product codename invented last quarter, a term of
art from a small professional field: the model has no learned representation of these, so it breaks
them into fragments and produces a vector describing the fragments. That vector is not wrong in an
interesting way. It is close to meaningless.

This is the single most useful thing to understand about vector search, because it is invisible from
the outside. Vector search always returns something, ranked confidently, whether or not it
understood the question.

## What "lexical" actually rests on

Lexical search counts. It has two counts and one measurement.

| Quantity | Question it answers | Why it matters |
|---|---|---|
| Term frequency | how often does this word appear in this document | a document mentioning your word repeatedly is more likely to be about it |
| Document frequency | how many documents contain this word at all | a word in every document distinguishes nothing; a word in one document distinguishes everything |
| Document length | how long is this document | long documents contain more of everything, so they match more queries by accident |

Document frequency is the interesting one, and it is where the word "rare" earns its keep. A word
occurring in one document out of a hundred thousand is an extraordinarily strong signal. A word
occurring in all of them is worth nothing. BM25, the standard scoring function and the one RagFabric
uses, expresses that as a factor called IDF, inverse document frequency.

The crucial property: **these counts come from your corpus, not from a model's training data.** Your
error codes are rare in your corpus, so the counting finds them, and it would have found them equally
well in a language nobody has built an embedding model for, or in a corpus of vocabulary invented
this morning. No training, no vendor, no GPU, no per-query cost.

Term frequency also does not grow without bound in the score. Mentioning a word four times helps more
than mentioning it once, but the fourth mention helps much less than the first, and the fortieth
barely at all. Otherwise a page repeating one word would beat a page that actually answers the
question.

## Where lexical wins

Each of these has the same shape: the user already knows the exact string, and the string is rare.

- **Identifiers and codes.** Error codes, function names, CVE numbers, ticket references, SKUs, part
  numbers, order numbers.
- **Quoted phrases.** Quotation marks are the user stating that the exact wording matters. Semantic
  search cannot honour that, because it compares meanings and the nearest match by meaning may not
  contain the phrase.
- **Rare proper nouns.** A person, a customer, a project codename. Rarity is what IDF measures.
- **Numbers and version strings.** `2.1.4` and `2.14.0` are nearly identical to an embedding model
  and completely different to the person asking.
- **Jargon-heavy, internal or low-resource corpora.** Wherever the vocabulary is not vocabulary a
  general model was trained on.
- **When the answer has to be explainable.** A lexical score decomposes into "this document ranked
  first because this term appears in one document out of six". A similarity of 0.83 does not
  decompose into anything.
- **When there is no budget.** No model, no API key, no rate limit, no cost per query on retrieval.

## Where lexical loses

Each of these has the same shape too: the question and the answer are about the same thing and share
no words.

- **Paraphrase.** "How long before it gives up" against "the retry limit is five attempts".
- **Synonymy.** "invoice" and "bill", "postcode" and "ZIP code", "SSO" and "single sign-on". One
  concept, two disjoint sets of words. Stemming will not save you: stemming reduces "retrying" to
  "retry", not "bill" to "invoice".
- **Answers that never repeat the question's words.** "Why is this slow?" answered by a passage about
  lock contention. The passage is correct and shares nothing with the question.
- **Conceptual or comparative questions.** "What are the trade-offs", "how does this differ from the
  other approach". These describe the shape of the answer, not its vocabulary.
- **Cross-lingual search.** A question in one language against documents in another shares no words
  at all.
- **Newcomers.** Someone unfamiliar with a subject asks in their own words. The documents answer in
  the subject's own words. That gap is exactly what semantic search bridges.

## The failure modes are shaped differently, and this matters

When lexical search fails, it usually returns **nothing**, or an obviously irrelevant handful. Zero
shared words means zero evidence, and the ranker has nothing to rank.

When semantic search fails, it returns **something plausible**. It is always able to find the nearest
vectors; whether they are relevant is a separate question the score does not answer honestly.

An empty result set is easy to notice, easy to log and easy to act on. A confidently ranked wrong
passage is none of those things. That difference is worth as much as any accuracy comparison when you
are deciding what to run in production, and it is one of the reasons a lexical path is worth having
even in a deployment whose default is semantic.

## Why RagFabric runs two lexical rankings and combines them

BM25 is a bag of words: it knows which words a document contains and how often, and nothing at all
about their order or how close together they are. "retry limit" scores identically whether the two
words are adjacent or four paragraphs apart.

PostgreSQL's own ranking function, `ts_rank_cd`, does account for proximity, but has no notion of how
rare a word is across the corpus. So it cannot tell a unique error code from the word "the".

Each knows something the other does not. RagFabric runs both and merges the two ranked lists by
position rather than by score, because the two scores are on scales that cannot be compared: one is
unbounded and depends on the corpus, the other lives roughly between 0 and 1. Positions compare
cleanly. Second place is second place whatever the numbers were.

A separate step then boosts documents containing a quoted phrase verbatim, or an identifier in its
exact written form, since those are the two things counting alone gets wrong. That step can only
reorder documents already retrieved; it can never pull a new one in, because retrieval is where
permission checking happened.

## Choosing

Ask what your users type.

| If your users mostly type... | Prefer |
|---|---|
| identifiers, codes, names, version numbers, quoted strings | lexical |
| questions in their own words about concepts | semantic |
| the exact vocabulary the documents use | lexical |
| vocabulary the documents do not use | semantic |
| in a language the documents are not written in | semantic |
| both, depending on the day | choose per request |

RagFabric does not make you pick once. The strategy is selected per request, so the same deployment
can answer an identifier lookup lexically and a conceptual question semantically. The concept
document has the exact API, CLI and SDK spellings, and a worked example with real measured numbers
showing both a query lexical search wins outright and a query it returns nothing at all for.

## What this document deliberately does not say

It does not claim a percentage. Not "BM25 is 20% better on identifier queries", not a recall figure,
not a latency number. Which of these two approaches wins on a given corpus depends on that corpus and
on the questions real users ask of it, and the only way to know is to measure it on yours. Quoting a
number from someone else's benchmark would suggest a precision that does not exist.
