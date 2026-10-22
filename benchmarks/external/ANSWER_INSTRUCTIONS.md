# Instructions for an answering agent

Use these with `ragfabric eval export-prompts` and `ragfabric eval import-answers`. Background and
limits: [Scoring answers from outside RagFabric](../../docs/evaluation.md).

## The instruction text

Copy this block into the call for each batch, then append the questions of that batch, each as its
`id` and its `messages` (system and user) from `prompts.jsonl`. Do not include `ground_truth` or the
rest of the line.

```text
You are answering questions for a retrieval benchmark. Each question below comes with the exact
messages (a system message and a user message) that a model would be sent. Follow them as written.

Rules:
1. Use ONLY the numbered passages inside the given messages. Do not use any tool. Do not read any
   file. Do not search the web. Do not use what you know from elsewhere.
2. Cite every claim with the passage number in square brackets, such as [1], as the system message
   says. Quote verbatim only inside double quotation marks.
3. If no passage answers the question, reply with exactly this sentence and nothing else:
   I could not find an answer to that in the documents provided.
4. Answer every question, once.
5. Output exactly one JSON object per question, one per line, no other text, no code fence:
   {"id": "<the question id>", "answer": "<your answer text>"}
   You may add "model" (the model name you believe you are) when you know it. Do not add token
   counts unless you were given them.
```

## Operator checklist

1. Export once per strategy and top-k: `ragfabric eval export-prompts ... --out prompts.jsonl`.
   Do not edit the file; its hash is recorded at import.
2. Split the questions into batches of 10 to 20 per call. Give each call only the `id` and
   `messages` of its questions, plus the instruction text above.
3. Record the model alias you asked for, the date, and the exact instruction text used.
4. Save every raw output unedited, one file per call, next to the prompts file.
5. Concatenate the outputs into `answers.jsonl`. Do not fix, reorder or fill in answers by hand;
   import reports a missing, duplicate, unknown or unreadable line as a named error.
6. Import with a label that names the model alias and the date, for example
   `--label "hosted model, alias A, 22 October 2026"`, and a batch name per round.
7. Use the same `--judge` as the local runs you compare against.
8. When quoting results, give the label and date and say the answers were written outside RagFabric.
   They are not a controlled run and not an API test.
