# A short walkthrough of EvidenceDesk

This is the sequence I'd use to show the project in an interview or a short screen recording. The staging service is at https://evidencedesk-ai-staging-bge.onrender.com. It may take a moment to wake up on Render's free tier.

## Start with the problem (about 30 seconds)

Explain what you were trying to solve: matching a question to a document isn't very useful unless the application can point to the exact supporting text. The other half of that problem is knowing when the documents don't contain an answer.

## Show a real answer (about a minute)

Ask: **“Can plane tickets be booked outside the corporate tool?”**

The travel policy should provide the line about booking through the company portal. Point out that the result includes the original sentence and the policy title, not an invented summary.

Then ask: **“Where should live system passwords be saved?”** The security policy should give the rule about approved secrets managers. These two queries helped reveal weaknesses in earlier retrieval versions, which makes them useful examples to explain.

## Show a question it cannot answer (about 30 seconds)

Ask: **“How many paid vacation days do employees get?”**

There's no vacation rule in the four demo documents. The expected response is an honest “I couldn't find enough supporting evidence,” with no source citation. That's an important part of the design.

## Open the evaluation dashboard (about a minute)

Show both the passing and failing cases, not just the big numbers. Explain that 30 questions cover direct wording, paraphrases, and deliberately unanswerable queries. Another 14 questions provide a separate development check over the same documents.

The numbers are helpful for catching regressions, but this is not an independent evaluation on real organizational files. The latency shown is warm local retrieval time recorded during the build, not the time a visitor waits on Render.

## Finish with the engineering choices (about 30 seconds)

Point to the [architecture diagram](ARCHITECTURE.md). Mention FastAPI, React, SQLite, local BGE embeddings, TF-IDF, citation handling, the Docker build, and the GitHub Actions tests that run with a 512 MiB limit.

The next technical steps would be testing genuinely new documents, checking multi-sentence support, and improving security before any real document ingestion.

## Screenshots to capture

Take your own screenshots of a cited answer, an abstention, the evaluation dashboard, and the passing GitHub Actions checks. Keep private information out of the frame. Add the images under `docs/screenshots/` once they're captured; don't use recreated screenshots in place of the actual running app.
