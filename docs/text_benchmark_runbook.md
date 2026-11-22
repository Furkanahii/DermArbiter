# Text MCQ pilot: implementation and reproduction

This pilot is separate from the legacy mock orchestrators. It supports seven conditions in `dermarbiter/research/text_benchmark.py`. It does not yet implement RAG, open-ended clinical scoring, paired statistical intervals, a second API provider, or a dollar budget cap. No live provider call was made during development.

## Environment

Use the repository's Python environment and dependencies. The existing package initializer imports the legacy orchestrator, so importing the new module still requires the project's core dependencies. `pyarrow` is needed only for the optional MedMCQA converter. The Gemini adapter uses Python's standard-library HTTP client.

## Data contract

Every JSONL object uses `schema_version: text-mcq-pilot-v1`, `modality: text`, a unique `case_id`, a `source_id` grouping related questions, `source_url`, `license`, `split` (dev/test), `text_origin`, `gold_origin`, `data_status`, `question`, `options` (A–J keys) and `answer` (one option key). Supported text origins: published_exam_question, native_clinical_text, fixture. Gold origins: published_answer_key, clinician_review, fixture. Clinician review needs reviewer_id. Data status: pilot_candidate, frozen, fixture.

Only question and options reach the model. Source metadata, the answer and source explanations stay outside prompts. Structural checks do not establish clinical validity, truthful provenance, absence of answer leakage, near-duplicate separation or training contamination. Review these separately. Live execution rejects candidate and fixture data. `frozen` is a declaration made after the dataset acceptance process, not a way to bypass it.

The earlier `scripts/validate_text_benchmark.py` is a different, open-ended vignette preflight. Its JSONL is not interchangeable with this MCQ schema. Legacy experiment runners remain mock-only.

## Prepare and plan

```bash
python scripts/prepare_medmcqa_pilot.py validation.parquet \
  --revision 91c6572c454088bf71b679ad90aa8dffcd0d5868 \
  --output pilot_candidates.jsonl

python scripts/run_text_benchmark.py plan pilot_candidates.jsonl \
  --split dev --method debate
```

The converter accepts a local validation file and preserves the upstream question/options. It does not download data, invent labels or silently map the public test's `-1` labels. Output is 11 candidate questions for the inspected upstream revision. Review before making an accepted dataset. No prompt selection should use the eventual test set.

## Live pilot, after data acceptance

Provide `GEMINI_API_KEY` through the local environment, never in tracked files or chat. Replace the explicit model placeholders below with verified available IDs. The expected version must match the provider's returned `modelVersion`; a mismatch records an error rather than falling back to another model. Seed support is not a determinism guarantee.

```bash
python scripts/run_text_benchmark.py run accepted_dev.jsonl \
  --split dev --method debate \
  --model MODEL_ID --expected-model-version RETURNED_VERSION \
  --max-calls 66 --output-dir runs/debate_dev_r0
```

`--max-calls` must cover the planned call count. It is not a monetary cap. The output directory must not already exist, preventing silent overwrite. Each completed case is flushed to predictions.jsonl. There is no automatic retry or resume, so interrupted runs must be audited before a separately identified rerun. manifest.json stays `started` on interruption. A failed provider call can have unknown billable usage.

The manifest records dataset, code and config hashes. Per-case traces contain input prompts, parsed answers, raw returned text, model version, generation seed, usage and elapsed time. Thinking text is not requested or logged; its token metadata is retained when the provider reports it. Do not use private patient data in this pilot.

## Score and interpret

```bash
python scripts/run_text_benchmark.py score accepted_dev.jsonl \
  --split dev --predictions runs/debate_dev_r0/predictions.jsonl
```

Scoring requires all planned cases, not just answered cases. Missing IDs count in the denominator; duplicates, unknown IDs, input changes and mixed method/model/repeat configs raise errors. The scoring CLI also checks the exact dataset file hash, including gold and metadata. Evaluate each condition separately. These hashes verify file identity, not the clinical correctness of the answer key.

Errors stop the current case without retry; partial panels do not get a successful final answer. Invalid JSON, extra fields, nonfinite confidence, truncated generations and version mismatch remain visible. Ties abstain. Vote agreement is distinct from probability. Revision counts are descriptive and not a causal estimate. Six calls in each arm do not imply equal total tokens or latency. All runtime calls are serial in this implementation.

## Offline validation

```bash
python -m pytest tests/test_text_benchmark.py tests/test_research_readiness.py -q -o addopts=
```

These tests use fake provider responses to verify experimental integrity; they do not measure medical performance or prove that a current hosted model accepts the live request. A small accepted-data API smoke test is still required before an experiment.
