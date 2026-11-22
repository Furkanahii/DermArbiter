# Text-only harness benchmark: runbook

`dermarbiter/research/text_benchmark.py` runs every harness arm on identical
text input, with fixed call budgets, a locked model version, no retries, no
fallback, no tools and no images. Two task types share the runner.

## Tracks (built 22 Nov 2026)

| Track | Builder | Schema | Cases (dev / test) | Gold | Answer-blind baseline (test) |
|---|---|---|---|---|---|
| A — SCIN self-report | `scin` | `text-dx-v1` (rank ≤3 of 30 candidates) | 2,154 (417 / 1,737) | dermatologist weighted labels | constant "Eczema" 25.5%, random 4.4% |
| B — MedMCQA Skin (train) | `medmcqa` | `text-mcq-pilot-v1` (A–D) | 1,269 (257 / 1,012) | published key | constant 27.1%, random 25% |
| B — MedXpertQA Integumentary | `medxpertqa` | `text-mcq-pilot-v1` (A–J) | 48 (test only) | published key | constant 20.8%, random 10% |

Provenance (source hashes, exclusions, vocabulary, baselines) is tracked in
`data/text_benchmark/*.report.json`. The JSONL files are rebuilt locally and
not committed (source licenses differ).

Design choices to report in the paper:

* **SCIN** labels were assigned by dermatologists from images; text-only accuracy
  has a low ceiling by design. Top-1 counts as correct if it matches any tied
  highest-weight label (925 of 3,061 labeled cases have tied top labels);
  `top1_soft` gives that label's weight. Vocabulary = 30 most frequent top labels.
  Self-reports are templated from form fields; race is kept in metadata only.
* **MedMCQA** options are shuffled per question (seeded) because the published
  key is skewed (~35–40% "A"). Train split ⇒ contamination is possible.
* Splits are hash-based and grouped by normalized visible text, so identical
  inputs never cross dev/test. Use dev for prompt freezing, test once.

### Build

```bash
# SCIN metadata (no images): https://github.com/google-research-datasets/scin
curl -O https://storage.googleapis.com/dx-scin-public-data/dataset/scin_cases.csv
curl -O https://storage.googleapis.com/dx-scin-public-data/dataset/scin_labels.csv
python scripts/build_text_benchmark.py scin --cases scin_cases.csv --labels scin_labels.csv \
    --output data/text_benchmark/scin_text_k30.jsonl

# MedMCQA (Apache-2.0), pinned revision
curl -L -o medmcqa_train.parquet https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/91c6572c454088bf71b679ad90aa8dffcd0d5868/data/train-00000-of-00001.parquet
python scripts/build_text_benchmark.py medmcqa --parquet medmcqa_train.parquet \
    --revision 91c6572c454088bf71b679ad90aa8dffcd0d5868 --output data/text_benchmark/medmcqa_skin.jsonl

# MedXpertQA (MIT), pinned revision
curl -L -o medxpertqa_text_test.jsonl https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/resolve/7e7c465a68eb2b866926bfa59c8c9d17a8daba65/Text/test.jsonl
python scripts/build_text_benchmark.py medxpertqa --jsonl medxpertqa_text_test.jsonl \
    --revision 7e7c465a68eb2b866926bfa59c8c9d17a8daba65 --output data/text_benchmark/medxpertqa_derm.jsonl
```

Compare the printed `output_sha256` with the tracked report. After the team has
reviewed a build, rebuild with `--data-status frozen --accepted-by NAME`; live
runs refuse non-frozen data.

## Harness arms

| Arm | Calls | What it isolates |
|---|---:|---|
| `direct` | 1 | raw backbone |
| `structured` | 1 | prompt structure |
| `self_refine` | 3 | extra compute, same agent |
| `vote3` / `vote6` | 3 / 6 | sampling only (plurality / Borda) |
| `isolated_refine` | 6 | debate topology without peer information |
| `debate` | 6 | homogeneous peer communication |
| `role_vote` | 3 | DermArbiter personas (specialist / generalist / skeptic), no communication |
| `role_debate` | 6 | personas + one simultaneous revision round seeing named peers |
| `dermarbiter` | 7 | `role_debate` + moderator makes the scored decision |

Personas are condensed from `dermarbiter/core/prompts/*.txt` (tool instructions
removed). Ties abstain in every aggregating arm. Pre-specified comparisons:
`debate`–`isolated_refine`, `debate`–`vote6`, `role_vote`–`vote3`,
`role_debate`–`debate`, `dermarbiter`–`role_debate`, each arm–`direct`.

## Providers

* `--provider gemini` — Gemini REST, key from `GEMINI_API_KEY`.
* `--provider openai_compatible --base-url URL` — OpenAI or any compatible
  server, e.g. vLLM on Colab for open-weight backbones
  (`vllm serve Qwen/Qwen3-8B` → `--base-url http://localhost:8000/v1`).
  Key from `OPENAI_API_KEY` or `--api-key-env NAME`; none needed for local servers.

`--expected-model-version` must equal the version string the provider returns
(Gemini `modelVersion`, OpenAI `model`); a mismatch is recorded as an error.

```bash
python scripts/run_text_benchmark.py plan data/text_benchmark/scin_text_k30.jsonl --split dev --method dermarbiter
python scripts/run_text_benchmark.py run  data/text_benchmark/scin_text_k30.jsonl --split dev --method dermarbiter \
    --provider openai_compatible --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-8B --expected-model-version Qwen/Qwen3-8B \
    --max-calls 2919 --output-dir runs/scin_dev/qwen3-8b/dermarbiter_r0
python scripts/run_text_benchmark.py score data/text_benchmark/scin_text_k30.jsonl --split dev \
    --predictions runs/scin_dev/qwen3-8b/dermarbiter_r0/predictions.jsonl
```

## Pilot (all arms, one command)

```bash
export GEMINI_API_KEY=...   # your own terminal only; never commit or paste it
# 1. One call: see which version string the provider returns
python scripts/run_text_benchmark.py probe data/text_benchmark/scin_text_k30.jsonl --split dev --model gemini-2.5-flash
# 2. Plan, then run all 10 arms on 20 dev cases (840 calls per track)
python scripts/run_pilot.py data/text_benchmark/scin_text_k30.jsonl --split dev --limit 20 --dev-pilot \
    --model gemini-2.5-flash --expected-model-version <returned_model_version> \
    --out runs/pilot/scin_dev20 --plan-only
python scripts/run_pilot.py data/text_benchmark/scin_text_k30.jsonl --split dev --limit 20 --dev-pilot \
    --model gemini-2.5-flash --expected-model-version <returned_model_version> \
    --out runs/pilot/scin_dev20
```

`--limit N` picks a fixed, hash-ordered subset (same N ⇒ same cases).
`--dev-pilot` allows unreviewed data on dev only; such runs are marked
`publication_eligible=false`. The pilot writes `summary.md` / `summary.json`
with accuracy, coverage, calls, tokens and latency per arm, and the difference
to `direct` with a case-level paired bootstrap 95% CI and exact McNemar p.
Re-running the same command skips completed arms.

What to read in the pilot: parse/format error rate per arm, abstention from
vote ties (role arms with three disagreeing agents abstain), tokens per case
(to price the full run), and whether any arm is clearly broken. Do not tune
prompts on test, and do not draw conclusions from 20 cases.

Offline check (22 Nov): a local stub server answering "Eczema" to every case
scored exactly the dev constant baseline (25.9%) through `direct` and
`dermarbiter`, confirming builder → runner → provider → scorer wiring. No live
model has been run yet.

---

# Appendix: original MCQ pilot notes (21 Nov)


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
