# An improved agentic harness for medical cases: a dermatology use case with call-matched comparisons on text-only patient reports

*Working draft, 22 November 2026. Sections 1–3 are complete. Results (§4) are
placeholders until the frozen test runs (planned 7–13 December).
Statements marked [TODO] need data or a decision.*

## Abstract

> [TODO: write last.] Template: Multi-agent LLM systems for diagnosis usually
> report gains over a single call without controlling for the extra compute. We
> benchmark ten inference harnesses, from one call to a role-based panel with peer
> revision and a moderator, on three text-only dermatology tracks: patient
> self-reports with dermatologist panel labels (SCIN), exam questions (MedMCQA Skin)
> and hard expert questions (MedXpertQA). We use two open-weight backbones. Each
> harness is compared with a control that uses the same number of calls. [Main
> finding.] [Variance share of harness vs. model.] [Cost.] Code, prompts and run
> manifests are released.

## 1. Introduction

LLM agents in medicine are increasingly built as *harnesses*: scaffolding around a
fixed model that decides how many times it is called, with which role, and what it
sees of earlier outputs. Recent controlled studies show that the harness alone can
move results. On 107 medical tasks, about a quarter of the outcome variance came from
the harness and its interaction with the model, and no single harness was best
everywhere (MedicalHarness; Wang et al., 2026, arXiv:2610.05778). In parallel,
general-domain work questions whether multi-agent debate helps once extra compute is
matched. Isolated self-correction beats unguided homogeneous debate (Bertalanič &
Fortuna, 2026, arXiv:2605.00914). Debate gains depend on whether the right answer is
proposed at all and on how it is read out (Zhao et al., 2026, arXiv:2610.04686).
Agent systems for clinical decision tasks give modest gains at more than ten times the
token cost (Liu et al., 2026, *npj Digit Med*).

Dermatology agents published in 2026 are image-centred and are evaluated against
single-model baselines (DermAgent, arXiv:2605.14403; SkinGPT-X, arXiv:2603.26122;
SkinMind, arXiv:2611.08935). A recent review notes that no end-to-end validated
dermatology agent exists yet (Yan et al., 2026, *Cell Rep Med*). Large multimodal
models have also been reported to rely on the text of dermatology cases more than on
the image (Nouyed et al., 2026, *JAAD*). This makes a text-only, controlled harness
comparison a necessary first step before images are added.

We do not propose a new model or claim a best system. We build **a set of harnesses
that differ one design choice at a time** and compare them under the same backbone,
the same visible case text, the same output schema and explicitly counted call
budgets. DermArbiter's role panel with a moderator is one member of the set. The
other arms are the controls that isolate each of its components.

**Contributions.**

1. A ladder of ten harness arms with call-matched controls that separates sampling
   (voting), extra computation (self-refinement), peer information (debate vs.
   isolated refinement), role personas and a moderator decision.
2. A text-only dermatology benchmark with three tracks: patient self-reports with
   dermatologist panel labels (SCIN), exam questions (MedMCQA Skin) and hard expert
   questions (MedXpertQA). The builds are deterministic and hash-pinned, and trivial
   baselines are reported.
3. A pre-registered analysis: paired tests with Holm correction for five
   pre-specified comparisons, all planned cases in the denominator, and cost (calls,
   tokens, latency) in the main table.
4. A case × model × harness variance decomposition comparable to MedicalHarness,
   across two open-weight backbone families. [TODO: report whether harness rankings
   transfer.]

## 2. Related work

**Harnesses as an object of study.** Several 2026 studies treat the harness as an
object of study in its own right:

- **Harness synthesis and evolution.** AutoHarness synthesizes code harnesses
  automatically (arXiv:2603.03329). Wang et al. re-evaluate harness evolution under
  equal budgets and held-out tests (arXiv:2607.12227).
- **Infrastructure.** AgentCompass separates benchmark, harness and environment
  (arXiv:2607.13705).
- **Blame attribution.** Zhang et al. attribute agent failures to the model or to the
  harness (arXiv:2610.32483).
- **Medical tasks.** MedicalHarness (arXiv:2610.05778) is the closest work: five open
  models × five single-agent harnesses that differ in context management, planning
  and tool access, across 107 medical tasks. MedHarnessBench adds process metrics for
  EHR and tool tasks (arXiv:2610.15714).

Our harnesses are multi-agent reasoning schemes without tools, and our task is
diagnosis within a single specialty.

**Multi-agent debate and its controls.** Isolated self-correction prevails over
homogeneous debate at equal budget (arXiv:2605.00914). Cognitive diversity through
personas, temperature or different models changes debate outcomes in small models
(arXiv:2609.35875). Proposal supply and verification-aware readout explain when
debate helps (arXiv:2610.04686). Agreement between agents can hide misaligned reasoning
(arXiv:2606.08457). In medicine, MedMASLab compares 11 multi-agent architectures with
per-call cost accounting (arXiv:2603.09909). Structured debate and mixture-of-agents
frameworks target complex diagnosis (arXiv:2609.05069), and uncertainty-aware routing
decides when to convene a panel (arXiv:2610.33544).

**Dermatology agents and benchmarks.** The 2026 dermatology agents are image-centred:
DermAgent, SkinGPT-X, SkinMind and DERM-3R (arXiv:2604.09596). Benchmarks include:

- DermaBench: clinician-annotated VQA on DDI (Yilmaz et al., arXiv:2601.14084).
- DermEVAL: 16 diseases, 11,347 images (WACV 2026).
- DermoBench (arXiv:2601.01868).
- DermCase: rare-disease case reports (arXiv:2603.18418).
- A skin-tone counterfactual stability benchmark (Findings IJCNLP-AACL 2026).

We found no controlled harness comparison on text-only dermatology cases. Our stage 2
adds images to the same SCIN cases and connects to DermaBench.

## 3. Methods

### 3.1 Tasks and data

| Track | Task | Source | Cases (dev / test) | Constant baseline (test) | Uniform random (test) |
|---|---|---|---|---|---|
| SCIN-text | rank ≤3 of 30 candidate conditions | SCIN patient self-reports, no images; dermatologist panel weights | 417 / 1,737 | 25.5% | 4.4% |
| MedMCQA-Skin | single-answer MCQ, 4 options | MedMCQA Skin subset (Apache-2.0) | 257 / 1,012 | 27.1% | 25.0% |
| MedXpertQA-Derm | single-answer MCQ, 10 options | MedXpertQA dermatology (MIT) | – / 48 | 20.8% | 10.0% |

**SCIN-text.**

- *Case selection:* cases with at least one dermatologist label.
- *Case text:* rendered deterministically from the structured self-report fields
  (age band, sex at birth, self-reported Fitzpatrick type, description, duration,
  location, texture, local and systemic symptoms). Cases with fewer than two reported
  fields are excluded (398).
- *Candidates:* the 30 most frequent labels. Cases whose reference is outside this
  vocabulary are excluded (509).
- *Reference:* dermatologist weights renormalized to sum to one. All top-weighted
  labels count as correct under ties (697 cases). In 274 cases a tied top label lies
  outside the candidates, which caps the soft score below one.

**MedMCQA-Skin.**

- *Exclusions:* multi-answer items (491), items with empty or duplicate options (7)
  and image-dependent items (4).
- *Option order:* shuffled per item with a fixed seed. The original key is skewed
  toward "A" at 35–40%.

**All tracks.**

- *Splits:* dev/test splits use a salted hash grouped by normalized visible text, so
  identical inputs never cross splits.
- *Provenance:* source revisions and file hashes are pinned.
- *Data freeze:* test runs are refused by code until a human review freezes the data
  (Appendix A).

### 3.2 Harness arms

All arms use one backbone, temperature 0.7, at most 1,024 output tokens per call, no
tools, no retrieval and no retries. Prompts state that the case text and peer outputs
are data, not instructions.

| Arm | Calls | Procedure | Isolates |
|---|---:|---|---|
| direct | 1 | answer | backbone |
| structured | 1 | findings → alternatives → answer | prompt structure |
| self_refine | 3 | structured, then two self-reassessments | sequential extra compute |
| vote3 / vote6 | 3 / 6 | independent structured samples, aggregated | sampling |
| isolated_refine | 6 | 3 samples, each reassesses only itself, aggregated | debate topology without peers |
| debate | 6 | 3 samples, each reassesses seeing the two peers (one simultaneous round), aggregated | peer information |
| role_vote | 3 | specialist, generalist, skeptic personas, aggregated | personas |
| role_debate | 6 | role_vote + one simultaneous peer revision with named roles | personas within debate |
| dermarbiter | 7 | role_debate + a moderator sees initial and revised panels and makes the scored decision | moderator |

- **Personas:** condensed from the DermArbiter system prompts, with tool
  instructions removed.
- **Aggregation:** plurality for MCQ; a tie abstains. For SCIN, Borda count over
  the ranked differentials (3/2/1 points). A tie on score for first place is broken
  by the number of first-place votes; a tie on both abstains. Without this
  tie-break, a condition that every agent ranked second could tie a 2-of-3 top-1
  majority (3+3 = 2+2+2), which made the panel abstain in 15 of 26 pilot abstentions
  (Appendix B).
- **Seed pairing:** call seeds depend only on the case, the repeat and the call
  index, so the first three calls of vote3, vote6, isolated_refine and debate are
  identical samples. Differences between these arms therefore come from what happens
  after sampling.
- **Failures:** a failed call (transport error, invalid output, truncated
  generation, unexpected model version) fails the whole case. This counts as
  incorrect, with no partial credit.

### 3.3 Output contract

Every call must return JSON with three fields:

- **Answer:** an option key (MCQ) or a list of ≤3 candidates (SCIN); an empty list
  or null means abstain. A repeated candidate is dropped, keeping the first
  occurrence, because grammar-constrained servers do not enforce `uniqueItems`.
  This never changes the top-1 answer, and the number dropped is logged.
- **Confidence:** a number in [0, 1]. On abstention it is ignored and stored as null.
- **Rationale:** at most 1,000 characters.

The same per-case JSON Schema is enforced by constrained decoding for every arm.
Parsing is strict otherwise: no repair of malformed output and no fuzzy matching.
Each constraint was added after a pilot failure on dev data, before any test run:

- Empty objects under plain JSON mode.
- A null confidence attached to an answer.
- A rationale that looped until the token limit.

These failures are reported in Appendix B because they affect arms unequally: a
looping call fails every multi-call arm on that case.

### 3.4 Backbones and serving

Two open-weight backbone families, served locally through an OpenAI-compatible
endpoint (Ollama). In the pilot, 3–4B models scored at the constant-answer
baseline on SCIN-text (Appendix B), which leaves no room to see harness effects.
The main runs therefore use backbones of at least 7B. [TODO: final choice on dev,
e.g. Qwen 2.5 7B and Gemma 3 12B.]

- **Version check:** the returned model string must equal the pinned one.
- **Hardware:** free GPU notebooks; cases are sent concurrently. The runner's
  concurrency does not change prompts, seeds or record order (verified against a
  deterministic server). Batched GPU inference can still be numerically
  nondeterministic, so every arm of a backbone runs with the same server settings
  and concurrency. [TODO: measure repeat-to-repeat agreement on dev.]

### 3.5 Metrics and statistics

**Primary metric.** Top-1 accuracy over all planned cases. Missing, failed and
abstained cases count as incorrect.

**Secondary metrics.**

- SCIN: top-3 accuracy and a soft score (weight of the predicted top-1).
- Answer coverage.
- Calls, tokens and latency per case.
- Beneficial and harmful revisions (initial vs. final correctness; cf.
  arXiv:2610.04686).

**Primary comparisons.** Five comparisons, Holm-corrected within each backbone and
track. Each pairs adjacent steps of the ladder:

- debate − isolated_refine
- debate − vote6
- role_vote − vote3
- role_debate − debate
- dermarbiter − role_debate

**Tests.** Each comparison uses a paired bootstrap 95% CI (5,000 case resamples) and
an exact McNemar test. Every arm versus direct is reported as a secondary analysis.

**Variance decomposition.** Sums of squares of top-1 correctness over a fully crossed
case × model × harness design. The harness-attributable share is harness +
model×harness + case×harness.

**Sample size.** With a discordant-pair rate p and α = 0.01 (Holm, five tests), 80%
power detects a difference of about 3.42·√(p/n). The dev pilot showed p = 7% over
the 20 primary comparisons (14 of 200 pairs; 3% on SCIN, 11% on MedMCQA). Small
backbones near the floor understate disagreement, so we plan with p = 20%: about
6.2 points with 600 SCIN test cases and 7.6 points with 400 MedMCQA cases (3.7 and
4.5 points if p stays at 7%).

## 4. Results

[TODO after test runs.] Planned layout:

- **Table 2:** main grid of arm × backbone × track, with top-1 (CI), coverage,
  calls, tokens and latency.
- **Table 3:** the five primary comparisons.
- **Figure 2:** accuracy vs. tokens per case (cost frontier).
- **Table 4:** variance decomposition.
- **Figure 3:** revision transitions.

## 5. Discussion

[TODO.] Points to address:

- Whether gains, if any, come from sampling or from communication.
- Whether personas help beyond diversity of samples.
- Whether the moderator helps or merely adds a call.
- Rank stability across backbones.
- What a text-only ceiling on SCIN self-reports implies for stage 2 (images).

## 6. Limitations

- **Patient self-reports:** SCIN text is self-reported, not a clinical vignette, so
  absolute accuracy is low by design.
- **Exam items:** MedMCQA tests knowledge rather than case reasoning and has known
  label noise.
- **Backbone size:** only open-weight backbones that fit free hardware are used;
  harness effects may differ for frontier models.
- **Budget matching:** call matching is not token matching, so tokens are reported
  separately.
- **Single setting:** one temperature and one revision round are used.
- **Pilot-driven constraints:** the output constraints were introduced in response to
  pilot failures. They are fixed before test, but they are part of the method.
- **Retrospective benchmark:** this is not a clinical evaluation (cf. Yan et al.,
  2026).

## Appendix A. Data review and freeze

See `docs/data_review_tr.md`. The acceptance threshold (≤10% flagged rows in a fixed
sample of 50 per track) was fixed before review.

## Appendix B. Pilot failures and fixes

**Setup.** Two backbones (Gemma 3 4B, Qwen 2.5 3B) × two tracks (SCIN-text, MedMCQA-skin) ×
10 arms × 10 fixed dev cases, served locally by Ollama on an Apple M2. That is 40
runs, 400 case-runs and 1,673 calls at about 19 s per call. All runs share one code
hash and one schema hash. The pilot tests robustness, not accuracy.

**Failures and fixes.** Each fix was made on dev data, before any test run.

| # | Failure | Effect | Fix |
|---|---|---|---|
| 1 | Empty `{}` under plain JSON mode | 10/10 invalid in `direct` | Per-case JSON Schema with constrained decoding |
| 2 | Rationale looped to the token limit | Same case timed out in every multi-call arm | Rationale ≤1,000 characters; recorded timeout |
| 3 | `confidence: null` with an answer | Answers rejected | Confidence is a required number, ignored on abstention |
| 4 | Repeated candidate (`["Acne", "Acne"]`) | 3 case-runs failed | Drop repeats, keep order, log count |
| 5 | Borda score tie (3+3 = 2+2+2) | 15 of 26 abstentions, mostly persona arms | Break ties by first-place votes |

Fixes 1–3 came from a first attempt that was archived; the pilot was restarted
from scratch. Fixes 4 and 5 came from the pilot reported here.

- **Persona arms:** individual personas rarely abstained (specialist 4.0%,
  generalist 2.5%, skeptic 4.5% of calls). Before fix 5, persona arms abstained
  because their more diverse answers tied more often.
- **Re-scoring:** aggregates never enter a prompt, so fix 5 was applied to the
  pilot exactly, without new calls (`scripts/reaggregate_runs.py`). Thirteen finals
  changed from abstention to an answer; none was correct, so top-1 accuracy did not
  change.
- **Unexplained failure:** one Gemma call wrote a curly quote inside the rationale,
  then unrelated text, and the server ended the response without a finish reason.
  The call is shared by `role_debate` and `dermarbiter` through seed pairing, so it
  does not bias their comparison.
- **Failure rate:** 5 of 400 case-runs (1.25%), counted as incorrect.

**Results.** Top-1 accuracy / answer coverage (%), 10 dev cases. SCIN is re-scored
with the current tie rule.

| Arm | SCIN Gemma | SCIN Qwen | MedMCQA Gemma | MedMCQA Qwen |
|---|---|---|---|---|
| direct | 20 / 100 | 10 / 100 | 60 / 100 | 70 / 90 |
| structured | 20 / 100 | 10 / 100 | 60 / 100 | 60 / 90 |
| self_refine | 40 / 100 | 0 / 80 | 60 / 100 | 60 / 90 |
| vote3 | 20 / 100 | 10 / 100 | 50 / 100 | 60 / 90 |
| vote6 | 20 / 100 | 0 / 90 | 50 / 90 | 60 / 90 |
| isolated_refine | 20 / 100 | 10 / 100 | 50 / 100 | 60 / 80 |
| debate | 20 / 100 | 10 / 100 | 60 / 100 | 70 / 90 |
| role_vote | 10 / 100 | 10 / 90 | 60 / 90 | 60 / 90 |
| role_debate | 10 / 90 | 10 / 100 | 50 / 100 | 70 / 100 |
| dermarbiter | 10 / 90 | 10 / 100 | 70 / 100 | 60 / 100 |

- **Primary comparisons:** none of the 20 is significant after Holm correction, as
  expected with n = 10.
- **Cost:** `dermarbiter` uses 7× the calls of `direct`, 13–15× the tokens and 7–12×
  the wall time.
- **SCIN baseline:** the constant answer "Eczema" scores 20% on these 10 cases, so
  the small backbones sit at the baseline.
- **Label collapse:** of 200 SCIN finals, 75 were "Acute and chronic dermatitis"
  and 58 "Acute dermatitis, NOS". These umbrella labels are the top reference in
  only one pilot case. We will report the share of the most frequent prediction per
  arm. [TODO: decide with the clinical co-author whether umbrella labels stay in
  the candidate list; the decision must be made before the test split is opened.]

## Reproducibility

Code, prompts, builders, the Kaggle notebook and all run manifests (code and schema
hashes, returned model versions, per-call traces) are in the repository.
