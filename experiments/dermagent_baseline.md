# DermAgent Baseline — Reference Numbers and Comparison Plan

**Source**: Liu et al., *DermAgent: A Self-Reflective Agentic System for
Dermatological Image Analysis with Multi-Tool Reasoning and Traceable
Decision-Making*. arXiv:2605.14403, MICCAI 2026.
Code: <https://github.com/YizeezLiu/DermAgent>

---

## 1. Why we are not running DermAgent end-to-end

| Requirement | DermAgent | Available locally / on Colab T4 |
|---|---|---|
| GPU VRAM | ~20–22 GB simultaneous (DermoGPT-RL 16 GB + Qwen3-VL 16 GB) | Colab T4 = 16 GB |
| API | `OPENAI_API_KEY` (GPT-4o reasoning backbone) | Paid; cost per case is non-trivial |
| Manual weights | DermoGPT-RL, Qwen3-Embedding-8B, Qwen3-Reranker-0.6B | Several GB total |
| Test set | 642-image class-balanced HAM10000 subset (`HAM10000_benchmark_500.csv`) | Pulled verbatim — see §2 |

Decision: **we use the published numbers from Liu et al. as the comparator,
and evaluate DermArbiter on the *exact same* 642-image subset** so the
accuracy delta is apples-to-apples. We do *not* attempt to re-run their
pipeline locally.

This is consistent with how prior agent papers (MDAgents, MedAgent-Pro,
MedRAX) are typically compared in the literature when their full reproduction
exceeds available compute.

---

## 2. Test split we evaluate against

| Property | Value |
|---|---|
| File (theirs) | `data/ham10000/dermagent_subset.csv` (pulled from upstream repo) |
| File (ours) | `data/ham10000/dermagent_subset.jsonl` (produced by `scripts/build_dermagent_subset.py`) |
| Image count | 642 |
| Classes | 7 (akiec, bcc, bkl, df, mel, nv, vasc) |
| Balance | Class-balanced rare classes (50 each of akiec/bcc/df/vasc), realistic head (331 nv, 58 mel, 53 bkl) |

Class distribution (matches DermAgent paper):

```
nv     331
mel     58
bkl     53
df      50
vasc    50
bcc     50
akiec   50
```

To rebuild after re-downloading HAM10000:

```bash
python scripts/download_datasets.py --dataset ham10000               # 3 GB; full
python scripts/build_dermagent_subset.py --require-image             # filtered to 642
```

---

## 3. Published numbers (Liu et al. Table 1)

**Performance comparison on dermatological diagnosis (Accuracy %), concept
annotation (F1-Macro %), and captioning (ROUGE-L %).** *Higher is better
across all columns.*

| Model | Type | HAM10000 | SNU | Derm7pt | SkinCon | SkinCAP |
|---|---|---:|---:|---:|---:|---:|
| LLaVA-Med-v1.5 | Medical MLLM | 44.24 | 1.20 | 51.70 | 13.10 | 15.32 |
| HuatuoGPT | Medical MLLM | 51.40 | 4.00 | 53.43 | 9.49 | 14.32 |
| Hulu-Med-7B | Medical MLLM | 53.00 | 0.80 | 51.63 | 9.63 | 11.43 |
| DermoGPT-RL | Dermatology MLLM | 50.00 | 9.20 | 56.86 | 30.72 | 15.41 |
| SkinGPT-PathMM | Dermatology MLLM | 45.17 | 3.40 | 53.14 | 13.30 | 14.44 |
| Qwen3-VL-8B | General MLLM | 51.09 | 7.80 | 53.70 | 22.82 | 12.47 |
| GPT-4o | General MLLM | 48.91 | 15.00 | 54.14 | 20.56 | 16.33 |
| GPT-5.2 | General MLLM | 55.98 | 14.80 | 53.86 | 26.62 | 12.35 |
| MDAgents | Medical Agent | 16.82 | 11.40 | 36.14 | 23.93 | 11.99 |
| MedAgent-Pro | Medical Agent | 57.63 | 11.60 | 64.62 | 18.34 | 13.48 |
| **DermAgent (Ours)** | **Medical Agent** | **81.83** | **32.60** | **65.06** | **32.95** | **19.46** |

Notes:

* DermAgent's 81.83 % HAM10000 number is on the 642-image class-balanced split
  (§2), not on full HAM10000. **Any DermArbiter HAM10000 number quoted in the
  same column must use that subset.**
* DermAgent uses GPT-4o as the reasoning backbone. DermArbiter uses Gemini 2.5
  Flash + Qwen3-8B + MedGemma-4B and is API-cost-free at the small-volume tier.
* The Critic module (post-hoc audit + self-correction) is unique to DermAgent;
  DermArbiter's analogous mechanism is the structured debate phase and the
  early-exit gate.

---

## 4. Published ablation (Liu et al. Table 2)

**Ablation on SkinCAP captioning ROUGE-L (%).** Δ is measured against "Full
Agent (w/o Critic) = 17.25".

| Configuration | ROUGE-L | Δ |
|---|---:|---:|
| Full Agent (w/ Critic) | 19.46 | +2.21 |
| Full Agent (w/o Critic) | 17.25 | — |
| w/o Case RAG | 15.80 | −1.47 |
| w/o Guideline RAG | 16.20 | −0.99 |
| w/o DermoGPT | 16.72 | −0.55 |
| w/o PanDerm | 16.76 | −0.51 |
| w/o MAKE | 16.79 | −0.48 |
| w/o Ontology | 17.12 | −0.15 |

Take-aways for DermArbiter design:

1. **Case RAG is the single largest contributor** in DermAgent's ablation
   (−1.47). Our DermArbiter tool pool already includes Case RAG (DermLIP +
   ChromaDB over Derm1M); the build script for that index is
   `scripts/build_case_rag_index.py`.
2. The Critic adds +2.21 ROUGE-L by itself. DermArbiter's analogous gain
   comes from the *Reveal & Critique* phase (#3) and *Targeted Debate*
   (#4) — these should be ablated in our own Table 2.
3. Ontology contributes the least (−0.15). We should still keep it for
   provenance / structured reasoning but de-prioritize hand-curating large
   taxonomies in Phase 1.

---

## 5. DermArbiter comparison table — to be filled

When DermArbiter's real-mode pipeline is wired up (waiting on
`model_router._call_local`), run the evaluator described in §6 and fill
the **DermArbiter** row(s).

| Model | HAM10000 (642) | ECE ↓ | Brier ↓ | Δacc Fitz V+VI vs I+II | Avg tool calls / case | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| DermAgent (paper) | **81.83** | — | — | — | — | — |
| GPT-5.2 (paper) | 55.98 | — | — | — | — | — |
| MedAgent-Pro (paper) | 57.63 | — | — | — | — | — |
| DermArbiter (full, w/ debate) | — | — | — | — | — | — |
| DermArbiter (no debate) | — | — | — | — | — | — |
| DermArbiter (no Case RAG) | — | — | — | — | — | — |
| DermArbiter (no Skeptic) | — | — | — | — | — | — |

ECE/Brier/Δacc are **DermArbiter-specific contributions** that DermAgent does
not report — those are the columns where we differentiate the paper narrative
on calibration + fairness even if raw accuracy comes out behind.

---

## 6. How to evaluate DermArbiter on this subset

```bash
# Mock pipeline (today — exercises the runner end-to-end without LLM costs):
python scripts/run_dermagent_subset.py --mock

# Real pipeline (after Furkan's model_router local backend lands):
python scripts/run_dermagent_subset.py \
    --config configs/agents.yaml \
    --tools configs/tools.yaml
```

Output goes to `experiments/results/dermagent_subset_<timestamp>.jsonl` and
a summary metrics block is written to
`experiments/results/dermagent_subset_<timestamp>.metrics.json`. Pipe both
through `dermarbiter.evaluation.metrics.MetricsCalculator` to populate §5.

---

## 7. Provenance

* `data/ham10000/dermagent_subset.csv` — verbatim copy of
  `HAM10000_benchmark_500.csv` from commit `master` of `YizeezLiu/DermAgent`,
  pulled on 2026-05-24. SHA-256 of original: see `git log` once committed.
* All numbers in §3 and §4 transcribed from arXiv:2605.14403v1 PDF, pages 7.
  Cross-check: paper claims "exceeding GPT-4o by 17.6 % in skin disease
  diagnostic accuracy" — DermAgent 81.83 vs GPT-4o on **SNU** is 32.60 − 15.00
  = 17.60. The 17.6 % gap is on SNU, not on HAM10000.
