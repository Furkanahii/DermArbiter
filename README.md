<![CDATA[# DermArbiter

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-purple.svg)](https://github.com/langchain-ai/langgraph)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)]()

> **Training-free, multi-LLM debate framework for dermatological diagnosis**

DermArbiter orchestrates four heterogeneous large language models in a structured five-phase debate protocol to produce accurate, explainable, and fair dermatological diagnoses — without any additional model training or fine-tuning. By combining frozen foundation-model tools with a novel debate architecture, DermArbiter achieves competitive diagnostic performance while providing built-in fairness guarantees across Fitzpatrick skin types I–VI. The framework is designed to be config-driven, fully reproducible, and extensible to new agents, tools, and benchmarks.

---

## Architecture

```
                           ┌──────────────────────────────────────────┐
                           │          DermArbiter Pipeline            │
                           └──────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                                                                             │
  │   Phase 1: PLAN & PROBE                                                     │
  │   ┌──────────┐  ┌──────────┐  ┌──────────┐                                 │
  │   │Specialist│  │Generalist│  │  Skeptic  │  → Propose tools                │
  │   └────┬─────┘  └────┬─────┘  └────┬─────┘                                 │
  │        └──────────────┼──────────────┘                                      │
  │                       ▼                                                     │
  │           ┌───────────────────────┐    9 frozen tools:                      │
  │           │    Tool Registry      │    PanDerm · MAKE · DermoGPT            │
  │           │    (batch execute)    │    MedGemma · GuidelineRAG · CaseRAG    │
  │           └───────────┬───────────┘    OntologyGraph · FairnessProbe        │
  │                       │                UncertaintyProbe                     │
  │                       ▼                                                     │
  │   Phase 2: INDEPENDENT READING                                              │
  │   ┌──────────┐  ┌──────────┐  ┌──────────┐                                 │
  │   │Specialist│  │Generalist│  │  Skeptic  │  → Independent briefs           │
  │   │  Brief   │  │  Brief   │  │  Brief    │    (differential + confidence)  │
  │   └────┬─────┘  └────┬─────┘  └────┬─────┘                                 │
  │        └──────────────┼──────────────┘                                      │
  │                       ▼                                                     │
  │   Phase 3: REVEAL & CRITIQUE                                                │
  │   ┌─────────────────────────────────┐                                       │
  │   │  Moderator: early-exit gate     │  ← Unanimous consensus? → Phase 5    │
  │   └────────────────┬────────────────┘                                       │
  │                    │ No consensus                                           │
  │                    ▼                                                        │
  │   Phase 4: TARGETED DEBATE                                                  │
  │   ┌──────────────────────────────┐                                          │
  │   │  Multi-round argument cycle  │  Specialist ↔ Generalist ↔ Skeptic      │
  │   │  (up to max_rounds rounds)   │  Token budget · Turn order enforced      │
  │   └────────────────┬─────────────┘                                          │
  │                    ▼                                                        │
  │   Phase 5: SYNTHESIS                                                        │
  │   ┌─────────────────────────────────┐                                       │
  │   │  Moderator: final clinical      │  → Top-K diagnoses                    │
  │   │  report + consensus score       │  → Confidence + dissent notes         │
  │   │  + fairness attestation         │  → Fairness attestation               │
  │   └─────────────────────────────────┘                                       │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Training-free multi-LLM debate** — No fine-tuning; leverages frozen foundation models through structured argumentation
- **9 frozen diagnostic tools** including 2 novel probes:
  - **Fairness Probe** — Fitzpatrick skin tone bias detection and mitigation
  - **Uncertainty Probe** — Calibrated prediction uncertainty quantification
- **Structured 5-phase debate protocol** with early-exit gating for efficiency
- **Config-driven architecture** — All agents, tools, and benchmarks configurable via YAML
- **Comprehensive fairness evaluation** — Fitzpatrick-stratified analysis, equalized odds, demographic parity
- **LangGraph state machine** — Deterministic orchestration with conditional routing
- **Reproducible benchmarking** — Experiment runner, ablation studies, and results analysis
- **Mock mode** — Full pipeline testing without GPU or API keys

---

## Agent–Model Mapping

| Agent | Model | Backend | Role |
|:------|:------|:--------|:-----|
| **Specialist** | Gemini 2.5 Flash | Google API | Domain-expert reasoning with dermatology focus |
| **Generalist** | MedGemma 4B | Local HF (GPU) | Broad medical knowledge and visual grounding |
| **Skeptic** | Qwen3-8B-Instruct | Local HF (GPU) | Adversarial challenge and counter-arguments |
| **Moderator** | Gemini 2.5 Flash | Google API | Debate synthesis, consensus resolution, report generation |

---

## Tool Pool

| # | Tool | Module | Source | Description |
|:-:|:-----|:-------|:-------|:------------|
| 1 | **PanDerm Classifier** | `panderm_tool.py` | [PanDerm](https://github.com/SiyuanYan1/PanDerm) | Universal dermatology foundation model; multi-class lesion classification |
| 2 | **MAKE Annotator** | `make_tool.py` | [MAKE](https://github.com/CristianoPatrwormo/MAKE) | Multi-attribute knowledge extraction (ABCDE criteria) |
| 3 | **DermoGPT VQA** | `dermogpt_tool.py` | [DermoGPT](https://github.com/Frankunv/DermoGPT) | Dermatology-specialized visual question answering |
| 4 | **MedGemma VQA** | `medgemma_tool.py` | [MedGemma](https://ai.google.dev/gemma/docs/medgemma) | General medical VQA — second opinion channel |
| 5 | **Guideline RAG** | `guideline_rag.py` | Internal | Retrieval-augmented generation over clinical guidelines |
| 6 | **Case RAG** | `case_rag.py` | Internal (ChromaDB) | Similar-case retrieval from dermatology case databases |
| 7 | **Ontology Graph** | `ontology_graph.py` | Internal | ICD-10, SNOMED-CT, DermLex code mapping & hierarchy traversal |
| 8 | **Fairness Probe** ★ | `fairness_probe.py` | **Novel** | Fitzpatrick skin tone bias detection and demographic parity checks |
| 9 | **Uncertainty Probe** ★ | `uncertainty_probe.py` | **Novel** | Calibrated prediction uncertainty quantification |

★ = Novel contributions introduced in this work.

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (for local HF models; optional with `--mock`)
- API keys for Google Gemini (see `.env.example`)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/DermArbiter.git
cd DermArbiter

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (GOOGLE_API_KEY, etc.)

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
# or using Poetry:
make install
```

### Run the Pipeline (Mock Mode — No GPU Required)

```bash
python scripts/run_e2e_gpu.py --mock --query "Changing mole on back"
```

### Run Tests

```bash
make test
# or: pytest
```

---

## Usage Examples

### Single Case Diagnosis

```bash
# Mock mode (no GPU / API keys)
python scripts/run_e2e_gpu.py \
    --mock \
    --query "Changing mole on back"

# Real mode (GPU + API keys)
python scripts/run_e2e_gpu.py \
    --config configs/ \
    --image data/sample.jpg \
    --query "Red scaly patch on elbow" \
    --age 45 --sex male --fitzpatrick 3
```

### Benchmark Evaluation

```bash
# Run on sample test cases
make benchmark-mock

# Analyze results
make analyze
```

### Metrics & Evaluation

```bash
# Classification metrics (accuracy, F1, calibration)
make evaluate

# Fairness analysis (equalized odds, demographic parity)
make fairness
```

### Tool Validation

```bash
# Verify all 9 tools import correctly
make validate-tools

# With smoke tests
python scripts/validate_tools.py --smoke-test --verbose
```

---

## Project Structure

```
DermArbiter/
├── configs/
│   ├── agents.yaml              # Agent model configurations & debate parameters
│   ├── benchmarks.yaml          # Benchmark dataset definitions & splits
│   ├── default.yaml             # Global settings (logging, output, debug)
│   └── tools.yaml               # Tool pool configurations & enable/disable
├── dermarbiter/
│   ├── __init__.py              # Package root
│   ├── core/                    # Core framework
│   │   ├── orchestrator.py      #   LangGraph state machine (5-phase workflow)
│   │   ├── blackboard.py        #   Shared state: arguments, votes, evidence
│   │   └── debate_protocol.py   #   Phase implementations
│   ├── agents/                  # Agent implementations
│   │   ├── base_agent.py        #   Abstract base agent
│   │   ├── specialist.py        #   Gemini 2.5 Flash — domain expert
│   │   ├── generalist.py        #   MedGemma 4B — broad medical
│   │   ├── skeptic.py           #   Qwen3-8B — adversarial
│   │   └── moderator.py         #   Gemini 2.5 Flash — synthesis
│   ├── tools/                   # Tool wrappers (9 tools)
│   │   ├── base_tool.py         #   BaseTool, ToolOutput, ToolRegistry
│   │   ├── panderm_tool.py      #   PanDerm foundation model classifier
│   │   ├── make_tool.py         #   MAKE ABCDE annotator
│   │   ├── dermogpt_tool.py     #   DermoGPT visual QA
│   │   ├── medgemma_tool.py     #   MedGemma general VQA
│   │   ├── guideline_rag.py     #   Clinical guideline retrieval
│   │   ├── case_rag.py          #   Similar case retrieval (ChromaDB)
│   │   ├── ontology_graph.py    #   ICD-10/SNOMED-CT mapping
│   │   ├── fairness_probe.py    #   Fitzpatrick fairness probe ★
│   │   └── uncertainty_probe.py #   Uncertainty quantification ★
│   ├── evaluation/              # Evaluation & fairness
│   │   ├── benchmark_runner.py  #   Benchmark harness
│   │   ├── metrics.py           #   Classification & captioning metrics
│   │   └── fairness_analyzer.py #   Equalized odds, demographic parity
│   └── experiments/             # Experiment orchestration
│       ├── runner.py            #   ExperimentRunner & BenchmarkRunner
│       ├── analyze.py           #   ResultsAnalyzer
│       └── ablation.py          #   Ablation studies
├── scripts/
│   ├── run_e2e_gpu.py           # Full pipeline runner (mock + real)
│   ├── run_e2e_colab.py         # Google Colab runner with auto-setup
│   ├── validate_tools.py        # Tool import & smoke tests
│   ├── validate_tools_colab.py  # Colab-specific tool validation
│   └── setup_colab.py           # Colab environment setup
├── notebooks/
│   ├── 01_panderm_test.py       # PanDerm standalone test
│   └── colab_validation.py      # Colab validation notebook
├── tests/
│   ├── conftest.py              # Shared fixtures & mock factories
│   ├── mocks/                   # Mock agents & tools
│   ├── test_agents.py           # Agent unit tests
│   ├── test_blackboard.py       # Blackboard state tests
│   ├── test_debate.py           # Debate protocol tests
│   ├── test_tools.py            # Tool wrapper tests
│   ├── test_week2_tools.py      # Week 2 tool tests (MAKE, DermoGPT, etc.)
│   ├── test_week3_tools.py      # Week 3 tool tests (Fairness, Uncertainty)
│   ├── test_evaluation.py       # Evaluation module tests
│   ├── test_experiments.py      # Experiment runner tests
│   ├── test_integration.py      # End-to-end integration tests
│   └── test_panderm.py          # PanDerm-specific tests
├── data/
│   └── sample_cases.jsonl       # Sample test cases for benchmarking
├── .env.example                 # Template for API keys
├── .gitignore
├── Makefile                     # Build, test, benchmark, and evaluation targets
├── pyproject.toml               # Project metadata & dependencies
└── README.md
```

---

## Configuration

All configuration is managed via YAML files in `configs/`:

| File | Purpose |
|:-----|:--------|
| `default.yaml` | Global settings — debug mode, logging level, output directory, token budgets |
| `agents.yaml` | Per-agent model configs, debate protocol parameters, turn order, specialist weight |
| `tools.yaml` | Enable/disable individual tools, model paths, inference parameters |
| `benchmarks.yaml` | Dataset paths, train/val/test splits, evaluation metrics, Fitzpatrick stratification |

---

## Testing

```bash
# Run full test suite
make test

# Run with pytest directly (verbose, short traceback)
pytest

# Run specific test modules
pytest tests/test_tools.py
pytest tests/test_debate.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=dermarbiter --cov-report=html
```

### Lint & Format

```bash
make lint      # ruff check
make format    # ruff format
```

---

## Benchmarks

DermArbiter is evaluated on the following dermatology benchmark datasets:

| Dataset | Task | Size | Fairness | Source |
|:--------|:-----|:-----|:---------|:-------|
| **HAM10000** | 7-class lesion classification | 10,015 images | — | [Tschandl et al. 2018](https://doi.org/10.1038/sdata.2018.161) |
| **Derm7pt** | Multi-attribute diagnosis | 1,011 cases | — | [Kawahara et al. 2019](https://doi.org/10.1109/JBHI.2018.2824327) |
| **SkinCon** | Concept-based diagnosis | 3,230 images | — | [Daneshjou et al. 2022](https://skincon-dataset.github.io/) |
| **Fitzpatrick17k** | Skin condition classification | 16,577 images | Fitzpatrick I–VI | [Groh et al. 2021](https://doi.org/10.1038/s41591-021-01595-0) |

---

## Citation

If you use DermArbiter in your research, please cite:

```bibtex
@article{dermarbiter2025,
  title     = {DermArbiter: Training-Free Multi-LLM Debate for Equitable
               Dermatological Diagnosis},
  author    = {Ahi, Furkan and Ayd{\i}n, Mahmut Emre},
  journal   = {Nature Medicine},
  year      = {2025},
  note      = {Manuscript in preparation}
}
```

---

## Authors

- **Furkan Ahi** — Lead developer & system architect
- **Mahmut Emre Aydın** — Co-developer & evaluation lead

---

## License

This project is licensed under the [MIT License](LICENSE).
]]>
