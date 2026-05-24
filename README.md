# DermArbiter

> **Training-free, multi-LLM debate framework for dermatological diagnosis**

DermArbiter orchestrates multiple large language models in a structured debate protocol to produce accurate, explainable, and fair dermatological diagnoses — without any additional model training or fine-tuning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DermArbiter                              │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Specialist   │  │  Generalist  │  │   Skeptic    │          │
│  │  (Gemini 2.5  │  │  (MedGemma   │  │  (Qwen3-8B   │          │
│  │   Flash)      │  │   4B)        │  │   Instruct)  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Shared Blackboard (State)               │       │
│  │   Arguments · Votes · Evidence · Confidence Scores   │       │
│  └──────────────────────┬──────────────────────────────┘       │
│                         │                                       │
│                         ▼                                       │
│              ┌──────────────────┐                               │
│              │    Moderator     │                               │
│              │  (Gemini 2.5     │                               │
│              │   Flash)         │                               │
│              └────────┬─────────┘                               │
│                       │                                         │
│         ┌─────────────┼─────────────┐                           │
│         ▼             ▼             ▼                           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                     │
│  │ Tool Pool │ │  Case RAG │ │ Fairness  │                     │
│  │ (PanDerm, │ │ (ChromaDB)│ │  & Bias   │                     │
│  │  MAKE,    │ │           │ │  Probes   │                     │
│  │  DermGPT) │ │           │ │           │                     │
│  └───────────┘ └───────────┘ └───────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent        | Model                  | Backend    | Role                                           |
|-------------|------------------------|------------|-------------------------------------------------|
| Specialist  | Gemini 2.5 Flash       | Google API | Domain-expert reasoning with dermatology focus  |
| Generalist  | MedGemma 4B            | Local HF   | Broad medical knowledge and visual grounding    |
| Skeptic     | Qwen3-8B-Instruct      | Local HF   | Adversarial challenge and counter-arguments     |
| Moderator   | Gemini 2.5 Flash       | Google API | Synthesizes debate, resolves consensus          |

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [Poetry](https://python-poetry.org/docs/#installation)
- CUDA-capable GPU (for local HF models)
- API keys (see `.env.example`)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/DermArbiter.git
cd DermArbiter

# Copy environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# Install dependencies
make install
# or: poetry install
```

### Run the Demo

```bash
make demo
# or: poetry run python -m dermarbiter.demo
```

### Run Tests

```bash
make test
```

### Lint & Format

```bash
make lint
make format
```

---

## Project Structure

```
DermArbiter/
├── configs/
│   ├── agents.yaml          # Agent model configurations
│   ├── benchmarks.yaml      # Benchmark dataset definitions
│   ├── default.yaml         # General settings
│   └── tools.yaml           # Tool pool configurations
├── dermarbiter/
│   ├── __init__.py           # Package root
│   ├── core/                 # Core framework: state, graph, blackboard
│   │   └── __init__.py
│   ├── agents/               # Agent implementations (specialist, generalist, etc.)
│   │   └── __init__.py
│   ├── tools/                # Tool wrappers (PanDerm, MAKE, RAG, etc.)
│   │   └── __init__.py
│   └── evaluation/           # Metrics, benchmarks, fairness analysis
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── mocks/
│       └── __init__.py
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Configuration

All configuration is managed via YAML files in `configs/`:

- **`default.yaml`** — Global settings (debug mode, logging, output directory)
- **`agents.yaml`** — Per-agent model configs and debate protocol parameters
- **`tools.yaml`** — Enable/disable and configure each tool in the tool pool
- **`benchmarks.yaml`** — Dataset paths, splits, and evaluation metrics

---

## Citation

If you use DermArbiter in your research, please cite:

```bibtex
@article{dermarbiter2025,
  title   = {DermArbiter: Training-Free Multi-LLM Debate Framework for Dermatological Diagnosis},
  author  = {Ahi, Furkan and Emre, Mahmut},
  year    = {2025},
  note    = {Manuscript in preparation}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
