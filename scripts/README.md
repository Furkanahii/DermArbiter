# DermArbiter — E2E Test Scripts

This directory contains end-to-end test and validation scripts for the
DermArbiter multi-agent diagnostic pipeline.

## Scripts Overview

| Script | Purpose | GPU Required |
|--------|---------|:------------:|
| `run_e2e_gpu.py` | Full pipeline runner (mock + real modes) | Optional |
| `run_e2e_colab.py` | Colab-optimized runner with auto-setup | Optional |
| `validate_tools.py` | Tool import/instantiation smoke tests | No |

---

## Quick Start

### 1. Validate Tool Installation

```bash
# Check all 9 tools can be imported
python scripts/validate_tools.py

# Run smoke tests (attempts to execute each tool)
python scripts/validate_tools.py --smoke-test

# Verbose mode with error tracebacks
python scripts/validate_tools.py --verbose

# Export results as JSON
python scripts/validate_tools.py --json outputs/tool_report.json
```

### 2. Run E2E Pipeline (Mock Mode — No GPU / API Keys)

```bash
python scripts/run_e2e_gpu.py \
    --mock \
    --query "Changing mole on back"
```

This uses deterministic mock agents and tools to verify the full 5-phase
pipeline works end-to-end without any external dependencies.

### 3. Run E2E Pipeline (Real Mode — GPU + API Keys)

```bash
# Set API keys
export GOOGLE_API_KEY="your-key-here"

# Full run
python scripts/run_e2e_gpu.py \
    --config configs/ \
    --image data/sample.jpg \
    --query "Changing mole on back"

# With patient context
python scripts/run_e2e_gpu.py \
    --config configs/ \
    --image data/sample.jpg \
    --query "Red scaly patch on elbow" \
    --age 45 --sex male --fitzpatrick 3

# Save results as JSON
python scripts/run_e2e_gpu.py \
    --config configs/ \
    --image data/sample.jpg \
    --query "Changing mole on back" \
    --output-json outputs/results.json
```

### 4. Run in Google Colab

Upload `run_e2e_colab.py` to Colab or clone the repo, then:

```python
# In a Colab cell:
%cd /content
!git clone <your-repo-url> DermArbiter
%cd DermArbiter
%run scripts/run_e2e_colab.py --query "Changing mole on back" --mock
```

Or for real mode with GPU:

```python
%run scripts/run_e2e_colab.py \
    --query "Red scaly patch on elbow" \
    --image data/sample.jpg
```

The Colab script automatically:
- Installs all dependencies
- Detects GPU (CUDA/MPS)
- Downloads a sample image if none provided
- Saves results to Google Drive

---

## Pipeline Phases

The scripts exercise all 5 DermArbiter pipeline phases:

```
Phase 1: Plan & Probe
  → Agents propose tools → ToolRegistry executes them
  → Evidence cards written to BlackboardState

Phase 2: Independent Reading
  → Each agent reviews evidence cards independently
  → Generates AgentBrief with differential diagnosis + confidence

Phase 3: Reveal & Critique
  → Briefs are revealed to all agents
  → Moderator checks early-exit gate (unanimous consensus?)

Phase 4: Targeted Debate
  → Multi-round argument/rebuttal on disagreements
  → Up to max_rounds of debate

Phase 5: Synthesis
  → Moderator generates final clinical report
  → Consensus score + dissent notes computed
```

---

## Mock Mode Details

Mock mode (`--mock`) provides:

- **9 deterministic tools** with realistic dermatology outputs:
  - PanDerm classifier (melanocytic nevus 62%, melanoma 18%)
  - MAKE ABCDE annotator (asymmetry, border, color, diameter, evolution)
  - DermoGPT VQA, General VQA, GuidelineRAG, CaseRAG
  - OntologyGraph (ICD-10, SNOMED), FairnessProbe, UncertaintyProbe

- **4 mock agents** with role-appropriate behaviors:
  - Specialist (high confidence, tool-driven)
  - Generalist (moderate confidence, broader differential)
  - Skeptic (conservative, raises follow-up concerns)
  - Moderator (consensus-seeking, report synthesis)

- **No external dependencies**: No GPU, no API keys, no model downloads

---

## CLI Reference

### `run_e2e_gpu.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/` | Path to config directory |
| `--image` | None | Clinical image path |
| `--query` | (required) | Clinical query text |
| `--mock` | False | Use mock agents/tools |
| `--log-level` | `INFO` | Logging verbosity |
| `--output-json` | None | Save results to JSON |
| `--age` | None | Patient age |
| `--sex` | None | Patient sex |
| `--fitzpatrick` | None | Fitzpatrick skin type (1-6) |
| `--location` | None | Lesion body location |
| `--duration` | None | Condition duration |

### `run_e2e_colab.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | (required) | Clinical query text |
| `--image` | None | Local image path |
| `--image-url` | None | URL to download sample image |
| `--config` | `configs/` | Config directory |
| `--mock` | False | Use mock mode |
| `--no-drive` | False | Skip Google Drive save |
| `--log-level` | `INFO` | Logging verbosity |

### `validate_tools.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--smoke-test` | False | Run smoke tests on tools |
| `--verbose` | False | Show error tracebacks |
| `--json` | None | Save results to JSON |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: dermarbiter` | Run `pip install -e .` from project root |
| `No GPU detected` | Use `--mock` flag or ensure CUDA is installed |
| `API key missing` | Set `GOOGLE_API_KEY` env variable or in `.env` |
| `Tool import fails` | Run `validate_tools.py --verbose` for details |
| `ToolRegistry error` | Check `dermarbiter/tools/base_tool.py` exists |
| `Colab: module not found` | Restart runtime after install (`Runtime > Restart`) |
