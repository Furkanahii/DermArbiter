.PHONY: test lint format demo clean install benchmark-mock analyze evaluate fairness validate-tools e2e-mock pipeline-demo

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check dermarbiter/ tests/

format:
	poetry run ruff format dermarbiter/ tests/

demo:
	python3 notebooks/02_agent_demo.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist/ build/ htmlcov/ .coverage

# ──────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────

benchmark-mock:
	python -m dermarbiter.experiments.runner --config configs/default.yaml --data data/sample_cases.jsonl --output results/mock_run.jsonl --mock

analyze:
	python -m dermarbiter.experiments.analyze --results results/mock_run.jsonl

evaluate:
	python -m dermarbiter.evaluation.metrics --results results/mock_run.jsonl

fairness:
	python -m dermarbiter.evaluation.fairness_analyzer --results results/mock_run.jsonl

# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

validate-tools:
	python scripts/validate_tools.py

e2e-mock:
	python scripts/run_e2e_gpu.py --mock --query "Evaluate this skin lesion"

# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

pipeline-demo:
	python notebooks/03_full_pipeline_demo.py

