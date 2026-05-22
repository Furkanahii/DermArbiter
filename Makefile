.PHONY: test lint format demo clean install

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check dermarbiter/ tests/

format:
	poetry run ruff format dermarbiter/ tests/

demo:
	poetry run python -m dermarbiter.demo

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist/ build/ htmlcov/ .coverage
