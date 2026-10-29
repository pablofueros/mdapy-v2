.PHONY: install format lint test

install:
	uv sync --locked --extra numba
	uvx prek install

format:
	uv run ruff check --fix
	uv run ruff format

lint:
	uv run ruff check
	uv run ty check

test:
	uv run pytest
