.PHONY: install test clean ex2 ex2-figures

install:
	uv sync

test:
	uv run pytest tests/ -v

ex2:
	uv run python -m perceptrons.ex2 run

ex2-figures:
	uv run python -m perceptrons.ex2 figures

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
