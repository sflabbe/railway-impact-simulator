UV ?= uv
PYTEST_ARGS ?= tests -q

.PHONY: sync lock lock-check test lint format typecheck smoke clean

sync:
	$(UV) sync --all-extras --dev

lock:
	$(UV) lock

lock-check:
	$(UV) lock --check

test:
	$(UV) run pytest $(PYTEST_ARGS)

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	@echo "manual review required: mypy is not configured for this repository; no typecheck is run by this target."

smoke:
	$(UV) run railway-sim --help

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info htmlcov .coverage
