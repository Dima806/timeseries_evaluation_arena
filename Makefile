.PHONY: help setup sync lint format check typecheck test test-cov \
        notebooks run lab clean reset ci dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup: install uv + all deps
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	uv run python -m ipykernel install --sys-prefix
	@echo "\n✅ Ready. Run 'make test' to verify."

sync: ## Sync deps from lockfile
	uv sync --all-extras

lint: format check typecheck ## Run all linters

format: ## Auto-format with ruff
	uv run ruff format src/ tests/ app/

check: ## Lint and auto-fix with ruff
	uv run ruff check --fix src/ tests/ app/

typecheck: ## Type-check with ty
	uv run ty check src/

test: ## Run pytest (coverage summary printed at end)
	uv run pytest

test-cov: ## Run pytest with full HTML + terminal coverage report
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html
	@echo "\n📊 HTML report: htmlcov/index.html"

notebooks: ## Execute all notebooks
	@uv run python scripts/gen_notebooks.py
	@for nb in notebooks/0[0-9]_*.ipynb; do \
		echo "▶ Executing $$nb ..."; \
		uv run jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=300 "$$nb" || exit 1; \
	done
	@echo "\n✅ All notebooks executed."

run: ## Launch Streamlit app
	uv run streamlit run app/streamlit_app.py --server.port 8501

lab: ## Launch JupyterLab
	uv run jupyter lab --no-browser --port 8888

clean: ## Remove caches and outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache htmlcov .coverage
	@echo "🧹 Cleaned."

reset: clean ## Full reset
	rm -rf .venv
	@echo "🔄 Reset. Run 'make setup' to rebuild."

ci: sync lint test ## CI pipeline
dev: lint test ## Fast offline dev loop
