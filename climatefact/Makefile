.PHONY: lint lint-fix format format-check typecheck test check clean help

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run ruff linter (displays issues)
	@echo "Running ruff check..."
	poetry run ruff check .

lint-fix: ## Run ruff linter and auto-fix issues
	@echo "Running ruff check with auto-fix..."
	poetry run ruff check --fix .

format: ## Format code with ruff
	@echo "Formatting code with ruff..."
	poetry run ruff format .

format-check: ## Check code formatting without modifying files
	@echo "Checking code formatting..."
	poetry run ruff format --check .

typecheck: ## Run type checking with pyright/mypy
	@echo "Running type checking..."
	uvx ty check

test: ## Run pytest tests
	@echo "Running pytest..."
	poetry run pytest

check: lint format-check typecheck test ## Run all checks (lint, format-check, typecheck, test)
	@echo "All checks completed successfully!"

clean: ## Clean up cache and temporary files
	@echo "Cleaning up cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Clean complete!"
