.PHONY: install lint format test smoke train serve dashboard clean

install:
	uv venv .venv
	uv pip install -e ".[dev,test]"

lint:
	uv run ruff check src/ tests/
	uv run black --check src/ tests/

format:
	uv run ruff check --fix src/ tests/
	uv run black src/ tests/

test:
	uv run pytest tests/ -v

smoke:
	uv run python scripts/smoke_test_foundation.py

train:
	uv run python scripts/run_training.py

serve:
	uv run uvicorn src.bnpl.serving.api:app --reload --host 0.0.0.0 --port 8000

dashboard:
	uv run streamlit run dashboard/app.py

clean:
	rm -rf .venv __pycache__ .pytest_cache .mypy_cache mlruns/ logs/*.log dist/ build/ *.egg-info
