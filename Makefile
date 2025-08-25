PY = python

.PHONY: dev install test lint type fmt api

install:
	$(PY) -m pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff check . --fix
	ruff format .

type:
	mypy .

api:
	uvicorn trade_agent.backend.main:app --reload

dev: install
	@echo "Dev environment ready"
