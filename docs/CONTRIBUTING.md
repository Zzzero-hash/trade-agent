# Contributing Guide

## Dev Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

## Tests
```
pytest -q
```

## Style
- Ruff (lint & format) enforced via pre-commit
- mypy strict typing target (progressively enforced)

## Pull Requests
- Add/adjust tests for new behavior
- Keep functions <= ~50 lines where practical
- Document new public APIs in docs/

## Roadmap Phases
See project milestones (M0..M6) in Linear description.
