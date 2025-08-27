# Trade Agent - Local-First ML Trading Platform

A trading agent platform for reinforcement learning and algorithmic trading.

## Project Structure

```
trade_agent/                    # Main Python package
├── __init__.py                # Package initialization
├── engine/                    # Core execution engine (ready for implementation)
│   └── __init__.py
└── backend/                   # Backend services (ready for implementation)
    └── __init__.py

docs/                          # Documentation
├── source/
│   └── glossary.md
└── Makefile

scripts/                       # Utility scripts
├── bump_version.py            # Version management
└── verify_release.py          # Release quality gates

tests/                         # Test suite
└── test_bump_version.py       # Version script tests

ops/                          # Operations and CI
└── ci/
    └── workflows/
        └── lint_type_test.sh

# Configuration files
├── pyproject.toml            # Project configuration
├── version.py                # Centralized version
├── LICENSE                   # MIT License
├── Dockerfile               # Container setup
└── README.md                # This file
```

## Getting Started

### Installation

```bash
pip install -e .
```

### Usage

```python
import trade_agent
print(f"Trade Agent version: {trade_agent.__version__}")
```

## Development

This project follows a clean, minimal structure focused on the core trading platform functionality. The `trade_agent/` package is organized for:

- **Engine**: Core execution and pipeline orchestration
- **Backend**: Services and utilities

### Quality Gates

Run the release verification script to check code quality:

```bash
python scripts/verify_release.py --all
```

### Version Management

Bump version using the included script:

```bash
python scripts/bump_version.py --part patch  # or minor, major
```

## License

MIT License - see LICENSE file for details.
