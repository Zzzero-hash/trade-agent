## Contributing Guide

Thank you for considering a contribution. This project targets a clean, testable, and type‑annotated codebase supporting supervised + reinforcement learning for trading.

### Core Principles

1. Small, reviewable PRs (aim < 400 line diff excluding generated docs).
2. Maintain backward compatibility for at least one release when moving APIs.
3. Every new public function/class: type hints + docstring + test.
4. Determinism where practical (fix seeds; isolate randomness).
5. No secrets or credentials in code or tests – use `.env` (git‑ignored).

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

### Pre-Commit Hooks

Hooks run `ruff`, `black`, `mypy` (config in `pyproject.toml`), and basic security/static checks. Ensure they pass locally before pushing.

### Branch / Commit Conventions

- Branch naming: `feat/`, `fix/`, `refactor/`, `docs/`, `chore/` prefixes.
- Conventional commits style recommended (e.g. `feat(sl): add transformer dropout config`).

### Testing Strategy

- Fast unit tests: keep under ~1s each; mark slow ones with `@pytest.mark.slow`.
- Coverage target: >= 80% (see `pyproject.toml`).
- Use property-based tests (Hypothesis) for numeric stability where helpful.

### Adding / Moving Modules (Migration Plan)

Refactor toward new package layout:

```
trade_agent/
  data/        # ingestion, feature loading
  agents/      # sl, rl, envs
  training/    # loops, schedulers
  evaluation/  # metrics, backtests
  api/         # service layer
  utils/       # shared helpers
```

When moving a module:

1. `git mv src/sl/evaluate.py trade_agent/agents/sl/evaluate.py`
2. Add a shim `src/sl/evaluate.py` re-exporting (temporary) if external code still imports old path.
3. Update internal imports to `from trade_agent.agents.sl.evaluate import ...`.
4. Run tests + linters.

Remove shims only after a deprecation cycle (one minor release) when downstream code has migrated.

### Documentation

Public symbols should appear in API docs (Sphinx autodoc). Add narrative docs for new subsystems. Failing doc builds (warnings) block merge.

### Security / Compliance

- Run `pip-audit` periodically (or via CI) for dependency vulnerabilities.
- Avoid dynamic `exec` / unsafe `eval`.

### Release Process (Proposed)

1. Bump version in `pyproject.toml` (semantic versioning).
2. Update `CHANGELOG.md` (future addition) summarizing user-impacting changes.
3. Tag and push (`git tag vX.Y.Z && git push origin vX.Y.Z`).
4. CI builds wheels / sdist and (future) publishes to PyPI / internal index.

### Getting Help

Open a GitHub Discussion or Issue for design questions before large refactors.

Happy hacking!
