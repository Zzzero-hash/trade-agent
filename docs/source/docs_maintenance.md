## Documentation Maintenance Guidelines

This file summarizes the current Sphinx build status and provides guidance for maintaining and (optionally) pruning oversized markdown documents.

### 1. Sphinx Build Status

Last build: succeeded (no warnings reported).

Output directory: `docs/_build/html`

Recommended periodic check (add to CI):

```
sphinx-build -b html docs/source docs/_build/html -W --keep-going
```

Using `-W` will fail the build on new warnings so they are caught early.

### 2. Largest Markdown Files (Audit)

Sizes (bytes) and line counts of the largest docs (snapshot):

| File                                                | Bytes | Lines | Notes                                                                     |
| --------------------------------------------------- | ----- | ----- | ------------------------------------------------------------------------- |
| eval/usage_guide.md                                 | 15860 | 739   | Could split into quickstart + advanced + API usage                        |
| eval/acceptance_tests.md                            | 16379 | 624   | Consider moving raw test code references to `/tests` and summarizing here |
| sl/step4_sl_model_detailed_plan.md                  | 14590 | 470   | Stable; could extract versioning + persistence into separate doc          |
| sl/sl_model_acceptance_tests.md                     | 14311 | 467   | Similar to eval acceptance; summarize and link to canonical test suite    |
| envs/rl_environment_makefile_tasks.md               | 13809 | 372   | Many repetitive Makefile echo lines; compress via grouped bullet list     |
| features/step3_feature_engineering_detailed_plan.md | 13031 | 350   | Acceptable; ensure any pseudo‑code stays synced with implementation       |
| features/feature_engineering_makefile_tasks.md      | 11917 | 382   | Can be shortened by referencing task pattern once                         |
| agents/sac_agent_makefile_tasks.md                  | 11634 | 523   | Same pattern repetition; collapse                                         |
| agents/sac_agent_acceptance_tests.md                | 11535 | 436   | Summarize acceptance criteria; link to tests                              |
| sl/sl_model_rollback_plan.md                        | 11534 | 419   | Keep; optionally relocate scripts to `scripts/` and reference             |
| eval/implementation_plan.md                         | 11367 | 374   | OK                                                                        |
| agents/sac_agent_detailed_plan.md                   | 11234 | 365   | OK                                                                        |
| agents/ppo_file_tree_structure.md                   | 10950 | 428   | Could auto-generate from actual tree in CI                                |
| agents/ppo_agent_rollback_plan.md                   | 8533  | 423   | Combine with generic rollback template                                    |
| interfaces.md                                       | 9602  | 429   | Break into domain-specific interface docs if editing grows                |

Threshold guideline: if a single markdown file exceeds ~500 lines or 20KB, evaluate splitting or summarizing.

### 3. Recommended Refactors

1. Acceptance test docs: Replace full inline pseudo/test code blocks with a concise matrix (Test Name, Purpose, Key Assertion) and link to actual tests in `tests/`.
2. Makefile task lists: Show pattern once, then list task names in a table; avoid repeating nearly identical `@echo` lines.
3. Rollback plans: Factor out shared rollback procedure into `docs/source/common/rollback_template.md`; have component docs reference deltas.
4. Large usage guides: Split into: Quickstart, Detailed Guide, Advanced / Extensibility.
5. File tree docs: Auto-generate during CI using a script to avoid drift (store script in `scripts/generate_doc_trees.py`).

### 4. Suggested Action Sequence

| Order | Action                                  | Effort | Impact                     |
| ----- | --------------------------------------- | ------ | -------------------------- |
| 1     | Add Sphinx build with `-W` to CI        | Low    | High (prevents silent rot) |
| 2     | Compress acceptance test docs           | Medium | Medium                     |
| 3     | Consolidate repeated Makefile task docs | Medium | Medium                     |
| 4     | Introduce common rollback template      | Medium | Medium                     |
| 5     | Split `eval/usage_guide.md`             | Medium | Medium                     |
| 6     | Auto-generate file tree docs            | Medium | Low/Medium                 |

### 5. Proposed Automation Snippets

Tree generation script concept:

````python
import subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
tree = subprocess.run(['bash','-lc','tree -L 4 src'],capture_output=True,text=True).stdout
with open('docs/source/generated/src_tree.md','w') as f:
    f.write('# Source Tree (Generated)\n\n```text\n')
    f.write(tree)
    f.write('\n```\n')
````

### 6. Style Conventions

| Aspect       | Guideline                                                        |
| ------------ | ---------------------------------------------------------------- |
| Headings     | Start at H1 per file; keep hierarchy ≤ 4 levels                  |
| Code Blocks  | Prefer short illustrative snippets; link to source for long code |
| Diagrams     | Prefer Mermaid; ensure they render (no syntax errors)            |
| Line Length  | Soft wrap at 120 chars for readability                           |
| Front Matter | Start with a one-paragraph summary                               |

### 7. Checklist for New / Edited Docs

- [ ] Purpose clearly stated at top
- [ ] No duplicated large code blocks from repository
- [ ] References link to actual modules/tests
- [ ] File size < 20KB or justified in PR description
- [ ] Builds cleanly (no Sphinx warnings)

### 8. Next Optional Steps

If approved, next automated edits can: (a) compress acceptance test docs, (b) introduce shared rollback template, (c) add generation script + Git ignore for generated file.

Let me know which (if any) to execute next.
