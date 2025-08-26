# Release Checklist

## 1. Pre-flight

- [ ] Working tree clean: `git status`
- [ ] Latest main merged

## 2. Version

- [ ] Decide bump (major/minor/patch)
- [ ] Run: `python scripts/bump_version.py -- part <patch|minor|major>`
- [ ] Commit: `feat(release): bump version to X.Y.Z`
- [ ] Tag (later automated): `git tag vX.Y.Z`

## 3. Quality Gates

- [ ] Run: `python scripts/verify_release.py --tests`
- [ ] Full gates: `python scripts/verify_release.py --all`
- [ ] Coverage threshold >= 70% `python scripts/verify_release.py --all --coverage-min 70`
- [ ] Complexity / Maintainability reviewed (if run)

## 4. Security

- [ ] `python scripts/verify_release.py --security`
- [ ] Review high/critical findings

## 5. Documentation

- [ ] Build docs: `python scripts/verify_release.py --docs`
- [ ] Changelog updated (add section for version)

## 6. Artifacts

- [ ] sdist / wheel build test: `python -m build`
- [ ] Install fresh venv smoke test

## 7. Tag & Push

- [ ] `git push && git push --tags`

## 8. Post Release

- [ ] Create GitHub Release (attach notes)
- [ ] Open next dev cycle issue (roadmap adjustments)

## 9. Telemetry (Future)

- [ ] Ensure opt-in path documented
- [ ] Event schema changes versioned

Notes:

- Scripts are scaffolds; tighten thresholds as project matures.
