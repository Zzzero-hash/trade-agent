# Documentation

Sphinx documentation lives in `docs/source`.

## Build Locally

```
make docs
open docs/_build/html/index.html  # or xdg-open on Linux
```

## Live Reload (optional)

Install `sphinx-autobuild` if desired:

```
pip install sphinx-autobuild
make docs-live
```

## GitHub Pages

On pushes to `master`, the GitHub Actions workflow `docs.yml` builds and deploys the docs to GitHub Pages. Enable Pages in the repository settings pointing to `GitHub Actions` as the source.
