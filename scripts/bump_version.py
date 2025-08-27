#! /usr/bin/env python3
"""
Semantic version bump utility.

Usage:
    python scripts/bump_version.py --part patch
    python scripts/bump_version.py --part minor
    python scripts/bump_version.py --part major
    python scripts/bump_version.py --set 1.4.0
    python scripts/bump_version.py --print

Updates:
    trade_agent/__version__.py
    pyproject.toml (if version field present)

Exits non-zero on error.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Literal
import logging

logging.basicConfig(level=logging.INFO)

SEMVER_RE = re.compile(r'^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$')
Part = Literal["major", "minor", "patch"]
ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION_PY = ROOT / "version.py"
PYPROJECT = ROOT / "pyproject.toml"
VERSION_RE = re.compile(r'__version__\s*=\s*"(?P<version>\d+\.\d+\.\d+)"')


def read_current_version() -> str:
    content = VERSION_PY.read_text(encoding="utf-8")
    m = VERSION_RE.search(content)
    if not m:
        logging.error(f"__version__ not found in {VERSION_PY}")
        sys.exit(2)
    return m.group("version")


def write_version_py(new_version: str) -> None:
    content = VERSION_PY.read_text(encoding="utf-8")
    new_content = VERSION_RE.sub(f'__version__ = "{new_version}"',
                                 content, count=1)
    VERSION_PY.write_text(new_content, encoding="utf-8")


def maybe_update_pyproject(new_version: str) -> None:
    if not PYPROJECT.exists():
        return
    text = PYPROJECT.read_text(encoding="utf-8").splitlines()
    changed = False
    for i, line in enumerate(text):
        # naive: version = "x.y.z" (skip if using dynamic = ["version"])
        if line.strip().startswith("version="):
            text[i] = f'version = "{new_version}"'
            changed = True
            break
    if changed:
        PYPROJECT.write_text("\n".join(text) + "\n", encoding="utf-8")


def bump(version: str, part: str) -> str:
    major, minor, patch = map(int, version.split("."))
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(part)
    return f"{major}.{minor}.{patch}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--part", choices=["major", "minor", "patch"])
    g.add_argument("--set", dest="set_version")
    g.add_argument("--print", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def validate_version(v: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", v):
        logging.error(f"Invalid semantic version: {v}")
        sys.exit(3)


def main() -> int:
    args = parse_args()
    current = read_current_version()
    if args.print:
        logging.info(current)
        return 0
    new_version = bump(current, args.part) if args.part else args.set_version
    validate_version(new_version)
    if new_version == current:
        logging.info(f"Version is already {current}, no change.")
        return 1
    if args.dry_run:
        logging.info(f"[dry-run] {current} -> {new_version}")
        return 0
    write_version_py(new_version)
    maybe_update_pyproject(new_version)
    logging.info(f"{current} -> {new_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
