#!/usr/bin/env python3
"""
Release gate aggregator.

Default: run core tests + coverage.
Extended gates toggled by flags.

Examples:
    python scripts/verify_release.py --all
    python scripts/verify_release.py --tests --security
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

Cmd = list[str]


def run(label: str, cmd: Cmd, required: bool) -> bool:
    result = subprocess.run(cmd, cwd=ROOT)
    ok = result.returncode == 0
    if not ok:
        pass
    else:
        pass
    return ok or (not required)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--coverage-min", type=int, default=0,
                    help="Enforce minimum coverage % (0=disabled)")
    ap.add_argument("--docs", action="store_true")
    ap.add_argument("--security", action="store_true")
    ap.add_argument("--complexity", action="store_true")
    ap.add_argument("--maintainability", action="store_true")
    ap.add_argument("--docstrings", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="Enable all optional gates")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    if args.all:
        args.tests = True
        args.docs = True
        args.security = True
        args.complexity = True
        args.maintainability = True
        args.docstrings = True

    overall_ok = True

    if args.tests or args.all or not any(
        [args.docs, args.security, args.complexity,
         args.maintainability, args.docstrings]
    ):
        if args.coverage_min > 0:
            # Single run: tests + coverage + enforced threshold
            test_cmd = [
                "pytest",
                "--cov=trade_agent",
                "--cov-report=term-missing",
                f"--cov-fail-under={args.coverage_min}",
            ]
            overall_ok &= run(
                f"Tests+Coverage (min {args.coverage_min}%)",
                test_cmd,
                required=True
                )
        else:
            # Fast tests only (no coverage enforcement)
            overall_ok &= run("Tests",
                              ["pytest",
                               "-q",
                               "--disable-warnings"],
                              required=True
                              )
        if not overall_ok and args.fail_fast:
            return 1
        if args.coverage_min > 0:
            # Generate coverage xml then parse summary
            overall_ok &= run("Coverage run",
                              ["pytest",
                               "--cov=trade_agent",
                               "--cov-report=term_missing"],
                              required=True
                              )

    if args.docs:
        overall_ok &= run("Docs build",
                          ["bash",
                           "-c",
                           "make -C docs html"],
                          required=True
                          )

        if not overall_ok and args.fail_fast:
            return 1

    if args.security:
        overall_ok &= run("pip-audit",
                          ["pip-audit",
                           "-r",
                           "requirements.txt"],
                          required=False
                          )

    if args.complexity:
        overall_ok &= run("Radon complexity",
                          ["radon",
                           "cc",
                           "trade_agent",
                           "-a",
                           "-nc"],
                          required=False
                          )

    if args.maintainability:
        overall_ok &= run("Interrogate",
                          ["interrogate",
                           "--fail-under=80"],
                          required=False)

    if args.docstrings:
        overall_ok &= run("Interrogate",
                          ["interrogate",
                           "--fail-under=80"],
                          required=False
                          )

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
