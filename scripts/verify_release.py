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

COLOR = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}


def color(txt: str, c: str) -> str:
    return f"{COLOR[c]}{txt}{COLOR['reset']}"


def run(label: str, cmd: Cmd, required: bool) -> bool:
    print(f"\n== {label}\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    ok = result.returncode == 0
    status = "PASS" if ok else ("WARN" if not required else "FAIL")
    col = "green" if ok else ("yellow" if not required else "red")
    print(f"[{color(status, col)}] {label}")
    return ok or (not required)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--coverage-min", type=int, default=0,
                    help="Enforce minimum coverage % (0=just report)")
    ap.add_argument("--docs", action="store_true")
    ap.add_argument("--security", action="store_true")
    ap.add_argument("--complexity", action="store_true")
    ap.add_argument("--maintainability", action="store_true",
                    help="Run Radon MI (maintainability index)")
    ap.add_argument("--mi-min", type=int, default=0,
                    help="Minimum Radon MI (0 disables threshold)")
    ap.add_argument("--docstrings", action="store_true",
                    help="Run Interrogate for docstring coverage")
    ap.add_argument("--all", action="store_true",
                    help="Enable all optional gates")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--cov-package", default="trade_agent",
                    help="Root package for coverage (default: trade_agent)")
    args = ap.parse_args()

    if args.all:
        args.tests = True
        args.docs = True
        args.security = True
        args.complexity = True
        args.maintainability = True
        args.docstrings = True

    overall_ok = True

    # Decide whether to run tests (default if no other gate chosen)
    run_tests = (
        args.tests
        or args.all
        or not any(
            [args.docs, args.security, args.complexity,
             args.maintainability, args.docstrings
             ])
    )
    if run_tests:
        # Always collect coverage; Threshold only if coverage-min > 0
        test_cmd = [
            "pytest",
            f"--cov={args.cov_package}",
            "--cov-report=term-missing",
        ]
        if args.coverage_min > 0:
            test_cmd.append(f"--cov-fail-under={args.coverage_min}")
            label = f"Tests+Coverage (min {args.coverage_min}%)"
        else:
            label = "Tests+Coverage"
        overall_ok &= run(label,
                          test_cmd,
                          required=True)
        if not overall_ok and args.fail_fast:
            return 1

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
                           args.cov_package,
                           "-a",
                           "-nc"],
                          required=False
                          )

    if args.maintainability:
        mi_cmd = ["radon", "mi", args.cov_package, "-s"]
        if args.mi_min > 0:
            mi_cmd.extend(["-n", str(args.mi_min)])
            label = f"Maintainability (Radon MI >= {args.mi_min})"
        else:
            label = "Maintainability (Radon MI report)"
        overall_ok &= run(label, mi_cmd, required=False)

    if args.docstrings:
        # Single Interrogate invocation
        overall_ok &= run("Docstrings (Interrogate >= 80%)",
                          ["interrogate",
                           "--fail-under=80"],
                          required=False
                          )

    print("\n== Summary ==")
    print(color("SUCCESS" if overall_ok else "FAILURE",
                "green" if overall_ok else "red"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
