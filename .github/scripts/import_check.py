#!/usr/bin/env python
"""
CI helper: parses (syntax check) and imports (runtime check) every top-level
.py file in the given directories, reporting failures with real tracebacks
instead of pytest collection noise. Not part of the runtime pipeline --
referenced only from .github/workflows/lint.yml.

Usage:
    python .github/scripts/import_check.py training dataset scripts
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def check_directory(dir_name: str) -> list[str]:
    target_dir = REPO_ROOT / dir_name
    failures = []
    sys.path.insert(0, str(target_dir))

    try:
        for py_file in sorted(target_dir.glob("*.py")):
            module_name = py_file.stem
            try:
                ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError as e:
                failures.append(f"{py_file.relative_to(REPO_ROOT)}: syntax error: {e}")
                continue

            try:
                importlib.import_module(module_name)
                print(f"OK: {py_file.relative_to(REPO_ROOT)}")
            except Exception as e:  # noqa: BLE001 - intentionally broad, this is a CI diagnostic
                failures.append(
                    f"{py_file.relative_to(REPO_ROOT)}: import error: {type(e).__name__}: {e}"
                )
            finally:
                sys.modules.pop(module_name, None)
    finally:
        sys.path.remove(str(target_dir))

    return failures


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: import_check.py <dir> [<dir> ...]")
        sys.exit(2)

    all_failures = []
    for dir_name in sys.argv[1:]:
        all_failures.extend(check_directory(dir_name))

    if all_failures:
        print("\nImport check failures:")
        for failure in all_failures:
            print(f" - {failure}")
        sys.exit(1)

    print("\nAll files parsed and imported successfully.")


if __name__ == "__main__":
    main()
