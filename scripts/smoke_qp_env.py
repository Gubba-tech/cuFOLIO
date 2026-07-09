"""Print versions for the QP development environment."""

from __future__ import annotations

import importlib
import sys


MODULES = [
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "pytest",
    "cvxpy",
]


def main() -> int:
    failed = False
    print(f"python {sys.version.split()[0]}")
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failed = True
            print(f"{module_name}: IMPORT FAILED: {exc}")
            continue
        version = getattr(module, "__version__", "<missing __version__>")
        print(f"{module_name}: {version}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
