"""Run all curated experiments.

This is a convenience entrypoint that executes each experiment script in a clean, deterministic
order. Each experiment remains runnable on its own.

Outputs are written under:
- outputs/figures/
- outputs/tables/
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _run(script_path: Path) -> None:
    print(f"\n=== Running {script_path.relative_to(REPO_ROOT)} ===")
    runpy.run_path(str(script_path), run_name="__main__")


REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure repo-root imports (clw, data, etc.) work even if invoked from elsewhere.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    scripts = [
        REPO_ROOT / "experiments" / "poster_baseline.py",
        REPO_ROOT / "experiments" / "noise_state_oracle.py",
        REPO_ROOT / "experiments" / "noise_state_numerical.py",
        REPO_ROOT / "experiments" / "extended_library.py",
    ]

    missing = [p for p in scripts if not p.exists()]
    if missing:
        missing_str = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing experiment scripts:\n{missing_str}")

    for script in scripts:
        _run(script)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
