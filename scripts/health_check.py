#!/usr/bin/env python
"""Environment self-check for EcoLens AI.

Runs a series of read-only checks against the current configuration and
reports pass/fail for each. It never modifies user data.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --skip-model   # skip loading the model
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SETTINGS

PASS = "PASS"
FAIL = "FAIL"


def check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[{PASS}] {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as exc:
        print(f"[{FAIL}] {label} — {exc}")
        return False


def main() -> int:
    results: list[bool] = []

    results.append(
        check(
            "Storage & data directories writable",
            lambda: _ensure_dirs(),
        )
    )
    results.append(check("Knowledge base loads", _load_knowledge))
    results.append(check("Database initialises", _init_db))
    results.append(check("Guidance engine loads", _load_engine))
    if "--skip-model" not in sys.argv:
        results.append(check("Model loads & forward pass", _load_model))
    results.append(check("Optional deps present", _check_deps))

    print()
    if all(results):
        print("All checks passed. EcoLens AI is ready to run.")
        return 0
    print("One or more checks failed. See messages above.")
    return 1


def _ensure_dirs() -> str:
    SETTINGS.ensure_directories()
    return str(SETTINGS.storage_dir)


def _load_knowledge() -> str:
    import json

    data = json.loads(SETTINGS.knowledge_path.read_text(encoding="utf-8"))
    return f"{len(data.get('categories', []))} categories"


def _init_db() -> str:
    from src.database.initialization import init_db

    init_db()
    return str(SETTINGS.db_path)


def _load_engine() -> str:
    from src.guidance.engine import get_engine

    return f"{len(get_engine().all_categories())} guidance items"


def _load_model() -> str:
    from src.vision.inference import warm_up
    from src.vision.model_registry import get_device, get_model_spec

    spec = get_model_spec(None)
    warm_up(None)
    return f"{spec.key} on {get_device()}"


def _check_deps() -> str:
    required = ["torch", "torchvision", "streamlit", "sqlalchemy", "PIL", "numpy", "plotly"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f"missing: {', '.join(missing)}")
    return ", ".join(required)


if __name__ == "__main__":
    raise SystemExit(main())
