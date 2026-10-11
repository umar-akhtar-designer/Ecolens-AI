#!/usr/bin/env python
"""Reset EcoLens AI local data.

By default this removes all analysis records and their stored images/explanations
and recreates an empty schema. It never touches source code or model weights.

Usage:
    python scripts/reset_database.py --yes     # destructive, no prompt
    python scripts/reset_database.py           # prompts for confirmation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SETTINGS
from src.database.initialization import reset_db
from src.database.repository import delete_all


def main() -> int:
    SETTINGS.ensure_directories()
    if "--yes" not in sys.argv:
        answer = input(
            f"This will delete ALL records in {SETTINGS.db_path} and stored files. "
            "Type 'yes' to continue: "
        )
        if answer.strip().lower() != "yes":
            print("Aborted.")
            return 1

    removed = delete_all()
    reset_db()
    print(f"Deleted {removed} analysis record(s). Database reset with an empty schema.")
    print(f"Storage: {SETTINGS.storage_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
