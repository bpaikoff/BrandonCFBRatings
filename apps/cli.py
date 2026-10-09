"""Backwards-compatible entry point: ``python -m apps.cli`` or ``python apps/cli.py``.

Prefer the installed console script ``cfbratings`` (see ``cfbratings.cli``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfbratings.cli import main

if __name__ == "__main__":
    sys.exit(main())
