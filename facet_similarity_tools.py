#!/usr/bin/env python3
"""CLI entry point: run from repository root so paths resolve."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from playerbert.similarity import main

if __name__ == "__main__":
    main()
