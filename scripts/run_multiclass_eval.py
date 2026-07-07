#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_independent_eval import main


if __name__ == "__main__":
    provided = set(sys.argv[1:])
    if "--quality-mode" not in provided:
        sys.argv.extend(["--quality-mode", "multiclass"])
    if "--skip-speed" not in provided:
        sys.argv.append("--skip-speed")
    main()
