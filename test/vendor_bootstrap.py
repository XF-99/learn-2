# -*- coding: utf-8 -*-
"""Make project-local dependencies importable before third-party imports."""
from pathlib import Path
import os
import sys


def activate() -> None:
    if os.environ.get("LEARN_SKIP_VENDOR") == "1":
        return
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / "vendor", script_dir.parent / "vendor"]
    for vendor in candidates:
        if not vendor.is_dir():
            continue
        vendor_path = str(vendor)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        break


activate()
