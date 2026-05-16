# -*- coding: utf-8 -*-
"""Make project-local dependencies importable before third-party imports."""
from pathlib import Path
import sys


def activate() -> None:
    vendor = Path(__file__).resolve().parent / "vendor"
    if vendor.is_dir():
        vendor_path = str(vendor)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)


activate()
