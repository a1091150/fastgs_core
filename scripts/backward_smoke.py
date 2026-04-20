#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def run(script: str) -> int:
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / script)]
    print(f"[INFO] running {script}")
    return subprocess.call(cmd)


def main() -> None:
    codes = [
        run("backward_vjp_smoke.py"),
        run("rasterize_backward_validation.py"),
        run("preprocess_backward_validation.py"),
        run("e2e_backward_smoke.py"),
    ]
    if all(c == 0 for c in codes):
        print("[OK] backward smoke suite passed")
        return
    print("[FAIL] backward smoke suite failed")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
