#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("/private/tmp/fastgs_recorded_reference/recorded_manifest.json")
DEFAULT_SWIFT_SUMMARY = Path("/private/tmp/fastgs_recorded_reference/recorded_swift_stage_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python/C++ recorded FastGS stage summaries with Swift summaries."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--swift-summary", type=Path, default=DEFAULT_SWIFT_SUMMARY)
    parser.add_argument("--abs-tol", type=float, default=5.0e-2)
    parser.add_argument("--rel-tol", type=float, default=1.0e-5)
    parser.add_argument("--max-diffs", type=int, default=80)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def integer_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 10)
        except ValueError:
            return None
    return None


def compare_values(
    path: str,
    expected: Any,
    actual: Any,
    abs_tol: float,
    rel_tol: float,
    diffs: list[str],
) -> None:
    expected_int = integer_value(expected)
    actual_int = integer_value(actual)
    if expected_int is not None and actual_int is not None:
        if expected_int != actual_int:
            diffs.append(f"{path}: expected {expected!r}, got {actual!r}")
        return

    expected_num = numeric_value(expected)
    actual_num = numeric_value(actual)
    if expected_num is not None and actual_num is not None:
        abs_diff = abs(expected_num - actual_num)
        rel_limit = rel_tol * max(abs(expected_num), abs(actual_num), 1.0)
        if abs_diff > max(abs_tol, rel_limit):
            diffs.append(
                f"{path}: expected {expected!r}, got {actual!r}, abs_diff={abs_diff:.8g}, "
                f"limit={max(abs_tol, rel_limit):.8g}"
            )
        return

    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            diffs.append(f"{path}.{key}: missing from Swift summary")
        for key in sorted(actual_keys - expected_keys):
            diffs.append(f"{path}.{key}: extra in Swift summary")
        for key in sorted(expected_keys & actual_keys):
            child_path = f"{path}.{key}" if path else key
            compare_values(child_path, expected[key], actual[key], abs_tol, rel_tol, diffs)
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length expected {len(expected)}, got {len(actual)}")
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            compare_values(f"{path}[{index}]", expected_item, actual_item, abs_tol, rel_tol, diffs)
        return

    if expected != actual:
        diffs.append(f"{path}: expected {expected!r}, got {actual!r}")


def max_numeric_difference(expected: Any, actual: Any) -> tuple[str, float]:
    best_path = ""
    best_diff = 0.0

    def visit(path: str, left: Any, right: Any) -> None:
        nonlocal best_path, best_diff
        left_num = numeric_value(left)
        right_num = numeric_value(right)
        if left_num is not None and right_num is not None:
            diff = abs(left_num - right_num)
            if diff > best_diff:
                best_path = path
                best_diff = diff
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) & set(right)):
                visit(f"{path}.{key}" if path else key, left[key], right[key])
            return
        if isinstance(left, list) and isinstance(right, list):
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(f"{path}[{index}]", left_item, right_item)

    visit("", expected, actual)
    return best_path, best_diff


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    swift_summary = load_json(args.swift_summary)
    expected = manifest.get("stageSummaries")
    if not isinstance(expected, dict):
        raise RuntimeError(f"{args.manifest} does not contain stageSummaries")

    diffs: list[str] = []
    compare_values("", expected, swift_summary, args.abs_tol, args.rel_tol, diffs)
    max_path, max_diff = max_numeric_difference(expected, swift_summary)

    label = f"{manifest.get('pointCount', '?')} points @ {manifest.get('width', '?')}x{manifest.get('height', '?')}"
    if diffs:
        print(f"[FAIL] recorded stage summary differs: {label}")
        print(f"manifest: {args.manifest}")
        print(f"swift: {args.swift_summary}")
        print(f"max numeric diff: {max_diff:.8g} at {max_path or '<none>'}")
        for diff in diffs[: args.max_diffs]:
            print("-", diff)
        if len(diffs) > args.max_diffs:
            print(f"... {len(diffs) - args.max_diffs} more diffs")
        raise SystemExit(1)

    print(f"[OK] recorded stage summary matches: {label}")
    print(f"manifest: {args.manifest}")
    print(f"swift: {args.swift_summary}")
    if math.isfinite(max_diff):
        print(f"max numeric diff: {max_diff:.8g} at {max_path or '<none>'}")
    if manifest.get("predPng"):
        print(f"python pred: {manifest['predPng']}")
    if manifest.get("sideBySidePng"):
        print(f"python side-by-side: {manifest['sideBySidePng']}")
    swift_png = args.swift_summary.with_name("recorded_swift.png")
    if swift_png.exists():
        print(f"swift png: {swift_png}")


if __name__ == "__main__":
    main()
