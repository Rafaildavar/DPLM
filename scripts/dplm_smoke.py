#!/usr/bin/env python3
"""Smoke self-test for the GestureBind Flet assistant.

Usage:
    python -m scripts.dplm_smoke
    python -m scripts.dplm_smoke --camera
"""
from __future__ import annotations

import argparse
import json
import sys

from app.services.diagnostics import (
    STATUS_FAIL,
    STATUS_INFO,
    STATUS_PASS,
    STATUS_WARN,
    run_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GestureBind assistant smoke self-test")
    parser.add_argument("--camera", action="store_true", help="Open the configured camera")
    parser.add_argument("--no-db", action="store_true", help="Skip database healthcheck")
    parser.add_argument("--json", action="store_true", help="Print raw JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_diagnostics(
        check_camera=bool(args.camera),
        check_database=not bool(args.no_db),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0 if report["ok"] else 1


def _print_report(report: dict) -> None:
    status_labels = {
        STATUS_PASS: "OK",
        STATUS_WARN: "WARN",
        STATUS_FAIL: "FAIL",
        STATUS_INFO: "INFO",
    }
    summary = report["summary"]
    print("GestureBind smoke self-test")
    print(f"Database URL: {report['databaseUrl']}")
    print(
        "Summary: "
        f"OK={summary.get(STATUS_PASS, 0)} "
        f"WARN={summary.get(STATUS_WARN, 0)} "
        f"FAIL={summary.get(STATUS_FAIL, 0)} "
        f"INFO={summary.get(STATUS_INFO, 0)}"
    )
    print()
    for item in report["items"]:
        status = status_labels.get(item["status"], item["status"].upper())
        print(f"[{status:4}] {item['label']}: {item['detail']}")


if __name__ == "__main__":
    sys.exit(main())
