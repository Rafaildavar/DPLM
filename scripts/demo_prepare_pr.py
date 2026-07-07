#!/usr/bin/env python3
"""Prepare a safe PR demo report without committing, pushing, or creating a PR."""

from __future__ import annotations

import platform
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "demo_pr"
REPORT = OUT_DIR / "pr_demo_report.md"
COMPARE_URL = "https://github.com/Rafaildavar/GestureBind/compare"


def run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git {' '.join(args)} failed: {exc}"
    return result.stdout.strip() or "(empty)"


def open_target(target: Path | str) -> None:
    value = str(target)
    system = platform.system().lower()
    try:
        if system == "darwin":
            subprocess.Popen(["open", value])
        elif system == "windows":
            subprocess.Popen(["cmd", "/c", "start", "", value])
        else:
            subprocess.Popen(["xdg-open", value])
    except OSError:
        webbrowser.open(value)


def build_report() -> str:
    branch = run_git("branch", "--show-current")
    status = run_git("status", "--short")
    diff_stat = run_git("diff", "--stat")
    staged_stat = run_git("diff", "--cached", "--stat")

    return "\n".join(
        [
            "# PR Demo Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Repository: {ROOT}",
            f"Branch: {branch}",
            "",
            "## Safety",
            "",
            "This demo does not commit, push, or create a pull request.",
            "It prepares local context and opens the comparison page for manual review.",
            "",
            "## Working Tree",
            "",
            "```text",
            status,
            "```",
            "",
            "## Unstaged Diff Stat",
            "",
            "```text",
            diff_stat,
            "```",
            "",
            "## Staged Diff Stat",
            "",
            "```text",
            staged_stat,
            "```",
            "",
            f"Compare page: {COMPARE_URL}",
            "",
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(), encoding="utf-8")
    open_target(REPORT)
    webbrowser.open(COMPARE_URL)


if __name__ == "__main__":
    main()
