"""Run the self-hosted GestureBind telemetry collector."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.services.telemetry_collector import serve_collector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("TELEMETRY_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("TELEMETRY_PORT", "8787"))
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.environ.get("TELEMETRY_DATABASE", "telemetry.sqlite")),
    )
    parser.add_argument(
        "--project-key",
        default=os.environ.get("TELEMETRY_PROJECT_KEY", ""),
    )
    parser.add_argument(
        "--admin-token",
        default=os.environ.get("TELEMETRY_ADMIN_TOKEN", ""),
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("TELEMETRY_RETENTION_DAYS", "90")),
    )
    parser.add_argument("--allow-insecure-local", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    local = args.host in {"127.0.0.1", "localhost", "::1"}
    if not args.project_key or not args.admin_token:
        if not (local and args.allow_insecure_local):
            raise SystemExit(
                "TELEMETRY_PROJECT_KEY and TELEMETRY_ADMIN_TOKEN are required; "
                "use --allow-insecure-local only for an isolated local check"
            )
    print(
        f"[telemetry] listening on http://{args.host}:{args.port}; "
        f"database={args.database}",
        flush=True,
    )
    serve_collector(
        host=args.host,
        port=args.port,
        database_path=args.database,
        project_key=args.project_key,
        admin_token=args.admin_token,
        retention_days=args.retention_days,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
