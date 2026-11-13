"""Command-line interface: ``glacier-pipeline download`` and ``glacier-pipeline run``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glacier-pipeline",
        description="Glacier mapping in the Austrian Alps from Sentinel-2",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="run",
        choices=["download", "run"],
        help="download: fetch and cache all input data; run: download (if needed) + full pipeline",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration file")
    parser.add_argument("--workspace", default=".", help="Project root directory")
    parser.add_argument(
        "--quick", action="store_true", help="Tiny models, for a fast end-to-end check"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).resolve()
    config_path = (workspace / args.config).resolve()

    if args.stage == "download":
        from glacier_monitoring.data.ingest import download_all, fetch_geosphere_climate
        from glacier_monitoring.pipeline import load_yaml_config

        cfg = load_yaml_config(config_path)
        years = list(cfg["data"]["years"])
        download_all(
            workspace / "data" / "external",
            workspace / "data" / "raw",
            years,
            float(cfg["data"]["pixel_size_m"]),
        )
        fetch_geosphere_climate(years, workspace / "data" / "external")
        return 0

    from glacier_monitoring.pipeline import run_full_pipeline

    summary = run_full_pipeline(workspace, config_path, quick=args.quick)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
