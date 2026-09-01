#!/usr/bin/env python3
"""Generate a sequential HTCondor DAG for a run-number and run-id range.

The output matches the structure used by ``sub-in-silico-hVOS-copy.dag``:

- four jobs per run block
- the same ``run_name`` variable passed to each job in the block
- a cross-run dependency from each run's ``package`` job to the next run's ``S1`` job
"""

from __future__ import annotations

import argparse
from pathlib import Path


def format_run_id(run_number: str, run_id: int, width: int) -> str:
    return f"run{run_number}_{run_id:0{width}d}"


def build_run_block(run_name: str) -> list[str]:
    return [
        "############################",
        f"#{run_name}",
        "############################",
        "",
        f"JOB {run_name}_S1 subdag-job.sub",
        f'VARS {run_name}_S1 run_name="{run_name}"',
        "",
        f"JOB {run_name}_extract job-extract.sub",
        f'VARS {run_name}_extract run_name="{run_name}"',
        "",
        f"JOB {run_name}_optical job-analyze-hVOS.sub",
        f'VARS {run_name}_optical run_name="{run_name}"',
        "",
        f"JOB {run_name}_package package.sub",
        f'VARS {run_name}_package run_name="{run_name}"',
        "",
        "",
        f"PARENT {run_name}_S1 CHILD {run_name}_extract",
        f"PARENT {run_name}_extract CHILD {run_name}_optical",
        f"PARENT {run_name}_optical CHILD {run_name}_package",
        "",
        "",
    ]


def build_dag(run_number: str, start_run_id: str, end_run_id: str) -> str:
    start = int(start_run_id)
    end = int(end_run_id)
    if start > end:
        raise ValueError("start_run_id must be less than or equal to end_run_id")

    width = max(len(start_run_id), len(end_run_id))
    width = max(width, 2)  # ensure at least 2 digits for run_id formatting
    run_names = [format_run_id(run_number, run_id, width) for run_id in range(start, end + 1)]

    lines: list[str] = []
    for index, run_name in enumerate(run_names):
        lines.extend(build_run_block(run_name))
        if index < len(run_names) - 1:
            next_run_name = run_names[index + 1]
            lines.extend([
                "############################",
                "# cross-run dependency",
                "############################",
                "",
                f"PARENT {run_name}_package CHILD {next_run_name}_S1",
                "",
                "",
            ])

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sequential HTCondor DAG over a run-id range."
    )
    parser.add_argument("run_number", help="Run number prefix, for example 15")
    parser.add_argument("start_run_id", help="First run id in the range, for example 56 or 00")
    parser.add_argument("end_run_id", help="Last run id in the range, for example 59 or 88")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output DAG path. Defaults to sub-in-silico-hVOS-run<run_number>.dag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output or f"sub-in-silico-hVOS-run{args.run_number}.dag")
    output_path.write_text(build_dag(args.run_number, args.start_run_id, args.end_run_id), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()