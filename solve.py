#!/usr/bin/env python3
"""Run the CG:SHOP 2027 solver on instances and write solutions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cgshop2027_pyutils.instance_database import InstanceDatabase
from cgshop2027_pyutils.io import read_instance
from cgshop2027_pyutils.zip import ZipWriter

from solver.coverage import solve_instance, verify_solution


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve CG:SHOP 2027 instances and write solution files."
    )
    parser.add_argument(
        "instances",
        type=Path,
        help="Path to an instance .json file, or a folder/zip of instances.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("solutions"),
        help="Output directory for .solution.json files (default: solutions).",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Also write all solutions into this zip archive.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Solve at most this many instances (for quick tests).",
    )
    return parser.parse_args()


def _iter_instances(path: Path):
    if path.is_file() and path.suffix == ".json":
        yield read_instance(path)
        return
    db = InstanceDatabase(path)
    for instance in db:
        yield instance


def main() -> int:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    solved = 0
    failed = 0
    zip_solutions = []

    for instance in _iter_instances(args.instances):
        if args.limit is not None and solved + failed >= args.limit:
            break
        solution = solve_instance(instance)
        errors = verify_solution(instance, solution)
        out_path = args.output / f"{instance.instance_uid}.solution.json"
        out_path.write_text(solution.model_dump_json(indent=2) + "\n", encoding="utf-8")
        zip_solutions.append(solution)

        if errors:
            failed += 1
            print(
                f"FAIL {instance.instance_uid}: {errors[0]}",
                file=sys.stderr,
            )
        else:
            solved += 1
            print(
                f"OK   {instance.instance_uid}: "
                f"max_tour_length={solution.max_tour_length}"
            )

    if args.zip and zip_solutions:
        with ZipWriter(args.zip) as writer:
            for solution in zip_solutions:
                writer.add_solution(solution)
        print(f"Wrote {len(zip_solutions)} solutions to {args.zip}")

    print(f"Done: {solved} feasible, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
