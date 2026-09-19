#!/usr/bin/env python3
"""Run a CG:SHOP 2027 solver on instances and write solutions."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

from cgshop2027_pyutils.instance_database import InstanceDatabase
from cgshop2027_pyutils.io import read_instance
from cgshop2027_pyutils.zip import ZipWriter

from solver.coverage import verify_solution as default_verify
import solver as solver_pkg


def _solver_names() -> list[str]:
    names: list[str] = []
    for info in pkgutil.iter_modules(solver_pkg.__path__):
        if not info.name.startswith("_"):
            names.append(info.name)
    return sorted(names)


def _load_solver(name: str):
    try:
        module = importlib.import_module(f"solver.{name}")
    except ModuleNotFoundError:
        available = ", ".join(_solver_names()) or "(none)"
        print(
            f"No solver/{name}.py. Available: {available}.\n"
            f"Add solver/{name}.py with a solve_instance(instance) function.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    solve_fn = getattr(module, "solve_instance", None)
    if solve_fn is None:
        print(
            f"solver/{name}.py needs a solve_instance(instance) function.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    verify_fn = getattr(module, "verify_solution", default_verify)
    return solve_fn, verify_fn


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
    parser.add_argument(
        "--algo",
        default="coverage",
        help=(
            "Solver module under solver/ (filename without .py). "
            "Examples: coverage, minmax, sweep, cluster. "
            f"Found now: {', '.join(_solver_names()) or 'none'}."
        ),
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
    solve_fn, verify_fn = _load_solver(args.algo)
    args.output.mkdir(parents=True, exist_ok=True)

    solved = 0
    failed = 0
    zip_solutions = []

    for instance in _iter_instances(args.instances):
        if args.limit is not None and solved + failed >= args.limit:
            break
        solution = solve_fn(instance)
        errors = verify_fn(instance, solution)
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
