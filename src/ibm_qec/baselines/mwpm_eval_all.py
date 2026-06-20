#!/usr/bin/env python3
"""
Batch MWPM evaluation across all available (D, r, basis) combinations.

This utility discovers experiment configurations recorded under a results root,
invokes :func:`mwpm_eval.run_evaluation` for each combination, and aggregates the
per-combination summaries into a single JSON report. Individual summary JSON
files (and optional bar plots) are still produced alongside the aggregate output.

Example
-------
    python mwpm_eval_all.py --results-root results_testing

By default the script mirrors the directory conventions of ``mwpm_eval.py``:
results are read from ``./results_testing`` and artifacts are stored in
``./benchmark_decoder/LER_results``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from ibm_qec.baselines.mwpm_eval import run_evaluation


def _load_combinations(
    results_root: Path,
    *,
    distances: Optional[Set[int]] = None,
    rounds: Optional[Set[int]] = None,
    bases: Optional[Set[str]] = None,
) -> List[Tuple[int, int, str]]:
    """Discover unique (D, r, basis) triples present in `results_root`."""
    combos: Set[Tuple[int, int, str]] = set()

    for job_dir in sorted(results_root.iterdir()):
        results_file = job_dir / "results.json"
        if not results_file.is_file():
            continue

        try:
            with results_file.open("r", encoding="utf-8") as handle:
                circuits = json.load(handle)
        except json.JSONDecodeError:
            continue

        for circuit in circuits:
            metadata = circuit.get("metadata", {})
            d_val = metadata.get("D")
            t_val = metadata.get("n_syndrome_rounds")
            basis = metadata.get("basis")
            if d_val is None or t_val is None or basis is None:
                continue

            basis = str(basis).upper()
            if distances is not None and d_val not in distances:
                continue
            if rounds is not None and t_val not in rounds:
                continue
            if bases is not None and basis not in bases:
                continue

            combos.add((int(d_val), int(t_val), basis))

    return sorted(combos, key=lambda item: (item[0], item[1], item[2]))


def _parse_int_set(values: Optional[Iterable[int]]) -> Optional[Set[int]]:
    if values is None:
        return None
    parsed = {int(v) for v in values}
    return parsed if parsed else None


def _parse_str_set(values: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if values is None:
        return None
    parsed = {str(v).upper() for v in values}
    return parsed if parsed else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch MWPM evaluation across all detected repetition-code experiments."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results_testing",
        help="Directory containing raw experiment job subdirectories (default: ./results_testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_decoder" / "LER_results",
        help="Directory where per-combination summaries/plots will be stored.",
    )
    parser.add_argument(
        "--aggregate-path",
        type=Path,
        default=None,
        help="Optional path for the consolidated JSON report (default: output_dir/mwpm_all_summary.json).",
    )
    parser.add_argument(
        "--distance",
        type=int,
        nargs="*",
        help="Optional list of code distances to include.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="*",
        help="Optional list of syndrome rounds to include.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        nargs="*",
        help="Optional list of bases to include (subset of {X, Z}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of job directories to inspect per combination.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generation of bar-chart visualisations.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    args = parser.parse_args()

    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root '{args.results_root}' does not exist.")

    distances = _parse_int_set(args.distance)
    rounds = _parse_int_set(args.rounds)
    bases = _parse_str_set(args.basis)

    combos = _load_combinations(
        args.results_root,
        distances=distances,
        rounds=rounds,
        bases=bases,
    )
    if not combos:
        raise RuntimeError("No matching (D, r, basis) combinations found in the provided results directory.")

    aggregate = {}
    for d_val, t_val, basis in combos:
        if not args.quiet:
            print(f"\n=== Evaluating D={d_val}, r={t_val}, basis={basis} ===")
        summary = run_evaluation(
            distance=d_val,
            rounds=t_val,
            basis=basis,
            results_root=args.results_root,
            output_dir=args.output_dir,
            limit=args.limit,
            make_plot=not args.no_plot,
            quiet=args.quiet,
        )
        if summary is not None:
            key = f"D{d_val}_T{t_val}_{basis}"
            aggregate[key] = summary

    if not aggregate:
        print("No summaries were generated; aggregate report will not be written.")
        return

    aggregate_path = args.aggregate_path or (args.output_dir / "mwpm_all_summary.json")
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    if not args.quiet:
        print(f"\nWrote aggregate summary to {aggregate_path}")
        print(f"Processed {len(aggregate)} combinations.")


if __name__ == "__main__":
    main()
