#!/usr/bin/env python3
"""
Evaluate the MWPM decoder on IBM repetition-code experiments.

This standalone script scans the raw experiment directories (`results_testing/`)
for jobs matching a given repetition-code distance `D`, number of syndrome rounds
`T`, and preparation basis (`X` or `Z`). For every chain contained in those jobs it
  - reconstructs detection-event histories,
  - decodes them with a repetition-code matching graph (PyMatching),
  - accumulates logical-error counts for |0> and |1> shots, and
  - writes a summary JSON plus (optionally) a per-chain bar chart visualising LERs.

Example
-------
    python mwpm_eval.py -d 5 -t 3 -b X

Outputs
-------
    benchmark_decoder/LER_results/mwpm_D5_T3_X.json
    benchmark_decoder/LER_results/mwpm_D5_T3_X.png   (unless --no-plot)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pymatching

try:  # matplotlib is optional; statistics still work without it.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover – optional dependency
    plt = None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_repetition_matching(distance: int, rounds: int) -> Tuple[pymatching.Matching, np.ndarray]:
    """Construct the repetition-code parity-check matrix and faults matrix."""
    stabilizers = distance - 1
    num_detectors = stabilizers * rounds

    data_faults = distance * rounds
    meas_faults = stabilizers * rounds
    total_faults = data_faults + meas_faults

    h_matrix = np.zeros((num_detectors, total_faults), dtype=np.uint8)
    faults_matrix = np.zeros((1, total_faults), dtype=np.uint8)

    # Data-qubit X errors.
    for t in range(rounds):
        for q in range(distance):
            col = t * distance + q
            if q > 0:
                h_matrix[t * stabilizers + (q - 1), col] ^= 1
            if q < distance - 1:
                h_matrix[t * stabilizers + q, col] ^= 1
            faults_matrix[0, col] = 1  # Any data X error flips logical Z.

    # Measurement errors (flip a stabilizer outcome).
    for t in range(rounds):
        for s in range(stabilizers):
            col = data_faults + t * stabilizers + s
            h_matrix[t * stabilizers + s, col] ^= 1
            if t + 1 < rounds:
                h_matrix[(t + 1) * stabilizers + s, col] ^= 1

    matching = pymatching.Matching(h_matrix, faults_matrix=faults_matrix)
    return matching, faults_matrix


def compute_detection_vectors(syndrome_shots: np.ndarray, rounds: int, stabilizers: int) -> np.ndarray:
    """Convert raw syndrome histories into detection-event vectors."""
    if syndrome_shots.ndim != 2:
        raise ValueError("Expected 2D array for syndrome shots.")
    expected_width = stabilizers * rounds
    if syndrome_shots.shape[1] != expected_width:
        raise ValueError(
            f"Syndrome shot width mismatch: expected {expected_width} columns, "
            f"received {syndrome_shots.shape[1]}."
        )

    shots = syndrome_shots.reshape(-1, rounds, stabilizers)
    detections = np.empty_like(shots)
    prev = np.zeros((shots.shape[0], stabilizers), dtype=np.uint8)
    for t in range(rounds):
        curr = shots[:, t, :]
        detections[:, t, :] = curr ^ prev
        prev = curr
    return detections.reshape(shots.shape[0], expected_width)


def decode_logical_flips(
    matching: pymatching.Matching,
    faults_matrix: np.ndarray,
    detector_vectors: np.ndarray,
) -> np.ndarray:
    """Return an array indicating which shots incurred logical flips after decoding."""
    logical_mask = faults_matrix[0].astype(bool)
    flips = np.zeros(detector_vectors.shape[0], dtype=np.uint8)

    for idx, det_vec in enumerate(detector_vectors):
        correction = matching.decode(det_vec.astype(np.uint8))
        correction = np.asarray(correction, dtype=np.uint8)

        if correction.size == faults_matrix.shape[0]:
            # Some PyMatching versions return logical flips directly when a faults_matrix is provided.
            flips[idx] = int(correction[0] & 1)
        else:   
            flips[idx] = int(correction[logical_mask].sum() % 2)

    return flips


def scan_jobs(
    results_root: Path,
    distance: int,
    rounds: int,
    basis: str,
    job_limit: int | None = None,
) -> Tuple[Dict[str, Dict[int, Dict[str, int]]], Dict[int, Dict[str, int]], List[str]]:
    """Scan experiment directories and accumulate logical-error statistics."""
    matching, faults_matrix = build_repetition_matching(distance, rounds)
    stabilizers = distance - 1

    chain_accumulator: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(
        lambda: {0: {"shots": 0, "errors": 0}, 1: {"shots": 0, "errors": 0}}
    )
    totals_by_state: Dict[int, Dict[str, int]] = {
        0: {"shots": 0, "errors": 0},
        1: {"shots": 0, "errors": 0},
    }
    processed_jobs: List[str] = []

    jobs_seen = 0
    for job_dir in sorted(results_root.iterdir()):
        if job_limit is not None and jobs_seen >= job_limit:
            break
        results_file = job_dir / "results.json"
        if not results_file.is_file():
            continue

        circuits = load_json(results_file)
        job_used = False

        for circuit in circuits:
            metadata = circuit.get("metadata", {})
            if (
                metadata.get("D") != distance
                or metadata.get("n_syndrome_rounds") != rounds
                or metadata.get("basis") != basis
            ):
                continue

            logical_state = int(metadata.get("logical_state", 0))
            per_cregs = circuit.get("per_shot_cregs", {})

            for data_key, data_shots in per_cregs.items():
                if not data_key.startswith("c_data_"):
                    continue
                suffix = data_key[len("c_data_") :]
                syndrome_key = f"c_syndrome_{suffix}"
                if syndrome_key not in per_cregs:
                    continue

                data_arr = np.asarray(per_cregs[data_key], dtype=np.uint8)
                syndrome_arr = np.asarray(per_cregs[syndrome_key], dtype=np.uint8)

                if data_arr.ndim != 2 or data_arr.shape[1] != distance:
                    continue
                if syndrome_arr.ndim != 2:
                    continue

                try:
                    detectors = compute_detection_vectors(syndrome_arr, rounds, stabilizers)
                except ValueError:
                    continue  # Skip malformed data.

                num_shots = data_arr.shape[0]
                if num_shots == 0:
                    continue

                logical_flips = decode_logical_flips(matching, faults_matrix, detectors)
                majority = (data_arr.sum(axis=1) > (distance // 2)).astype(np.uint8)
                final_logical = majority ^ logical_flips
                errors = int((final_logical != logical_state).sum())

                chain_accumulator[suffix][logical_state]["shots"] += num_shots
                chain_accumulator[suffix][logical_state]["errors"] += errors
                totals_by_state[logical_state]["shots"] += num_shots
                totals_by_state[logical_state]["errors"] += errors

                job_used = True

        if job_used:
            processed_jobs.append(job_dir.name)
            jobs_seen += 1

    return chain_accumulator, totals_by_state, processed_jobs


def summarise_results(
    chain_accumulator: Dict[str, Dict[int, Dict[str, int]]],
    totals_by_state: Dict[int, Dict[str, int]],
    distance: int,
    rounds: int,
    basis: str,
    processed_jobs: List[str],
) -> Dict:
    """Convert raw counts into logical-error-rate summary statistics."""
    chains_summary = {}
    for chain_id, chain_data in chain_accumulator.items():
        shots_0 = chain_data[0]["shots"]
        shots_1 = chain_data[1]["shots"]
        errors_0 = chain_data[0]["errors"]
        errors_1 = chain_data[1]["errors"]
        shots_total = shots_0 + shots_1
        errors_total = errors_0 + errors_1

        chains_summary[chain_id] = {
            "shots_total": shots_total,
            "errors_total": errors_total,
            "ler_total": errors_total / shots_total if shots_total else None,
            "shots_0": shots_0,
            "errors_0": errors_0,
            "ler_0": errors_0 / shots_0 if shots_0 else None,
            "shots_1": shots_1,
            "errors_1": errors_1,
            "ler_1": errors_1 / shots_1 if shots_1 else None,
        }

    shots_0 = totals_by_state[0]["shots"]
    errors_0 = totals_by_state[0]["errors"]
    shots_1 = totals_by_state[1]["shots"]
    errors_1 = totals_by_state[1]["errors"]
    shots_total = shots_0 + shots_1
    errors_total = errors_0 + errors_1

    overall = {
        "shots_total": shots_total,
        "errors_total": errors_total,
        "ler_total": errors_total / shots_total if shots_total else None,
        "shots_0": shots_0,
        "errors_0": errors_0,
        "ler_0": errors_0 / shots_0 if shots_0 else None,
        "shots_1": shots_1,
        "errors_1": errors_1,
        "ler_1": errors_1 / shots_1 if shots_1 else None,
    }

    return {
        "distance": distance,
        "rounds": rounds,
        "basis": basis,
        "jobs_processed": processed_jobs,
        "num_jobs": len(processed_jobs),
        "num_chains": len(chains_summary),
        "chains": chains_summary,
        "totals": overall,
    }


def format_rate(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate:.4%}"


def plot_chain_ler(summary: Dict, output_path: Path) -> None:
    if plt is None:
        print("matplotlib not available – skipping plot.")
        return

    chains = summary.get("chains", {})
    if not chains:
        print("No chain data found; skipping plot.")
        return

    items = sorted(chains.items())
    labels = [chain_id.replace("_", "\n") for chain_id, _ in items]
    ler0 = [data["ler_0"] if data["ler_0"] is not None else 0.0 for _, data in items]
    ler1 = [data["ler_1"] if data["ler_1"] is not None else 0.0 for _, data in items]

    x = np.arange(len(items))
    width = 0.35
    fig_width = max(6.0, 0.5 * len(items))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(x - width / 2, ler0, width, label="|0> shots")
    ax.bar(x + width / 2, ler1, width, label="|1> shots")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Logical error rate")
    ax.set_title(
        f"MWPM LER – D={summary['distance']}, r={summary['rounds']}, basis={summary['basis']}"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    overall = summary.get("totals", {}).get("ler_total")
    if overall is not None:
        ax.axhline(overall, color="black", linestyle=":", linewidth=1.0)
        ax.text(
            len(items) - 0.5,
            overall,
            f"overall {overall:.2%}",
            ha="right",
            va="bottom",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6),
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Wrote plot to {output_path}")


def run_evaluation(
    distance: int,
    rounds: int,
    basis: str,
    results_root: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    make_plot: bool = True,
    quiet: bool = False,
) -> Dict | None:
    """
    Evaluate a single (distance, rounds, basis) combination and persist outputs.

    Returns the summary dictionary, or ``None`` when no matching shots are found.
    """
    chain_accumulator, totals_by_state, processed_jobs = scan_jobs(
        results_root=results_root,
        distance=distance,
        rounds=rounds,
        basis=basis,
        job_limit=limit,
    )

    summary = summarise_results(
        chain_accumulator,
        totals_by_state,
        distance=distance,
        rounds=rounds,
        basis=basis,
        processed_jobs=processed_jobs,
    )

    totals = summary["totals"]
    if totals["shots_total"] == 0:
        if not quiet:
            print(
                f"No matching shots found for D={distance}, r={rounds}, basis={basis} "
                f"in '{results_root}'."
            )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"mwpm_D{distance}_T{rounds}_{basis}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if not quiet:
        print(f"Wrote summary to {json_path}")

    if make_plot:
        plot_path = output_dir / f"mwpm_D{distance}_T{rounds}_{basis}.png"
        plot_chain_ler(summary, plot_path)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MWPM logical error rates for IBM repetition-code experiments."
    )
    parser.add_argument("-d", "--distance", type=int, required=True, help="Code distance D.")
    parser.add_argument("-t", "--rounds", type=int, required=True, help="Number of syndrome rounds T.")
    parser.add_argument("-b", "--basis", type=str, choices=("X", "Z"), required=True, help="Preparation basis.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results_testing",
        help="Directory holding raw experiment subdirectories (default: ./results_testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_decoder" / "LER_results",
        help="Directory for summary JSON/figures (default: benchmark_decoder/LER_results).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of job directories to process (for quick testing).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib visualisation.")
    args = parser.parse_args()

    if args.distance < 2:
        raise ValueError("Distance must be at least 2 for a repetition code.")
    if args.rounds < 1:
        raise ValueError("Number of rounds must be at least 1.")
    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root '{args.results_root}' does not exist.")

    summary = run_evaluation(
        distance=args.distance,
        rounds=args.rounds,
        basis=args.basis,
        results_root=args.results_root,
        output_dir=args.output_dir,
        limit=args.limit,
        make_plot=not args.no_plot,
    )

    if summary is None:
        return

    totals = summary["totals"]
    print(
        f"Processed {summary['num_jobs']} job(s), {summary['num_chains']} chains "
        f"for D={args.distance}, r={args.rounds}, basis={args.basis}"
    )
    print(
        f"  |0> shots: {totals['shots_0']:,}  LER={format_rate(totals['ler_0'])} "
        f"|  |1> shots: {totals['shots_1']:,}  LER={format_rate(totals['ler_1'])}"
    )
    print(f"  Overall shots: {totals['shots_total']:,}  LER={format_rate(totals['ler_total'])}")


if __name__ == "__main__":
    main()
