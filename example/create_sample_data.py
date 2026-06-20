#!/usr/bin/env python3
"""
Extract a small sample from real IBM quantum hardware data to create
a self-contained example dataset that can be checked into the repo.

This script is NOT needed by end users — the sample data it produces
is already included in example/sample_data/.  Run it only if you want
to regenerate the sample from the full dataset.

Usage:
    python example/create_sample_data.py
"""

import json
import shutil
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR_Z = REPO_ROOT / "processed_data" / "d3j7jg1fk6qs73el9hog"  # D=11, r=11, Z
SOURCE_DIR_X = REPO_ROOT / "processed_data" / "d3j7io1fk6qs73el9ft0"  # D=11, r=11, X
OUTPUT_ROOT = Path(__file__).resolve().parent / "sample_data"

N_SHOTS = 256  # 128 per logical state — enough to demonstrate, small enough for git


def extract_sample(source_dir: Path, output_dir: Path, n_shots: int = N_SHOTS):
    """Extract a balanced sample of shots and the associated metadata."""
    print(f"Loading {source_dir.name} ...")
    shots = torch.load(source_dir / "shots.pt", weights_only=False)
    metadata = torch.load(source_dir / "metadata.pt", weights_only=False)

    # Split by logical state and take equal halves
    shots_0 = [s for s in shots if s["intended_logical_state"] == 0]
    shots_1 = [s for s in shots if s["intended_logical_state"] == 1]
    half = n_shots // 2
    sample = shots_0[:half] + shots_1[:half]
    print(f"  Selected {len(sample)} shots ({half} per logical state) from {len(shots)} total")

    # Only keep subgraph chains that appear in the sample
    used_chains = {s["chain_id_str"] for s in sample}
    trimmed_subgraphs = {
        k: v for k, v in metadata["system_subgraphs_by_chain"].items() if k in used_chains
    }
    trimmed_metadata = {
        "system_subgraphs_by_chain": trimmed_subgraphs,
        "global_system_graph": metadata["global_system_graph"],
        "NODE_FEATURES": metadata["NODE_FEATURES"],
        "OP_FEATURES": metadata["OP_FEATURES"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sample, output_dir / "shots.pt")
    torch.save(trimmed_metadata, output_dir / "metadata.pt")

    # Write params.json
    s0 = sample[0]
    _, T, D_minus_1 = s0["syndrome_block"].shape
    params = {"D": D_minus_1 + 1, "T": T, "bases": [s0["basis"]]}
    (output_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"  Saved to {output_dir}")
    print(f"  params: D={params['D']}, r={params.get('r', params.get('T'))}, basis={params['bases']}")

    sz_shots = (output_dir / "shots.pt").stat().st_size / 1024
    sz_meta = (output_dir / "metadata.pt").stat().st_size / 1024
    print(f"  File sizes: shots.pt={sz_shots:.0f}KB, metadata.pt={sz_meta:.0f}KB")


def main():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for source, label in [(SOURCE_DIR_Z, "job_z"), (SOURCE_DIR_X, "job_x")]:
        if not source.exists():
            print(f"WARNING: Source directory not found: {source}")
            print("  Skipping — you need the full dataset to regenerate samples.")
            continue
        extract_sample(source, OUTPUT_ROOT / label)

    print(f"\nDone. Sample data written to {OUTPUT_ROOT}/")
    total_kb = sum(f.stat().st_size for f in OUTPUT_ROOT.rglob("*") if f.is_file()) / 1024
    print(f"Total size: {total_kb:.0f}KB")


if __name__ == "__main__":
    main()
