#!/usr/bin/env python3
"""Consolidate raw IBM experiment jobs into the released ``experiment_data/`` layout.

Raw jobs are reorganized into a clean, anonymized tree grouped by device and code
configuration. IBM Runtime job identifiers are dropped (snapshots are renamed
``job_1``, ``job_2`` ... within each group) so no internal IDs are released.

Output layout (documented in ``experiment_data/README.md``)::

    experiment_data/
    ├── index.csv
    └── <backend>/                         # ibm_kingston, ibm_fez, ibm_pittsburgh
        └── d<D>_r<R>/                      # code distance D, syndrome rounds R
            └── job_<n>/
                ├── info.json              # backend, d, rounds, basis, shots, date, n_chains
                ├── calibration.json       # device calibration snapshot (backend_properties)
                ├── circuit_state0.qasm    # transpiled circuit, logical |0>
                ├── circuit_state1.qasm    # transpiled circuit, logical |1>
                └── bitstrings.json        # raw per-shot measurement records (per chain)

Each job runs several repetition-code chains in parallel; every chain's per-shot data
and qubit layout are preserved in ``bitstrings.json``.

Usage
-----
    python scripts/consolidate_data.py \
        --sources results results_testing \
                  results_validation_ibm_fez \
                  results_validation_ibm_kingston \
                  results_validation_ibm_pittsburgh \
        --out experiment_data

Pass ``--index-only`` to (re)build just ``index.csv`` without writing the (large) job
directories.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
from pathlib import Path


def _load_json(path: Path):
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_job(job_dir: Path):
    """Return (config, metadata, results) for a raw job dir, or None if not a job."""
    metadata = _load_json(job_dir / "metadata.json")
    results = _load_json(job_dir / "results.json")
    if metadata is None or results is None or not isinstance(results, list) or not results:
        return None

    ds, rs, bases, states = set(), set(), set(), []
    for entry in results:
        m = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        ds.add(m.get("D"))
        rs.add(m.get("n_syndrome_rounds"))
        bases.add(m.get("basis"))
        states.append(m.get("logical_state"))
    ds.discard(None); rs.discard(None); bases.discard(None)

    cregs = results[0].get("per_shot_cregs", {}) if isinstance(results[0], dict) else {}
    # Recover the code distance from the data-register width (data qubits = D) when
    # some early jobs omit the D field in their metadata.
    if not ds:
        widths = {len(v[0]) for k, v in cregs.items()
                  if k.startswith("c_data") and isinstance(v, list) and v
                  and isinstance(v[0], list)}
        if len(widths) == 1:
            ds = widths
    if len(ds) != 1 or len(rs) != 1 or len(bases) != 1:
        return None  # not a clean single-(d,r,basis) job

    n_chains = sum(1 for k in cregs if k.startswith("c_data"))
    config = {
        "backend_name": metadata.get("backend_name"),
        "d": next(iter(ds)),
        "rounds": next(iter(rs)),
        "basis": next(iter(bases)),
        "logical_states": sorted(s for s in states if s is not None),
        "shots": metadata.get("shots"),
        "n_chains": n_chains,
    }
    return config, metadata, results


def find_jobs(root: Path):
    for results_path in root.rglob("results.json"):
        job_dir = results_path.parent
        if (job_dir / "metadata.json").exists():
            yield job_dir


def write_job(out_dir: Path, job_dir: Path, config, metadata, results) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # info.json — experiment parameters, no identifiers
    (out_dir / "info.json").write_text(json.dumps({
        "backend": config["backend_name"],
        "d": config["d"],
        "rounds": config["rounds"],
        "basis": config["basis"],
        "logical_states": config["logical_states"],
        "shots": config["shots"],
        "n_chains": config["n_chains"],
    }, indent=2))
    # calibration.json — the device calibration snapshot
    (out_dir / "calibration.json").write_text(
        json.dumps(metadata.get("backend_properties", {}), indent=2))
    # bitstrings.json — raw per-shot measurement records (already free of job ids)
    shutil.copyfile(job_dir / "results.json", out_dir / "bitstrings.json")
    # circuits — one transpiled qasm per logical state, with barrier statements removed
    for qasm in sorted(job_dir.glob("transpiled_*.qasm")):
        state = "0"
        for tok in qasm.stem.split("_"):
            if tok.startswith("state"):
                state = tok.replace("state", "")
        lines = qasm.read_text().splitlines(keepends=True)
        kept = [ln for ln in lines if not ln.lstrip().startswith("barrier")]
        (out_dir / f"circuit_state{state}.qasm").write_text("".join(kept))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True,
                   help="raw collection directories to consolidate")
    p.add_argument("--out", default="experiment_data",
                   help="output directory (default: experiment_data)")
    p.add_argument("--index-only", action="store_true",
                   help="rebuild index.csv only; do not write job directories")
    p.add_argument("--max-d", type=int, default=None,
                   help="only include snapshots with code distance d <= MAX_D")
    p.add_argument("--max-r", type=int, default=None,
                   help="only include snapshots with syndrome rounds <= MAX_R")
    p.add_argument("--archive", metavar="DIR",
                   help="after building, write per-device <backend>.tar.gz archives plus "
                        "index.csv and README.md into DIR (for Zenodo, which caps records "
                        "at 100 files)")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Pass 1: collect every job keyed by (backend, d, rounds) for deterministic numbering.
    groups: dict[tuple, list] = {}
    for source in args.sources:
        root = Path(source)
        if not root.exists():
            print(f"warning: source not found, skipping: {source}", file=sys.stderr)
            continue
        n = 0
        for job_dir in find_jobs(root):
            parsed = read_job(job_dir)
            if parsed is None:
                continue
            config, metadata, results = parsed
            if args.max_d is not None and config["d"] > args.max_d:
                continue
            if args.max_r is not None and config["rounds"] > args.max_r:
                continue
            key = (config["backend_name"], config["d"], config["rounds"])
            # stable, ID-independent ordering for job_<n> numbering
            sort_key = (config["basis"], job_dir.name)
            groups.setdefault(key, []).append((sort_key, job_dir, config, metadata, results))
            n += 1
        print(f"  {source}: {n} jobs")

    # Pass 2: assign job_N and write.
    rows = []
    for (backend, d, rounds), items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        items.sort(key=lambda t: t[0])
        dr_dir = out / str(backend) / f"d{d}_r{rounds}"
        for i, (_sk, job_dir, config, metadata, results) in enumerate(items, start=1):
            rel = f"{backend}/d{d}_r{rounds}/job_{i}"
            if not args.index_only:
                write_job(dr_dir / f"job_{i}", job_dir, config, metadata, results)
            rows.append({
                "path": rel,
                "backend": backend,
                "d": d,
                "rounds": rounds,
                "basis": config["basis"],
                "logical_states": ";".join(str(s) for s in config["logical_states"]),
                "n_chains": config["n_chains"],
                "shots": config["shots"],
            })

    rows.sort(key=lambda r: r["path"])
    index_path = out / "index.csv"
    fields = ["path", "backend", "d", "rounds", "basis",
              "logical_states", "n_chains", "shots"]
    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Indexed' if args.index_only else 'Consolidated'} {len(rows)} jobs "
          f"across {len(groups)} (device, d, r) groups")
    print(f"Wrote index: {index_path}")

    if args.archive and not args.index_only:
        adir = Path(args.archive)
        adir.mkdir(parents=True, exist_ok=True)
        backends = sorted({backend for backend, _, _ in groups})
        for backend in backends:
            tar_path = adir / f"{backend}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(out / backend, arcname=backend)
            print(f"  archived {backend} -> {tar_path}")
        for name in ("index.csv", "README.md"):
            if (out / name).exists():
                shutil.copyfile(out / name, adir / name)
        print(f"Wrote {len(backends)} device archives + index/README to {adir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
