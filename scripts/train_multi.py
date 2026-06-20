import argparse
import json
import sys
from collections.abc import Iterable as CollectionsIterable
from pathlib import Path
from typing import List

# Import the training function from the sibling train script
import importlib.util
_spec = importlib.util.spec_from_file_location("_train", Path(__file__).with_name("train.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
train_from_preprocessed_data = _mod.train_from_preprocessed_data


DEFAULT_EXPERIMENTS = [
    {"r": r, "D": D, "basis": basis}
    for D in [9]
    for r in [7]
    for basis in ["Z"]
]



def _load_experiments_from_file(path: Path) -> List[dict]:
    if not path.is_file():
        raise ValueError(f"Experiments file '{path}' does not exist.")
    try:
        content = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Experiments file '{path}' is not valid JSON: {exc}") from exc
    return _normalize_experiments(content)


def _normalize_experiments(raw: CollectionsIterable) -> List[dict]:
    if not isinstance(raw, CollectionsIterable) or isinstance(raw, (str, bytes)):
        raise ValueError("Experiments specification must be a list of experiments.")

    normalized: List[dict] = []
    for idx, entry in enumerate(raw):
        if isinstance(entry, dict):
            try:
                r = entry["r"] if "r" in entry else entry["T"]  # accept legacy "T" key
                D = entry["D"]
                basis = entry["basis"]
            except KeyError as exc:
                raise ValueError(f"Experiment {idx} is missing key {exc.args[0]!r}.") from exc
        elif isinstance(entry, (list, tuple)):
            if len(entry) != 3:
                raise ValueError(f"Experiment {idx} must have exactly three elements (r, D, basis).")
            r, D, basis = entry
        else:
            raise ValueError("Each experiment must be a list or dict with r, D, basis.")

        if not isinstance(r, int) or not isinstance(D, int):
            raise ValueError(f"Experiment {idx} has non-integer r or D values: {r}, {D}.")
        if not isinstance(basis, str) or not basis:
            raise ValueError(f"Experiment {idx} must specify a non-empty basis string.")

        normalized.append({"r": r, "D": D, "basis": basis})

    if not normalized:
        raise ValueError("No experiments were provided.")

    return normalized


def _discover_job_dirs(parent_path: Path, target_D: int, target_r: int, basis: str) -> List[str]:
    discovered: List[str] = []
    for subdir in sorted(parent_path.iterdir()):
        if not subdir.is_dir():
            continue
        if not (subdir / "params.json").is_file():
            continue

        try:
            with open(subdir / "params.json", "r") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  > Skipping '{subdir.name}' due to unreadable params.json: {exc}")
            continue

        job_D = metadata.get("D")
        job_r = metadata.get("r", metadata.get("T"))
        job_bases = metadata.get("bases") or []

        if job_D != target_D or job_r != target_r:
            continue

        if job_bases and basis not in job_bases:
            continue

        if not (subdir / "shots.pt").is_file() or not (subdir / "metadata.pt").is_file():
            print(f"  > Directory '{subdir.name}' missing shots.pt/metadata.pt. Skipping.")
            continue

        discovered.append(str(subdir))

    return discovered


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple IBM QEC decoders over a grid of (r, D, basis) configurations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "data_path",
        type=str,
        nargs="+",
        help="One or more parent directories containing processed job subdirectories (e.g., 'processed_data/').",
    )

    parser.add_argument(
        "--experiments-file",
        type=str,
        help="Optional JSON file overriding the built-in experiments list.",
    )

    parser.add_argument(
        "--models-root",
        type=str,
        default="models",
        help="Directory where trained models will be stored (default: models).",
    )

    args = parser.parse_args()

    parent_paths = [Path(p) for p in args.data_path]
    for parent_path in parent_paths:
        if not parent_path.is_dir():
            print(f"Error: The provided data path '{parent_path}' is not a directory.")
            sys.exit(1)

    if args.experiments_file:
        experiments = _load_experiments_from_file(Path(args.experiments_file))
    else:
        experiments = _normalize_experiments(DEFAULT_EXPERIMENTS)

    models_root = Path(args.models_root)

    for experiment in experiments:
        r = experiment["r"]
        D = experiment["D"]
        basis = experiment["basis"].upper()

        print("\n====================================================")
        print(f"Preparing training run for r={r}, D={D}, basis={basis}")

        job_dirs: List[str] = []
        for parent_path in parent_paths:
            job_dirs.extend(_discover_job_dirs(parent_path, D, r, basis))
        job_dirs = sorted(set(job_dirs))

        if not job_dirs:
            print("  > No job directories found for this configuration. Skipping.")
            continue

        print("  > Selected job directories:")
        for job_dir in job_dirs:
            print(f"    - {Path(job_dir).name}")

        save_dir = models_root / f"r{r}_D{D}_{basis}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "model.pth"
        log_path = save_dir / "training_log.json"
        print(f"  > Model artifacts will be saved to '{save_path}'")
        print(f"  > Training log will be written to '{log_path}'")

        try:
            train_from_preprocessed_data(
                job_dirs,
                basis_filter=[basis],
                model_save_path=save_path,
                log_path=log_path,
            )
        except FileNotFoundError as exc:
            print(f"  > Training aborted for this configuration: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  > Unexpected error during training: {exc}")


if __name__ == "__main__":
    main()
