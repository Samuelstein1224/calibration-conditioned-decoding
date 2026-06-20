import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
from ibm_qec.evaluation.metrics import evaluate_decoder
from ibm_qec.model.decoder import GeneralConditionedRepCodeDecoder, GeneralRepCodeDecoder
from ibm_qec.device import DEVICE
from torch.utils.data import DataLoader


def _parse_model_directory(path: Path) -> Dict[str, str | int]:
    # Run directories are named r<rounds>_D<distance>_<basis>, e.g. r9_D11_Z.
    # The legacy "T" prefix (T9_D11_Z) is also accepted.
    name = path.name
    if not name.startswith(("r", "T")) or "_D" not in name:
        raise ValueError
    try:
        left, basis = name.rsplit("_", 1)
        r_part, d_part = left.split("_D")
        r_val = int(r_part[1:])
        D_val = int(d_part)
    except (ValueError, IndexError) as exc:
        raise ValueError from exc
    return {"r": r_val, "D": D_val, "basis": basis.upper()}


def _discover_job_dirs(parent_path: Path, target_D: int, target_r: int, basis: str) -> List[str]:
    discovered: List[str] = []
    for subdir in sorted(parent_path.iterdir()):
        if not subdir.is_dir():
            continue
        params_path = subdir / "params.json"
        if not params_path.is_file():
            continue

        try:
            with open(params_path, "r") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        job_D = metadata.get("D")
        job_r = metadata.get("r", metadata.get("T"))
        job_bases = metadata.get("bases") or []

        if job_D != target_D or job_r != target_r:
            continue

        if job_bases and basis not in job_bases:
            continue

        if not (subdir / "shots.pt").is_file() or not (subdir / "metadata.pt").is_file():
            continue

        discovered.append(str(subdir))

    return discovered


def evaluate_model_on_validation(
    model_path: Path,
    data_dirs: List[str],
    basis: str,
    model_config: Dict[str, int] | None,
) -> Dict[str, float]:
    dataset = MultiJobQECDataset(data_dirs, basis_filter=[basis])
    loader = DataLoader(
        dataset,
        batch_size=1024,
        collate_fn=variable_size_collate_fn,
        shuffle=False,
    )

    graph_sample = dataset[0]["system_subgraph"]
    node_features = graph_sample.num_node_features
    edge_features = graph_sample.num_edge_features or 0

    channels = 64
    embedding_dim = 128
    if model_config:
        channels = model_config.get("channels", channels)
        embedding_dim = model_config.get("embedding_dim", embedding_dim)

    model = 'cnn'
    if model == 'film':
        model = GeneralConditionedRepCodeDecoder(
            D_max=dataset.max_D,
            r_max=dataset.max_r,
            system_node_features=node_features,
            system_edge_features=edge_features,
            channels=channels,
            embedding_dim=embedding_dim,
        ).to(DEVICE)
    elif model == 'cnn':
        model = GeneralRepCodeDecoder(
            D_max=dataset.max_D,
            r_max=dataset.max_r,
            system_node_features=None, 
            system_edge_features=None,
            channels=channels,
            embedding_dim=embedding_dim,
        ).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    metrics = evaluate_decoder(model, loader)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models against validation datasets that match T, D, and basis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "models_root",
        type=str,
        help="Directory containing trained model subdirectories (e.g., models).",
    )
    parser.add_argument(
        "validation_root",
        type=str,
        nargs="+",
        help="One or more directories with processed validation job subdirectories (e.g., processed_validation).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Optional path to store aggregated evaluation metrics.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        help="Optional list of run directory names to evaluate (e.g., r7_D9_Z).",
    )
    args = parser.parse_args()

    models_root = Path(args.models_root)
    validation_roots = [Path(root) for root in args.validation_root]

    if not models_root.is_dir():
        print(f"Error: models_root '{models_root}' is not a directory.")
        sys.exit(1)
    for root in validation_roots:
        if not root.is_dir():
            print(f"Error: validation_root '{root}' is not a directory.")
            sys.exit(1)

    aggregated_results = {}
    validation_label = "__".join(root.name for root in validation_roots)
    eval_root = Path("evaluation") / models_root.name / validation_label
    eval_root.mkdir(parents=True, exist_ok=True)
    runs_filter = set(args.runs) if args.runs else None

    for run_dir in sorted(models_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if runs_filter and run_dir.name not in runs_filter:
            continue
        model_path = run_dir / "model.pth"
        log_path = run_dir / "training_log.json"
        if not model_path.is_file():
            continue

        try:
            config = _parse_model_directory(run_dir)
        except ValueError:
            print(f"Skipping '{run_dir.name}': unrecognized directory naming convention.")
            continue

        basis = config["basis"]
        D_val = config["D"]
        r_val = config["r"]

        print("\n====================================================")
        print(f"Evaluating model '{run_dir.name}' (r={r_val}, D={D_val}, basis={basis})")

        job_dirs: List[str] = []
        for root in validation_roots:
            job_dirs.extend(_discover_job_dirs(root, D_val, r_val, basis))
        if not job_dirs:
            print("  > No matching validation datasets found. Skipping.")
            continue

        print("  > Validation datasets:")
        for item in job_dirs:
            print(f"    - {Path(item).name}")

        training_config = None
        if log_path.is_file():
            try:
                log_data = json.loads(log_path.read_text())
                training_config = log_data.get("config") or None
            except json.JSONDecodeError:
                training_config = None

        metrics = evaluate_model_on_validation(model_path, job_dirs, basis, training_config)

        print("  > Results:")
        print(f"    LER(|0>): {metrics['ler_0']:.4%}")
        print(f"    LER(|1>): {metrics['ler_1']:.4%}")
        print(f"    Overall Accuracy: {metrics['accuracy']:.4%}")

        bound_0 = metrics.get('ler_0_bound_low'), metrics.get('ler_0_bound_high')
        bound_1 = metrics.get('ler_1_bound_low'), metrics.get('ler_1_bound_high')
        bound_all = metrics.get('overall_ler_bound_low'), metrics.get('overall_ler_bound_high')

        if all(b is not None for b in bound_0):
            print(f"    LER Bound |0>: [{bound_0[0]:.4%}, {bound_0[1]:.4%}]")
        if all(b is not None for b in bound_1):
            print(f"    LER Bound |1>: [{bound_1[0]:.4%}, {bound_1[1]:.4%}]")
        if all(b is not None for b in bound_all):
            print(f"    LER Bound Overall: [{bound_all[0]:.4%}, {bound_all[1]:.4%}]")

        if 'avg_post_conf_0' in metrics and 'avg_post_conf_1' in metrics:
            print(f"    Avg Post-Correction Confidence |0>: {metrics['avg_post_conf_0']:.4f}")
            print(f"    Avg Post-Correction Confidence |1>: {metrics['avg_post_conf_1']:.4f}")

        per_model_output = eval_root / run_dir.name / "validation_metrics.json"
        per_model_output.parent.mkdir(parents=True, exist_ok=True)
        per_model_output.write_text(json.dumps(metrics, indent=2))

        aggregated_results[run_dir.name] = metrics

    if aggregated_results:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = eval_root / output_path
        existing_results = {}
        if output_path.is_file():
            try:
                existing_results = json.loads(output_path.read_text())
            except json.JSONDecodeError:
                existing_results = {}
        if isinstance(existing_results, dict):
            existing_results.update(aggregated_results)
            output_payload = existing_results
        else:
            output_payload = aggregated_results
        output_path.write_text(json.dumps(output_payload, indent=2))
        print(f"\nAggregated results written to '{output_path}'.")
    else:
        print("\nNo models were evaluated; no output generated.")


if __name__ == "__main__":
    main()
