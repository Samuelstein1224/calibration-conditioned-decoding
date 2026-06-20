#!/usr/bin/env python3
"""
End-to-end worked example: train and evaluate an ML decoder on real
IBM quantum hardware data (D=11, r=11 repetition code).

This uses a small sample (256 shots) already bundled in example/sample_data/
so no large datasets need to be downloaded.

The workflow mirrors the full pipeline:
  1. Load preprocessed experimental data
  2. Inspect the data (syndrome blocks, hardware graphs, calibration features)
  3. Train a Conv2D decoder from scratch
  4. Evaluate the trained decoder (logical error rate, confidence)
  5. (Optional) Load a pretrained model from models/ if available

Usage:
    pip install -e .          # install ibm_qec package (once)
    python example/run_example.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── ibm_qec package imports ────────────────────────────────────────
from ibm_qec.model.decoder import GeneralRepCodeDecoder, GeneralConditionedRepCodeDecoder
from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
from ibm_qec.evaluation.metrics import evaluate_decoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent
SAMPLE_DATA = EXAMPLE_DIR / "sample_data"


# ====================================================================
# Step 1 — Load and inspect the data
# ====================================================================
def step1_load_and_inspect():
    print("=" * 64)
    print("STEP 1: Load and inspect sample data")
    print("=" * 64)

    # Each job directory contains shots.pt, metadata.pt, and params.json
    # produced by the data preparation pipeline from raw IBM Quantum results.
    for job_dir in sorted(SAMPLE_DATA.iterdir()):
        if not job_dir.is_dir():
            continue
        params = json.loads((job_dir / "params.json").read_text())
        shots = torch.load(job_dir / "shots.pt", weights_only=False)
        metadata = torch.load(job_dir / "metadata.pt", weights_only=False)

        print(f"\n  Job: {job_dir.name}")
        print(f"  Code distance D={params['D']}, syndrome rounds r={params.get('r', params.get('T'))}, basis={params['bases']}")
        print(f"  Number of shots: {len(shots)}")

        # Each shot is a dict with syndrome measurements and correction labels
        s = shots[0]
        print(f"  Shot keys: {list(s.keys())}")
        print(f"  syndrome_block shape: {s['syndrome_block'].shape}  (1, r={params.get('r', params.get('T'))}, D-1={params['D']-1})")
        print(f"  target_correction shape: {s['target_correction'].shape}  (D={params['D']})")
        print(f"  intended_logical_state: {s['intended_logical_state']}")

        # The metadata contains per-chain hardware graphs with calibration data
        chains = metadata["system_subgraphs_by_chain"]
        chain_id = list(chains.keys())[0]
        graph = chains[chain_id]
        print(f"\n  Hardware graph for chain [{chain_id[:40]}...]:")
        print(f"    Nodes: {graph.x.shape[0]} qubits, {graph.x.shape[1]} features per node")
        print(f"    Node features: {metadata['NODE_FEATURES']}")
        print(f"    Edges: {graph.edge_index.shape[1]} connections")
        print(f"    Edge features: {graph.edge_attr.shape}")
        print(f"    Sample node features (first 3 qubits):")
        for i in range(min(3, graph.x.shape[0])):
            feats = ", ".join(f"{metadata['NODE_FEATURES'][j]}={graph.x[i,j]:.4f}" for j in range(4))
            print(f"      qubit {i}: {feats}")


# ====================================================================
# Step 2 — Train a decoder from scratch
# ====================================================================
def step2_train(basis="Z", epochs=30):
    print("\n" + "=" * 64)
    print(f"STEP 2: Train a decoder on basis={basis} sample data")
    print("=" * 64)

    # Discover job directories matching the requested basis
    job_dirs = []
    for job_dir in sorted(SAMPLE_DATA.iterdir()):
        if not job_dir.is_dir():
            continue
        params = json.loads((job_dir / "params.json").read_text())
        if basis in params.get("bases", []):
            job_dirs.append(str(job_dir))

    if not job_dirs:
        print(f"  No sample data found for basis={basis}")
        return None

    # Create dataset — this is the same class used in full training
    dataset = MultiJobQECDataset(job_dirs, basis_filter=[basis])

    # Split into train/val (70/30)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True,
        collate_fn=variable_size_collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=64,
        collate_fn=variable_size_collate_fn,
    )

    # Build the model — using smaller channels for this demo
    # Full training uses channels=128, but channels=32 trains faster
    channels = 32
    model = GeneralRepCodeDecoder(
        D_max=dataset.max_D,
        r_max=dataset.max_r,
        system_node_features=None,
        system_edge_features=None,
        channels=channels,
        embedding_dim=64,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: GeneralRepCodeDecoder")
    print(f"  D_max={dataset.max_D}, r_max={dataset.max_r}, channels={channels}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Device: {DEVICE}")
    print(f"  Training: {train_size} shots, Validation: {val_size} shots")
    print(f"  Epochs: {epochs}")
    print()

    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    loss_fn = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for syndromes, system_graph, labels, _, _ in train_loader:
            syndromes = syndromes.to(DEVICE)
            system_graph = system_graph.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            pred = model(syndromes, system_graph)
            loss = loss_fn(pred[:, 0, :labels.shape[1]], labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # Save the trained model
    save_path = EXAMPLE_DIR / "example_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n  Model saved to {save_path}")

    return model, val_loader, dataset


# ====================================================================
# Step 3 — Evaluate the decoder
# ====================================================================
def step3_evaluate(model, val_loader):
    print("\n" + "=" * 64)
    print("STEP 3: Evaluate the decoder")
    print("=" * 64)

    metrics = evaluate_decoder(model, val_loader)

    print(f"\n  Logical Error Rate |0>: {metrics['ler_0']:.4%}  ({metrics['logical_errors_0']}/{metrics['shots_0']} errors)")
    print(f"  Logical Error Rate |1>: {metrics['ler_1']:.4%}  ({metrics['logical_errors_1']}/{metrics['shots_1']} errors)")
    print(f"  Overall Accuracy:       {metrics['accuracy']:.4%}")
    print()
    print(f"  Pre-correction confidence  |0>: {metrics['avg_pre_conf_0']:.4f}")
    print(f"  Post-correction confidence |0>: {metrics['avg_post_conf_0']:.4f}")
    print(f"  Pre-correction confidence  |1>: {metrics['avg_pre_conf_1']:.4f}")
    print(f"  Post-correction confidence |1>: {metrics['avg_post_conf_1']:.4f}")

    if metrics.get("ler_1_bound_low") is not None:
        print(f"\n  LER confidence intervals (binomial likelihood):")
        print(f"    |0>: [{metrics['ler_0_bound_low']:.4%}, {metrics['ler_0_bound_high']:.4%}]")
        print(f"    |1>: [{metrics['ler_1_bound_low']:.4%}, {metrics['ler_1_bound_high']:.4%}]")

    return metrics


# ====================================================================
# Step 4 — (Optional) Load a pretrained model
# ====================================================================
def step4_pretrained(basis="Z"):
    print("\n" + "=" * 64)
    print("STEP 4: Load a pretrained r11_D11 model (optional)")
    print("=" * 64)

    model_dir = REPO_ROOT / "models" / f"r11_D11_{basis}"
    model_path = model_dir / "model.pth"

    if not model_path.exists():
        print(f"\n  Pretrained model not found at {model_dir}/")
        print("  This is expected — model checkpoints are not included in the repo")
        print("  due to size (~13MB each). The training step above shows how to")
        print("  create one from scratch.")
        return

    # Read training config to match architecture
    config = {"channels": 128, "embedding_dim": 256}
    log_path = model_dir / "training_log.json"
    if log_path.exists():
        log_data = json.loads(log_path.read_text())
        config.update(log_data.get("config", {}))
        print(f"\n  Training config: {config}")

    # Load the sample data for evaluation
    job_dirs = []
    for job_dir in sorted(SAMPLE_DATA.iterdir()):
        if not job_dir.is_dir():
            continue
        params = json.loads((job_dir / "params.json").read_text())
        if basis in params.get("bases", []):
            job_dirs.append(str(job_dir))

    dataset = MultiJobQECDataset(job_dirs, basis_filter=[basis])
    loader = DataLoader(
        dataset, batch_size=256, collate_fn=variable_size_collate_fn,
    )

    # Auto-detect model type from state dict keys
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    has_film = any(k.startswith("system_encoder.") for k in state_dict)

    graph_sample = dataset[0]["system_subgraph"]
    node_features = graph_sample.num_node_features
    edge_features = graph_sample.num_edge_features or 0

    if has_film:
        print(f"  Detected FiLM-conditioned model (GeneralConditionedRepCodeDecoder)")
        model = GeneralConditionedRepCodeDecoder(
            D_max=dataset.max_D,
            r_max=dataset.max_r,
            system_node_features=node_features,
            system_edge_features=edge_features,
            channels=config["channels"],
            embedding_dim=config["embedding_dim"],
        ).to(DEVICE)
    else:
        print(f"  Detected unconditioned model (GeneralRepCodeDecoder)")
        model = GeneralRepCodeDecoder(
            D_max=dataset.max_D,
            r_max=dataset.max_r,
            system_node_features=None,
            system_edge_features=None,
            channels=config["channels"],
            embedding_dim=config["embedding_dim"],
        ).to(DEVICE)

    model.load_state_dict(state_dict)
    print(f"  Loaded pretrained model from {model_path}")

    metrics = evaluate_decoder(model, loader)
    print(f"\n  Pretrained model results on sample data:")
    print(f"  LER |0>: {metrics['ler_0']:.4%}  LER |1>: {metrics['ler_1']:.4%}  Accuracy: {metrics['accuracy']:.4%}")

    # Compare with published validation metrics
    val_metrics_path = model_dir / "validation_metrics.json"
    if val_metrics_path.exists():
        val = json.loads(val_metrics_path.read_text())
        print(f"\n  Published validation metrics (full dataset, {val['shots_total']} shots):")
        print(f"  LER |0>: {val['ler_0']:.4%}  LER |1>: {val['ler_1']:.4%}  Accuracy: {val['accuracy']:.4%}")


# ====================================================================
# Main
# ====================================================================
def main():
    print("IBM QEC — End-to-End Worked Example")
    print("Decoder for D=11, r=11 repetition code on real IBM hardware data\n")

    if not SAMPLE_DATA.exists():
        print(f"ERROR: Sample data not found at {SAMPLE_DATA}/")
        print("Run 'python example/create_sample_data.py' first (requires full dataset).")
        sys.exit(1)

    step1_load_and_inspect()

    result = step2_train(basis="Z", epochs=30)
    if result is None:
        sys.exit(1)
    model, val_loader, dataset = result

    step3_evaluate(model, val_loader)
    step4_pretrained(basis="Z")

    print("\n" + "=" * 64)
    print("Done. See example/README.md for more details.")
    print("=" * 64)


if __name__ == "__main__":
    main()
