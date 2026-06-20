import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from ibm_qec.model.decoder import GeneralConditionedRepCodeDecoder, GeneralRepCodeDecoder
from ibm_qec.device import DEVICE
from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
from ibm_qec.evaluation.metrics import evaluate_decoder


def train_from_preprocessed_data(data_dirs: list, basis_filter=None, model_save_path=None,
                                 log_path=None, model_type="cnn"):
    CONFIG = {"batch_size": 4096, "learning_rate": 5e-3, "epochs": 100,
              "channels": 128, "embedding_dim": 256, "evaluation_interval": 50}

    dataset = MultiJobQECDataset(data_dirs, basis_filter=basis_filter)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              collate_fn=variable_size_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], collate_fn=variable_size_collate_fn)

    # "cnn" -> unconditioned baseline; "film" -> calibration-conditioned (FiLM) decoder
    if model_type not in ("cnn", "film"):
        raise ValueError(f"model_type must be 'cnn' or 'film', got {model_type!r}")

    print(f"> Selected {model_type} based neural network. Training...  ")
    if model_type == "cnn":
        model = GeneralRepCodeDecoder(
            D_max=dataset.max_D, r_max=dataset.max_r,
            system_node_features=None, system_edge_features=None,
            channels=CONFIG['channels'], embedding_dim=CONFIG['embedding_dim']
        ).to(DEVICE)
    else:  # film
        graph_sample = dataset[0]['system_subgraph']
        node_features = graph_sample.num_node_features
        edge_features = graph_sample.num_edge_features if graph_sample.num_edge_features else 0
        model = GeneralConditionedRepCodeDecoder(
            D_max=dataset.max_D, r_max=dataset.max_r,
            system_node_features=node_features, system_edge_features=edge_features,
            channels=CONFIG['channels'], embedding_dim=CONFIG['embedding_dim']
        ).to(DEVICE)

    print("\nCompiling the model with torch.compile()...")
    compiled_model = model
    print("Compilation complete.")

    optimizer = optim.Adam(compiled_model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'] * len(train_loader))
    loss_function = nn.BCELoss()

    scaler = torch.amp.GradScaler()

    best_val_accuracy = -1.0
    history = []

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_train_loss = 0
        for syndromes, system_graph, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            syndromes = syndromes.to(DEVICE, non_blocking=True)
            system_graph = system_graph.to(DEVICE)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_probs = compiled_model(syndromes, system_graph)
            loss = loss_function(pred_probs[:, 0, :labels.shape[1]], labels) # Ensure correct shape for loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} Complete | Avg Train Loss: {avg_train_loss:.6f} | LR: {current_lr:.2e}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "learning_rate": current_lr,
        }

        if epoch % CONFIG['evaluation_interval'] == 0 or epoch == CONFIG['epochs']:
            print(f"\n--- Running LER Evaluation at Epoch {epoch} ---")
            ler_metrics = evaluate_decoder(compiled_model, val_loader)
            print(f"  > Validation LER(|0>): {ler_metrics['ler_0']:.4%}")
            print(f"  > Validation LER(|1>): {ler_metrics['ler_1']:.4%}")
            print(f"  > Overall Validation Accuracy: {ler_metrics['accuracy']:.4%}")
            epoch_record["validation"] = ler_metrics

            if ler_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = ler_metrics['accuracy']
                if model_save_path is not None:
                    save_path = Path(model_save_path)
                else:
                    save_path = Path(data_dirs[0]).parent / f"best_decoder_{Path(data_dirs[0]).name}.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  > New best validation accuracy! Saving model to '{save_path}'...")
                torch.save(model.state_dict(), save_path)

        history.append(epoch_record)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "config": CONFIG,
            "data_dirs": data_dirs,
            "basis_filter": sorted(basis_filter) if basis_filter else None,
            "history": history,
            "best_val_accuracy": best_val_accuracy,
        }
        log_path.write_text(json.dumps(log_payload, indent=2))


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys
    import json  # Import the json library

    parser = argparse.ArgumentParser(
        description="Train a general decoder on pre-processed IBM data directories. "
                    "Automatically discovers and filters job data based on metadata.",
        formatter_class=argparse.RawTextHelpFormatter  # For better help text formatting
    )
    # --- Input Data Path ---
    parser.add_argument(
        "data_path",
        type=str,
        help="The parent directory containing all processed job subdirectories (e.g., 'processed_data/')."
    )
    parser.add_argument(
        "--filter-D",
        type=int,
        nargs='*',
        default=None,
        help="Select only jobs with these code distances (D). \nExample: --filter-D 5 7"
    )
    parser.add_argument(
        "--filter-r",
        type=int,
        nargs='*',  # Can accept zero, one, or more values
        default=None,
        help="Select only jobs with these syndrome rounds (r). \nExample: --filter-r 3"
    )
    parser.add_argument(
        "--filter-basis",
        type=str,
        nargs='*',
        default=None,
        help="Select only jobs that include at least one of the specified bases. \nExample: --filter-basis X Z"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "film"],
        default="cnn",
        help="Decoder variant to train:\n"
             "  cnn  - unconditioned CNN baseline (GeneralRepCodeDecoder)\n"
             "  film - calibration-conditioned FiLM decoder (GeneralConditionedRepCodeDecoder)\n"
             "Default: cnn"
    )
    args = parser.parse_args()


    parent_path = Path(args.data_path)
    if not parent_path.is_dir():
        print(f"Error: The provided data path '{parent_path}' is not a directory.")
        sys.exit(1)

    requested_bases = set(args.filter_basis) if args.filter_basis else None
    print(f"Scanning for job directories inside '{parent_path}'...")

    discovered_job_dirs = []

    for sub_p in parent_path.iterdir():
        if sub_p.is_dir() and (sub_p / "params.json").is_file():  # Use params.json as a reliable job indicator

            passes_filters = True
            try:
                # Load job-level parameters (D/T/basis) produced during preprocessing
                with open(sub_p / "params.json", 'r') as f:
                    job_metadata = json.load(f)

                job_bases = job_metadata.get('bases') or []

                # Check D filter
                if args.filter_D is not None:
                    job_D = job_metadata.get('D')
                    if job_D not in args.filter_D:
                        passes_filters = False

                # Check r (rounds) filter (if the D filter passed)
                if passes_filters and args.filter_r is not None:
                    job_r = job_metadata.get('r', job_metadata.get('T'))
                    if job_r not in args.filter_r:
                        passes_filters = False

                # Check basis filter (if any)
                if passes_filters and requested_bases is not None:
                    if not job_bases:
                        passes_filters = False
                    elif not set(job_bases).intersection(requested_bases):
                        passes_filters = False

            except (IOError, json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"  > Warning: Could not read or parse params.json in '{sub_p.name}'. Skipping. Error: {e}")
                passes_filters = False

            if passes_filters:
                if (sub_p / "shots.pt").is_file() and (sub_p / "metadata.pt").is_file():
                    discovered_job_dirs.append(str(sub_p))
                else:
                    print(
                        f"  > Directory '{sub_p.name}' passed filters but is missing processed files (shots.pt/metadata.pt). Run `python -m ibm_qec.data.prepare` first.")

    if not discovered_job_dirs:
        print("\nError: No valid job directories were found that match the specified filters.")
        sys.exit(1)

    final_job_dirs = sorted(discovered_job_dirs)
    print("\n--- The following job directories have been selected for training: ---")
    for job_dir in final_job_dirs:
        print(f"  - {Path(job_dir).name}")
    print(f"--- Total: {len(final_job_dirs)} director(y/ies) ---")

    train_from_preprocessed_data(final_job_dirs, basis_filter=requested_bases, model_type=args.model)
