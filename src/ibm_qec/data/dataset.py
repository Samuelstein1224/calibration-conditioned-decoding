import json
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class MultiJobQECDataset(Dataset):
    """
    A PyTorch Dataset that concatenates shot data from multiple processed job directories.
    Loads a single 'shots.pt' file from each job directory.
    It can optionally drop shots by measurement basis when a filter is provided.
    """

    def __init__(
        self,
        processed_data_dirs: list,
        basis_filter=None,
        distance_filter=None,
        round_filter=None,
    ):
        self.metadata_by_dir = {}
        self.max_D = 0
        self.max_r = 0
        self.loaded_shots = []
        self.basis_filter = {str(b).upper() for b in basis_filter} if basis_filter else None
        self.distance_filter = set(distance_filter) if distance_filter else None
        self.round_filter = set(round_filter) if round_filter else None

        print("--- Loading consolidated shot data from multiple job directories ---")
        if self.basis_filter:
            print(f"  > Applying basis filter: {sorted(self.basis_filter)}")
        if self.distance_filter:
            print(f"  > Applying distance filter: {sorted(self.distance_filter)}")
        if self.round_filter:
            print(f"  > Applying round-count filter: {sorted(self.round_filter)}")
        for data_dir in processed_data_dirs:
            data_path = Path(data_dir)

            params_file = data_path / "params.json"
            if params_file.exists():
                try:
                    with params_file.open("r") as handle:
                        params = json.load(handle)
                    job_bases = params.get("bases")
                    if self.basis_filter and isinstance(job_bases, (list, tuple)) and job_bases:
                        normalized = {str(basis).upper() for basis in job_bases}
                        if normalized.isdisjoint(self.basis_filter):
                            print(
                                f"  > Skipping {data_path.name}: bases {sorted(normalized)} not in requested filter {sorted(self.basis_filter)}."
                            )
                            continue
                    if self.distance_filter is not None:
                        job_D = params.get("D")
                        if job_D is not None and job_D not in self.distance_filter:
                            print(
                                f"  > Skipping {data_path.name}: distance D={job_D} not in {sorted(self.distance_filter)}."
                            )
                            continue
                    if self.round_filter is not None:
                        job_r = params.get("r", params.get("T"))
                        if job_r is not None and job_r not in self.round_filter:
                            print(
                                f"  > Skipping {data_path.name}: rounds r={job_r} not in {sorted(self.round_filter)}."
                            )
                            continue
                except (json.JSONDecodeError, OSError) as exc:
                    warnings.warn(f"Could not parse params.json in '{data_path}': {exc}. Proceeding with per-shot filtering.")

            shots_file = data_path / "shots.pt"
            metadata_file = data_path / "metadata.pt"

            if not shots_file.exists() or not metadata_file.exists():
                warnings.warn(f"Skipping directory '{data_dir}': missing shots.pt or metadata.pt")
                continue

            print(f"  > Loading data from {data_path.name}...")
            metadata_key = str(data_path)
            self.metadata_by_dir[metadata_key] = torch.load(metadata_file, weights_only=False)
            shots_from_this_job = torch.load(shots_file, weights_only=False)
            total_job_shots = len(shots_from_this_job)
            filtered_shots = []
            missing_basis_warned = False

            for shot_data in shots_from_this_job:
                shot_basis = shot_data.get('basis')
                if self.basis_filter:
                    if shot_basis is None:
                        if not missing_basis_warned:
                            warnings.warn(
                                f"Skipping shots without basis metadata in '{data_path.name}' because a basis filter was requested."
                            )
                            missing_basis_warned = True
                        continue
                    normalized_basis = str(shot_basis).upper()
                    if normalized_basis not in self.basis_filter:
                        continue

                if self.distance_filter:
                    shot_D = len(shot_data['target_correction'])
                    if shot_D not in self.distance_filter:
                        continue

                if self.round_filter:
                    _, r, _ = shot_data['syndrome_block'].shape
                    if r not in self.round_filter:
                        continue

                shot_data['parent_dir_str'] = metadata_key

                _, r, D_minus_1 = shot_data['syndrome_block'].shape
                self.max_r = max(self.max_r, r)
                self.max_D = max(self.max_D, D_minus_1 + 1)

                filtered_shots.append(shot_data)

            if self.basis_filter:
                if total_job_shots:
                    print(f"    > Retained {len(filtered_shots)} of {total_job_shots} shots after basis filtering.")
                if not filtered_shots:
                    print(f"    > No shots matched the requested basis filter in '{data_path.name}'. Skipping this job.")
                    self.metadata_by_dir.pop(metadata_key, None)
                    continue

            self.loaded_shots.extend(filtered_shots)

        if not self.loaded_shots:
            if self.basis_filter:
                raise FileNotFoundError(
                    "No shot data matched the requested basis filter across the provided directories."
                )
            raise FileNotFoundError("No valid shot data was found across any of the provided directories.")

        print(f"\n--- Loading complete. Found {len(self.loaded_shots):,} total shots. ---")
        if self.loaded_shots:
            sample_dir = Path(self.loaded_shots[0]['parent_dir_str'])
            print(f"--- Example job directory: {sample_dir} ---")
        print(f"--- Inferred Max D={self.max_D}, Max r={self.max_r} across all jobs. ---")

    def __len__(self):
        return len(self.loaded_shots)

    def __getitem__(self, idx):
        shot_data = self.loaded_shots[idx]

        parent_dir_str = shot_data['parent_dir_str']
        metadata = self.metadata_by_dir[parent_dir_str]

        chain_id_str = shot_data['chain_id_str']
        try:
            system_subgraph = metadata['system_subgraphs_by_chain'][chain_id_str]
        except KeyError:
            print(f"FATAL: Could not find pre-computed subgraph for chain '{chain_id_str}' in job '{parent_dir_str}'.")
            print("Please re-run `python -m ibm_qec.data.prepare` with the latest version of the script.")
            raise

        return {
            "syndrome_block": shot_data['syndrome_block'],
            "target_correction": shot_data['target_correction'],
            "system_subgraph": system_subgraph,
            "initial_logical_state": shot_data['intended_logical_state'],
            "final_measured_data": shot_data['final_measurement']
        }


def variable_size_collate_fn(batch_list):
    """
    Custom collate function that pads syndromes and labels to the max size in the batch.
    """
    from torch.nn.utils.rnn import pad_sequence

    max_t = max([item['syndrome_block'].shape[1] for item in batch_list])
    max_d_minus_1 = max([item['syndrome_block'].shape[2] for item in batch_list])
    max_d = max([len(item['target_correction']) for item in batch_list])

    syndromes = []
    for item in batch_list:
        s = item['syndrome_block']
        padding = (0, max_d_minus_1 - s.shape[2], 0, max_t - s.shape[1])
        syndromes.append(torch.nn.functional.pad(s, padding, "constant", 0))
    syndromes = torch.cat(syndromes, dim=0).unsqueeze(1)

    labels = [item['target_correction'] for item in batch_list]
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    final_data = [item['final_measured_data'] for item in batch_list]
    final_data = pad_sequence(final_data, batch_first=True, padding_value=0)

    initial_states = torch.tensor([item['initial_logical_state'] for item in batch_list], dtype=torch.long)
    graph_list = [item['system_subgraph'] for item in batch_list]
    system_graph_batch = Batch.from_data_list(graph_list)

    return syndromes, system_graph_batch, labels, initial_states, final_data
