import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import warnings
from torch_geometric.utils import subgraph
import numpy as np
import torch
from qiskit import qasm3
from qiskit.converters import circuit_to_dag
from torch_geometric.data import Data
from tqdm import tqdm

# --- Define Feature Names as Global Constants ---
NODE_FEATURES = ['T1_norm', 'T2_norm', 'readout_error', 'sx_error']
OP_FEATURES = ['IDLE', 'SINGLE_Q_GATE', 'CZ_QUBIT', 'MEASURE']
EDGE_FEATURES = ['cz_error']


def create_rich_system_graph(backend_properties: dict) -> Data:
    """Creates a PyG Data object for the hardware with rich node and edge features."""
    print("Building rich global system graph from backend properties...")
    if not backend_properties:
        warnings.warn("Backend properties are missing. The system graph will be empty.")
        return Data()

    qubits_data = backend_properties.get('qubits', [])
    gates_data = backend_properties.get('gates', [])
    num_qubits = len(qubits_data)

    node_features = []
    for i in range(num_qubits):
        q_props = {p['name']: p['value'] for p in qubits_data[i]}
        sx_gate = next((g for g in gates_data if g.get('gate') == 'sx' and g.get('qubits') == [i]), None)
        sx_error = 0.0
        if sx_gate:
            sx_error_param = next((p for p in sx_gate.get('parameters', []) if p['name'] == 'gate_error'), None)
            if sx_error_param:
                sx_error = sx_error_param['value']

        features = [
            q_props.get('T1', 0) / 1e-6, q_props.get('T2', 0) / 1e-6,
            q_props.get('readout_error', 0), sx_error
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)
    x[torch.isinf(x) | torch.isnan(x)] = 0
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std[std == 0] = 1.0
    x = (x - mean) / std

    try:
        coupling_map = eval(str(backend_properties.get('coupling_map', [])))
    except Exception:
        coupling_map = []
        warnings.warn("Could not parse coupling_map.")

    edge_index, edge_features = [], []
    two_qubit_gate_name = 'cz'

    gate_error_lookup = {}
    for gate in gates_data:
        if len(gate.get('qubits', [])) == 2:
            key = tuple(sorted(gate['qubits']))
            error_param = next((p for p in gate.get('parameters', []) if p['name'] == 'gate_error'), None)
            if error_param:
                gate_error_lookup[(gate.get('gate'), key)] = error_param['value']

    for edge in coupling_map:
        key = (two_qubit_gate_name, tuple(sorted(edge)))
        cz_error = gate_error_lookup.get(key, 0.0)
        edge_index.append(edge)
        edge_features.append([cz_error])

    if not edge_index:
        return Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 1)))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    edge_index_rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def create_circuit_temporal_map(qasm_file_path: Path, chain_qubits: list, D: int, r: int) -> torch.Tensor:
    """
    Creates a dense feature map from a QASM file using the QASM3 parser and a robust DAG-based approach.
    """
    qubit_map = {global_idx: local_idx for local_idx, global_idx in enumerate(chain_qubits)}

    try:
        # --- MODIFIED: Use qasm3.load with the file path ---
        qc = qasm3.load(qasm_file_path)
        dag = circuit_to_dag(qc)
    except Exception as e:
        warnings.warn(f"Qiskit failed to parse QASM file '{qasm_file_path}': {e}. Returning empty circuit map.")
        return torch.empty(0)

    # Create a lookup map from Qubit object to its integer index
    qubit_object_to_int_map = {qubit: i for i, qubit in enumerate(qc.qubits)}

    layers_by_barrier = dag.serial_layers()
    round_dag = None
    for layer in layers_by_barrier:
        if any(node.op.name == 'measure' for node in layer['graph'].op_nodes()):
            round_dag = layer['graph']
            break

    if round_dag is None:
        warnings.warn("Could not identify a measurement round in QASM DAG.")
        return torch.empty(0)

    parallel_layers_in_round = list(round_dag.layers())
    num_timesteps = len(parallel_layers_in_round)
    if num_timesteps == 0: return torch.empty(0)

    num_qubits_in_chain = len(chain_qubits)
    num_op_features = len(OP_FEATURES)
    feature_map = torch.zeros(num_timesteps, num_qubits_in_chain, num_op_features, dtype=torch.float)
    feature_map[:, :, OP_FEATURES.index('IDLE')] = 1.0

    for t, layer in enumerate(parallel_layers_in_round):
        for node in layer['graph'].op_nodes():
            qubits_involved = [qubit_object_to_int_map[q_obj] for q_obj in node.qargs]
            print(qubits_involved)
            for q_idx in qubits_involved:
                if q_idx in qubit_map:
                    local_q_idx = qubit_map[q_idx]
                    feature_map[t, local_q_idx, OP_FEATURES.index('IDLE')] = 0.0
                    op_name = node.op.name
                    if op_name == 'measure':
                        feature_map[t, local_q_idx, OP_FEATURES.index('MEASURE')] = 1.0
                    elif op_name in ['cz', 'ecr']:
                        local_q0 = qubit_map[qubits_involved[0]]
                        local_q1 = qubit_map[qubits_involved[1]]
                        feature_map[t, local_q0, OP_FEATURES.index('CZ_QUBIT')] = 1.0
                        feature_map[t, local_q1, OP_FEATURES.index('CZ_QUBIT')] = 1.0
                    elif len(node.qargs) == 1:
                        feature_map[t, local_q_idx, OP_FEATURES.index('SINGLE_Q_GATE')] = 1.0

    final_map = feature_map.permute(2, 1, 0)
    return final_map.unsqueeze(0)


def process_and_save_data(job_dir: str, out_dir: str):
    """
    Processes a raw job directory to create a self-contained, ML-ready data directory.

    This function follows a one-shot "Extract, Transform, Save" pipeline:
    1.  Loads all raw data from the job directory.
    2.  Pre-computes hardware graphs, subgraphs, and circuit maps.
    3.  Processes every experimental shot into a tensor format.
    4.  Saves three final artifacts:
        - params.json:  Small JSON with D, r for fast filtering.
        - metadata.pt:  Large file with all graph/map tensor data.
        - shots.pt:     Large file with the list of all processed shots.
    """
    # --- 1. SETUP AND LOAD ALL RAW DATA ---
    job_path = Path(job_dir)
    job_id = job_path.name
    processed_job_path = Path(out_dir) / job_id
    processed_job_path.mkdir(parents=True, exist_ok=True)

    manifest_json = []
    manifest_metadata_list = []
    manifest_lookup_by_circuit = {}
    manifest_lookup_by_index = {}
    unique_bases = []

    print(f"--- Processing Job ID: {job_id} ---")
    print(f"  > Output will be saved in: '{processed_job_path}'")

    try:
        with open(job_path / "metadata.json", 'r') as f:
            metadata_json = json.load(f)
        with open(job_path / 'manifest.json', 'r') as f:
            manifest_json = json.load(f)
            manifest_metadata_list = [entry.get('metadata', {}) for entry in manifest_json]
            manifest_lookup_by_circuit = {
                entry.get('circuit_name'): entry.get('metadata', {})
                for entry in manifest_json
                if entry.get('circuit_name')
            }
            manifest_lookup_by_index = {
                entry.get('index'): entry.get('metadata', {})
                for entry in manifest_json
                if entry.get('index') is not None
            }
            unique_bases = sorted({
                meta.get('basis')
                for meta in manifest_metadata_list
                if meta.get('basis') is not None
            })
        with open(job_path / "results.json", 'r') as f:
            results_data = json.load(f)
        qasm_file_path = next(job_path.glob("transpiled_*.qasm"))
    except (IOError, StopIteration, json.JSONDecodeError) as e:
        warnings.warn(f"Fatal: Could not load essential raw data for job '{job_id}'. Skipping. Error: {e}")
        return

    # --- 2. EXTRACT PRIMARY INFO & SAVE PARAMS.JSON ---

    # Extract all unique physical qubit chains from the results. This is done once.
    all_chains = set()
    for res in results_data:
        for reg_name in res['per_shot_cregs'].keys():
            match = re.search(r"c_data_([\d_]+)", reg_name)
            if match:
                all_chains.add(match.group(1))

    # Save the small, human-readable params file for easy filtering later.
    try:
        def _first_metadata_value(key):
            for meta in manifest_metadata_list:
                value = meta.get(key)
                if value is not None:
                    return value
            return None

        extracted_D = _first_metadata_value('D')
        extracted_r = _first_metadata_value('n_syndrome_rounds')
        if extracted_r is None and results_data:
            extracted_r = results_data[0].get('metadata', {}).get('n_syndrome_rounds')

        params = {
            'D': extracted_D,
            'r': extracted_r,
            'bases': unique_bases
        }

        with open(processed_job_path / "params.json", 'w') as f:
            json.dump(params, f, indent=2)
        print(f"  > Saved filterable parameters to 'params.json'")
    except (IndexError, KeyError) as e:
        warnings.warn(f"Could not extract D/r to create params.json. Filtering may fail. Error: {e}")

    # --- 3. PERFORM ALL PRE-COMPUTATIONS FOR METADATA.PT ---

    # Create the global hardware graph once.
    global_system_graph = create_rich_system_graph(metadata_json.get('backend_properties'))

    # Pre-compute the subgraph for each unique chain once.
    print("  > Pre-computing system subgraphs for all unique qubit chains...")
    system_subgraphs_by_chain = {}
    if global_system_graph.num_nodes > 0:
        for chain_id_str in sorted(list(all_chains)):
            chain_qubits = [int(q) for q in chain_id_str.split('_')]
            subset = torch.tensor(chain_qubits, dtype=torch.long)
            edge_index, edge_attr = subgraph(
                subset, global_system_graph.edge_index, global_system_graph.edge_attr,
                relabel_nodes=True, num_nodes=global_system_graph.num_nodes
            )
            x = global_system_graph.x[subset]
            system_subgraphs_by_chain[chain_id_str] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"    > Computed {len(system_subgraphs_by_chain)} unique subgraphs.")

    # Create the circuit feature map for each unique chain once.
    print("  > Processing circuit feature maps...")
    circuit_feature_maps_by_chain = {}
    for chain_id_str in sorted(list(all_chains)):
        chain_qubits = [int(q) for q in chain_id_str.split('_')]
        D = len(chain_qubits)
        params_r = params.get('r', params.get('T'))
        if D <= 1 or params.get('D') is None or params_r is None: continue

        circuit_map = create_circuit_temporal_map(qasm_file_path, chain_qubits, params['D'], params_r)
        if circuit_map.numel() > 0:
            circuit_feature_maps_by_chain[chain_id_str] = circuit_map

    # --- 4. PROCESS ALL SHOTS INTO A SINGLE LIST ---

    missing_manifest_circuits = set()
    missing_basis_circuits = set()
    all_shots_data = []
    for idx, circuit_result in enumerate(tqdm(results_data, desc="Processing and collecting shots")):
        circuit_name = circuit_result.get('circuit_name')
        manifest_meta = manifest_lookup_by_circuit.get(circuit_name)
        if not manifest_meta and idx < len(manifest_metadata_list):
            manifest_meta = manifest_metadata_list[idx]
        if not manifest_meta and manifest_lookup_by_index:
            manifest_meta = manifest_lookup_by_index.get(idx)

        if circuit_name and not manifest_meta and circuit_name not in missing_manifest_circuits:
            warnings.warn(
                f"Missing manifest metadata for circuit '{circuit_name}' (index {idx}) in job '{job_id}'. Basis metadata will be unavailable for these shots."
            )
            missing_manifest_circuits.add(circuit_name)
            manifest_meta = {}

        metadata_shot = circuit_result.get('metadata', {})
        intended_logical_state = metadata_shot.get('logical_state')
        if intended_logical_state is None:
            intended_logical_state = manifest_meta.get('logical_state')
        num_rounds = metadata_shot.get('n_syndrome_rounds')
        if num_rounds is None:
            num_rounds = manifest_meta.get('n_syndrome_rounds')

        shot_basis = metadata_shot.get('basis') or manifest_meta.get('basis')
        if (
            shot_basis is None
            and circuit_name
            and circuit_name not in missing_basis_circuits
            and circuit_name not in missing_manifest_circuits
        ):
            warnings.warn(f"Basis metadata missing for circuit '{circuit_name}' in job '{job_id}'.")
            missing_basis_circuits.add(circuit_name)

        if intended_logical_state is None or num_rounds is None:
            continue

        for reg_name, shots in circuit_result['per_shot_cregs'].items():
            match = re.search(r"c_data_([\d_]+)", reg_name)
            if not match: continue

            chain_id_str = match.group(1)
            syndrome_reg_name = f"c_syndrome_{chain_id_str}"
            if syndrome_reg_name not in circuit_result['per_shot_cregs']: continue

            data_shots = np.array(shots)
            syndrome_shots = np.array(circuit_result['per_shot_cregs'][syndrome_reg_name])
            if data_shots.ndim < 2 or data_shots.shape[0] == 0: continue

            D_shot = data_shots.shape[1]
            if D_shot <= 1: continue
            num_stabilizers = D_shot - 1

            for i in range(len(data_shots)):
                syndrome_history = syndrome_shots[i].reshape(num_rounds, num_stabilizers)
                s_padded = np.pad(syndrome_history, ((1, 0), (0, 0)), 'constant', constant_values=0)
                detection_events = (s_padded[:-1, :] + s_padded[1:, :]) % 2

                perfect_state = np.ones(D_shot, dtype=int) * intended_logical_state
                target_correction = (perfect_state + data_shots[i]) % 2

                sample = {
                    'syndrome_block': torch.tensor(detection_events, dtype=torch.float).unsqueeze(0),
                    'target_correction': torch.from_numpy(target_correction).float(),
                    'final_measurement': torch.tensor(data_shots[i], dtype=torch.long),
                    'intended_logical_state': intended_logical_state,
                    'chain_id_str': chain_id_str,
                    'basis': shot_basis,
                    'circuit_name': circuit_name,
                    'circuit_index': idx,
                }
                all_shots_data.append(sample)

    # --- 5. SAVE THE FINAL, CONSOLIDATED ARTIFACTS ---

    consolidated_metadata = {
        'global_system_graph': global_system_graph,
        'system_subgraphs_by_chain': system_subgraphs_by_chain,
        'circuit_feature_maps_by_chain': circuit_feature_maps_by_chain,
        'NODE_FEATURES': NODE_FEATURES,
        'OP_FEATURES': OP_FEATURES,
        'unique_bases': unique_bases,
        'manifest_metadata_by_circuit': manifest_lookup_by_circuit,
        'manifest_metadata_by_index': manifest_lookup_by_index,
        'manifest_metadata_list': manifest_metadata_list,
    }
    metadata_filepath = processed_job_path / "metadata.pt"
    torch.save(consolidated_metadata, metadata_filepath)
    print(f"\nSaved consolidated metadata to '{metadata_filepath}'")

    shots_filepath = processed_job_path / "shots.pt"
    torch.save(all_shots_data, shots_filepath)
    print(f"Saved {len(all_shots_data)} processed shots to '{shots_filepath}'")
    print(f"--- Successfully finished processing Job ID: {job_id} ---")

def main():
    """
    Main execution script. Finds all valid job directories in the input 'results'
    directory and processes each one sequentially.
    """
    parser = argparse.ArgumentParser(
        description="Prepare all experimental data in a results directory for ML decoding."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to the top-level results directory containing multiple job subdirectories."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="processed_data",
        help="Top-level directory to save all processed ML data."
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"Error: Provided results directory does not exist: '{results_path}'")
        return

    # --- Find all valid job directories ---
    job_dirs_to_process = []
    for sub_dir in results_path.iterdir():
        if sub_dir.is_dir() and (sub_dir / "results.json").exists():
            job_dirs_to_process.append(sub_dir)

    if not job_dirs_to_process:
        print(f"Error: No valid job directories (containing a results.json) found in '{results_path}'")
        return

    print(f"Found {len(job_dirs_to_process)} job directories to process.")

    # --- Loop through each job and process it ---
    for job_path in job_dirs_to_process:
        print("\n" + "=" * 80)
        print(f"--- Processing Job Directory: {job_path.name} ---")
        print("=" * 80)
        try:
            # Call the existing function to do the heavy lifting for this one job
            process_and_save_data(str(job_path), args.out_dir)
        except Exception as e:
            print(f"\n!!! An error occurred while processing job '{job_path.name}': {e}")
            print("!!! Skipping this job and moving to the next one.")
            continue

    print("\n\n--- All jobs have been processed. ---")


if __name__ == "__main__":
    # --- MODIFIED: The script now takes the top-level directory as input ---
    # The argparse logic is now handled inside main()
    main()