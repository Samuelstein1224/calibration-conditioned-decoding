"""Shared fixtures for ibm_qec test suite."""
import json
import pathlib

import pytest
import torch
from torch_geometric.data import Data, Batch

SAMPLE_DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "example" / "sample_data"


# ---------------------------------------------------------------------------
# Sample data paths
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_job_z_path():
    return str(SAMPLE_DATA_ROOT / "job_z")


@pytest.fixture
def sample_job_x_path():
    return str(SAMPLE_DATA_ROOT / "job_x")


# ---------------------------------------------------------------------------
# Synthetic graph helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_graph():
    """Linear-chain graph with 21 nodes and 4 node features (mimics D=11 rep code)."""
    def _make(num_nodes=21, node_features=4):
        x = torch.randn(num_nodes, node_features)
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    return _make


@pytest.fixture
def synthetic_batch(synthetic_graph):
    """Create a batch of syndromes + graph + labels ready for model forward pass."""
    def _make(N=4, r=11, D=11, graph=None):
        if graph is None:
            num_nodes = 2 * D - 1
            graph = synthetic_graph(num_nodes=num_nodes, node_features=4)

        syndromes = torch.randn(N, 1, r, D - 1)
        graph_batch = Batch.from_data_list([graph] * N)
        labels = torch.randint(0, 2, (N, D)).float()
        initial_states = torch.randint(0, 2, (N,))
        final_data = torch.randint(0, 2, (N, D)).long()
        return syndromes, graph_batch, labels, initial_states, final_data
    return _make


# ---------------------------------------------------------------------------
# Small model fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_model():
    """Tiny GeneralRepCodeDecoder for fast tests."""
    from ibm_qec.model.decoder import GeneralRepCodeDecoder
    return GeneralRepCodeDecoder(
        D_max=11, r_max=11,
        system_node_features=4, system_edge_features=0,
        channels=8, embedding_dim=16,
    )


@pytest.fixture
def small_conditioned_model():
    """Tiny GeneralConditionedRepCodeDecoder for fast tests."""
    from ibm_qec.model.decoder import GeneralConditionedRepCodeDecoder
    return GeneralConditionedRepCodeDecoder(
        D_max=11, r_max=11,
        system_node_features=4, system_edge_features=0,
        channels=8, embedding_dim=16,
    )


# ---------------------------------------------------------------------------
# Temporary job directory for dataset tests
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_job_dir(tmp_path):
    """Create a minimal job directory with shots.pt / metadata.pt / params.json."""
    D, r = 5, 3
    num_qubits = 2 * D - 1
    chain_id = "_".join(str(i) for i in range(num_qubits))

    graph = Data(
        x=torch.randn(num_qubits, 4),
        edge_index=torch.tensor(
            [[i, i + 1] for i in range(num_qubits - 1)]
            + [[i + 1, i] for i in range(num_qubits - 1)],
            dtype=torch.long,
        ).t().contiguous(),
    )

    shots = []
    for _ in range(8):
        shots.append({
            "syndrome_block": torch.randn(1, r, D - 1),
            "target_correction": torch.randint(0, 2, (D,)).float(),
            "final_measurement": torch.randint(0, 2, (D,)).long(),
            "intended_logical_state": int(torch.randint(0, 2, (1,)).item()),
            "chain_id_str": chain_id,
            "basis": "Z",
            "circuit_name": "test_circuit",
            "circuit_index": 0,
        })

    metadata = {
        "system_subgraphs_by_chain": {chain_id: graph},
        "global_system_graph": graph,
        "NODE_FEATURES": ["T1_norm", "T2_norm", "readout_error_norm", "freq_offset_norm"],
        "OP_FEATURES": ["IDLE", "X", "Y", "Z"],
    }

    job_dir = tmp_path / "test_job"
    job_dir.mkdir()
    torch.save(shots, job_dir / "shots.pt")
    torch.save(metadata, job_dir / "metadata.pt")
    with open(job_dir / "params.json", "w") as f:
        json.dump({"D": D, "r": r, "bases": ["Z"]}, f)

    return job_dir
