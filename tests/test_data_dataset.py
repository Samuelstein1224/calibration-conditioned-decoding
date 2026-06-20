"""Tests for ibm_qec.data.dataset — dataset loading, filtering, and collation."""
import pytest
import torch
from torch_geometric.data import Batch


class TestMultiJobQECDatasetLoading:
    def test_load_sample_z(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        ds = MultiJobQECDataset([sample_job_z_path])
        assert len(ds) == 256
        assert ds.max_D == 11
        assert ds.max_r == 11

    def test_load_both_jobs(self, sample_job_z_path, sample_job_x_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        ds = MultiJobQECDataset([sample_job_z_path, sample_job_x_path])
        assert len(ds) == 512

    def test_basis_filter_z_only(self, sample_job_z_path, sample_job_x_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        ds = MultiJobQECDataset([sample_job_z_path, sample_job_x_path], basis_filter=["Z"])
        assert len(ds) == 256

    def test_basis_filter_x_only(self, sample_job_z_path, sample_job_x_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        ds = MultiJobQECDataset([sample_job_z_path, sample_job_x_path], basis_filter=["X"])
        assert len(ds) == 256

    def test_distance_filter_keeps_all(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        ds = MultiJobQECDataset([sample_job_z_path], distance_filter={11})
        assert len(ds) == 256

    def test_distance_filter_removes_all(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        with pytest.raises(FileNotFoundError):
            MultiJobQECDataset([sample_job_z_path], distance_filter={3})

    def test_round_filter_keeps_all(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        ds = MultiJobQECDataset([sample_job_z_path], round_filter={11})
        assert len(ds) == 256

    def test_round_filter_removes_all(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        with pytest.raises(FileNotFoundError):
            MultiJobQECDataset([sample_job_z_path], round_filter={3})

    def test_missing_directory(self):
        from ibm_qec.data.dataset import MultiJobQECDataset
        with pytest.raises(FileNotFoundError):
            MultiJobQECDataset(["/nonexistent/path/job_fake"])


class TestDatasetGetitem:
    @pytest.fixture(autouse=True)
    def _load_dataset(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset
        self.ds = MultiJobQECDataset([sample_job_z_path])

    def test_getitem_keys(self):
        item = self.ds[0]
        expected_keys = {"syndrome_block", "target_correction", "system_subgraph",
                         "initial_logical_state", "final_measured_data"}
        assert set(item.keys()) == expected_keys

    def test_syndrome_block_shape(self):
        item = self.ds[0]
        assert item["syndrome_block"].shape == (1, 11, 10)

    def test_target_correction_shape(self):
        item = self.ds[0]
        assert item["target_correction"].shape == (11,)

    def test_system_subgraph_is_data(self):
        from torch_geometric.data import Data
        item = self.ds[0]
        assert isinstance(item["system_subgraph"], Data)


class TestVariableSizeCollateFn:
    def test_collate_output_shapes(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
        ds = MultiJobQECDataset([sample_job_z_path])
        batch_list = [ds[i] for i in range(4)]
        syndromes, graph_batch, labels, initial_states, final_data = variable_size_collate_fn(batch_list)

        assert syndromes.shape == (4, 1, 11, 10)
        assert labels.shape[0] == 4
        assert initial_states.shape == (4,)
        assert final_data.shape[0] == 4

    def test_collate_output_types(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
        ds = MultiJobQECDataset([sample_job_z_path])
        batch_list = [ds[i] for i in range(4)]
        syndromes, graph_batch, labels, initial_states, final_data = variable_size_collate_fn(batch_list)

        assert isinstance(syndromes, torch.Tensor)
        assert isinstance(graph_batch, Batch)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(initial_states, torch.Tensor)
        assert isinstance(final_data, torch.Tensor)

    def test_graph_batch_num_graphs(self, sample_job_z_path):
        from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
        ds = MultiJobQECDataset([sample_job_z_path])
        batch_list = [ds[i] for i in range(4)]
        _, graph_batch, _, _, _ = variable_size_collate_fn(batch_list)
        assert graph_batch.num_graphs == 4

    def test_collate_with_tmp_job(self, tmp_job_dir):
        """Collation works with synthetic minimal job data."""
        from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
        ds = MultiJobQECDataset([str(tmp_job_dir)])
        batch_list = [ds[i] for i in range(min(4, len(ds)))]
        syndromes, graph_batch, labels, initial_states, final_data = variable_size_collate_fn(batch_list)
        assert syndromes.ndim == 4  # (N, 1, T, D-1)
        assert syndromes.shape[1] == 1
