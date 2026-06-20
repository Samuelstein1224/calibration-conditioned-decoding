"""Integration tests — end-to-end pipelines through ibm_qec."""
import pytest
import torch
from torch.utils.data import DataLoader

from ibm_qec.device import DEVICE


class TestDataToModelPipeline:
    def test_sample_data_through_model(self, sample_job_z_path):
        """Load sample data, build model, run evaluate_decoder."""
        from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
        from ibm_qec.model.decoder import GeneralRepCodeDecoder
        from ibm_qec.evaluation.metrics import evaluate_decoder

        ds = MultiJobQECDataset([sample_job_z_path])
        loader = DataLoader(ds, batch_size=16, collate_fn=variable_size_collate_fn)

        model = GeneralRepCodeDecoder(
            D_max=ds.max_D, r_max=ds.max_r,
            system_node_features=4, system_edge_features=0,
            channels=8,
        ).to(DEVICE)

        result = evaluate_decoder(model, loader)
        assert result["shots_total"] == 256
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["shots_0"] + result["shots_1"] == 256

    def test_conditioned_model_pipeline(self, sample_job_z_path):
        """Same pipeline with GeneralConditionedRepCodeDecoder."""
        from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
        from ibm_qec.model.decoder import GeneralConditionedRepCodeDecoder
        from ibm_qec.evaluation.metrics import evaluate_decoder

        ds = MultiJobQECDataset([sample_job_z_path])
        loader = DataLoader(ds, batch_size=16, collate_fn=variable_size_collate_fn)

        model = GeneralConditionedRepCodeDecoder(
            D_max=ds.max_D, r_max=ds.max_r,
            system_node_features=4, system_edge_features=0,
            channels=8,
        ).to(DEVICE)

        result = evaluate_decoder(model, loader)
        assert result["shots_total"] == 256
        assert 0.0 <= result["accuracy"] <= 1.0


class TestTrainingStep:
    def test_forward_backward_optimizer(self, small_model, synthetic_batch):
        """Forward + backward + optimizer.step works without error."""
        small_model.train()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        syndromes, graph_batch, labels, _, _ = synthetic_batch(N=4, r=11, D=11)
        out = small_model(syndromes, graph_batch)
        loss = torch.nn.functional.mse_loss(out[:, 0, :], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0.0

    def test_conditioned_training_step(self, small_conditioned_model, synthetic_batch):
        """Training step for conditioned model."""
        small_conditioned_model.train()
        optimizer = torch.optim.Adam(small_conditioned_model.parameters(), lr=1e-3)

        syndromes, graph_batch, labels, _, _ = synthetic_batch(N=4, r=11, D=11)
        out = small_conditioned_model(syndromes, graph_batch)
        loss = torch.nn.functional.mse_loss(out[:, 0, :], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0.0


class TestCollatedBatchThroughModel:
    def test_collated_batch_forward(self, sample_job_z_path):
        """variable_size_collate_fn output goes through model without shape errors."""
        from ibm_qec.data.dataset import MultiJobQECDataset, variable_size_collate_fn
        from ibm_qec.model.decoder import GeneralRepCodeDecoder

        ds = MultiJobQECDataset([sample_job_z_path])
        batch_list = [ds[i] for i in range(8)]
        syndromes, graph_batch, labels, initial_states, final_data = variable_size_collate_fn(batch_list)

        model = GeneralRepCodeDecoder(
            D_max=ds.max_D, r_max=ds.max_r,
            system_node_features=4, system_edge_features=0,
            channels=8,
        )
        out = model(syndromes, graph_batch)
        assert out.shape[0] == 8
        assert out.shape[1] == 1
