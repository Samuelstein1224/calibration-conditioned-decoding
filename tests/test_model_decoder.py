"""Tests for ibm_qec.model.decoder — model construction and forward pass."""
import pytest
import torch
from torch_geometric.data import Data, Batch


class TestGeneralRepCodeDecoderConstruction:
    def test_construct_default_channels(self):
        from ibm_qec.model.decoder import GeneralRepCodeDecoder
        model = GeneralRepCodeDecoder(D_max=11, r_max=11, system_node_features=4, system_edge_features=0)
        assert model is not None

    def test_construct_small_channels(self):
        from ibm_qec.model.decoder import GeneralRepCodeDecoder
        model = GeneralRepCodeDecoder(D_max=5, r_max=3, system_node_features=4, system_edge_features=0, channels=8)
        assert model is not None

    def test_construct_various_sizes(self):
        from ibm_qec.model.decoder import GeneralRepCodeDecoder
        for D, r in [(3, 3), (5, 5), (11, 11)]:
            model = GeneralRepCodeDecoder(D_max=D, r_max=r, system_node_features=4, system_edge_features=0, channels=8)
            assert model.D_max == D
            assert model.r_max == r


class TestGeneralRepCodeDecoderForward:
    def test_output_shape(self, small_model, synthetic_batch):
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=4, r=11, D=11)
        out = small_model(syndromes, graph_batch)
        assert out.shape == (4, 1, 11)

    def test_output_range(self, small_model, synthetic_batch):
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=4, r=11, D=11)
        out = small_model(syndromes, graph_batch)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_variable_d_t_smaller(self, small_model, synthetic_batch):
        """Model built with D_max=11, r_max=11 handles smaller D=5, r=3."""
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=2, r=3, D=5)
        out = small_model(syndromes, graph_batch)
        assert out.shape == (2, 1, 5)

    def test_oversized_input_raises(self, small_model, synthetic_batch):
        """Input exceeding D_max/r_max raises ValueError."""
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=2, r=20, D=20)
        with pytest.raises(ValueError, match="exceeds model's max size"):
            small_model(syndromes, graph_batch)

    def test_batch_size_one(self, small_model, synthetic_batch):
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=1, r=11, D=11)
        out = small_model(syndromes, graph_batch)
        assert out.shape == (1, 1, 11)

    def test_deterministic_eval(self, small_model, synthetic_batch):
        small_model.eval()
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=2, r=11, D=11)
        out1 = small_model(syndromes, graph_batch)
        out2 = small_model(syndromes, graph_batch)
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self, small_model, synthetic_batch):
        small_model.train()
        syndromes, graph_batch, labels, _, _ = synthetic_batch(N=4, r=11, D=11)
        out = small_model(syndromes, graph_batch)
        loss = torch.nn.functional.mse_loss(out[:, 0, :], labels)
        loss.backward()
        for name, p in small_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestConditionedDecoderForward:
    def test_output_shape(self, small_conditioned_model, synthetic_batch):
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=4, r=11, D=11)
        out = small_conditioned_model(syndromes, graph_batch)
        assert out.shape == (4, 1, 11)

    def test_output_range(self, small_conditioned_model, synthetic_batch):
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=4, r=11, D=11)
        out = small_conditioned_model(syndromes, graph_batch)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_film_conditioning_changes_output(self, small_conditioned_model, synthetic_graph):
        """Different graph features produce different outputs."""
        small_conditioned_model.eval()
        N, T, D = 2, 5, 5
        syndromes = torch.randn(N, 1, T, D - 1)

        graph_a = synthetic_graph(num_nodes=2 * D - 1, node_features=4)
        graph_b = synthetic_graph(num_nodes=2 * D - 1, node_features=4)
        # Ensure features differ
        graph_b.x = graph_a.x + 10.0

        batch_a = Batch.from_data_list([graph_a] * N)
        batch_b = Batch.from_data_list([graph_b] * N)

        out_a = small_conditioned_model(syndromes, batch_a)
        out_b = small_conditioned_model(syndromes, batch_b)
        assert not torch.allclose(out_a, out_b, atol=1e-6)

    def test_variable_d_t(self, small_conditioned_model, synthetic_batch):
        syndromes, graph_batch, _, _, _ = synthetic_batch(N=2, r=3, D=5)
        out = small_conditioned_model(syndromes, graph_batch)
        assert out.shape == (2, 1, 5)

    def test_gradient_flow(self, small_conditioned_model, synthetic_batch):
        small_conditioned_model.train()
        syndromes, graph_batch, labels, _, _ = synthetic_batch(N=4, r=11, D=11)
        out = small_conditioned_model(syndromes, graph_batch)
        loss = torch.nn.functional.mse_loss(out[:, 0, :], labels)
        loss.backward()
        for name, p in small_conditioned_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
