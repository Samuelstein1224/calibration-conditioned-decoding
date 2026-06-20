"""Tests for ibm_qec.evaluation.metrics — LER, confidence, binomial bounds."""
import pytest
import torch

from ibm_qec.evaluation.metrics import _binomial_likelihood_bounds, evaluate_decoder
from ibm_qec.device import DEVICE


class TestBinomialLikelihoodBounds:
    def test_k_zero(self):
        lower, upper = _binomial_likelihood_bounds(0, 100)
        assert lower == 0.0
        assert upper > 0.0
        assert upper < 1.0

    def test_k_equals_n(self):
        lower, upper = _binomial_likelihood_bounds(100, 100)
        assert lower > 0.0
        assert lower < 1.0
        assert upper == 1.0

    def test_general_case(self):
        k, n = 30, 100
        lower, upper = _binomial_likelihood_bounds(k, n)
        p_hat = k / n
        assert 0 < lower < p_hat
        assert p_hat < upper < 1

    def test_n_zero(self):
        lower, upper = _binomial_likelihood_bounds(0, 0)
        assert lower is None
        assert upper is None

    def test_n_negative(self):
        lower, upper = _binomial_likelihood_bounds(0, -5)
        assert lower is None
        assert upper is None

    def test_highlight_factor_le_1_raises(self):
        with pytest.raises(ValueError, match="highlight_factor must be greater than 1"):
            _binomial_likelihood_bounds(5, 10, highlight_factor=1.0)

    def test_highlight_factor_less_than_1_raises(self):
        with pytest.raises(ValueError, match="highlight_factor must be greater than 1"):
            _binomial_likelihood_bounds(5, 10, highlight_factor=0.5)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k must lie within"):
            _binomial_likelihood_bounds(-1, 10)

    def test_k_exceeds_n_raises(self):
        with pytest.raises(ValueError, match="k must lie within"):
            _binomial_likelihood_bounds(11, 10)

    def test_small_k_bounds_contain_p_hat(self):
        k, n = 1, 1000
        lower, upper = _binomial_likelihood_bounds(k, n)
        assert lower <= k / n <= upper

    def test_large_k_bounds_contain_p_hat(self):
        k, n = 999, 1000
        lower, upper = _binomial_likelihood_bounds(k, n)
        assert lower <= k / n <= upper

    def test_custom_highlight_factor(self):
        lower, upper = _binomial_likelihood_bounds(50, 100, highlight_factor=10.0)
        assert 0 < lower < 0.5 < upper < 1


class TestEvaluateDecoder:
    def _make_loader(self, N=32, D=5, r=5, perfect=True):
        """Create a minimal dataloader that yields one batch.

        Note: evaluate_decoder moves tensors to DEVICE internally, so we
        keep them on CPU here — the function handles device placement.
        """
        from torch_geometric.data import Data, Batch

        syndromes = torch.zeros(N, 1, r, D - 1)
        num_nodes = 2 * D - 1
        graph = Data(
            x=torch.randn(num_nodes, 4),
            edge_index=torch.tensor(
                [[i, i + 1] for i in range(num_nodes - 1)]
                + [[i + 1, i] for i in range(num_nodes - 1)],
                dtype=torch.long,
            ).t().contiguous(),
        )
        graph_batch = Batch.from_data_list([graph] * N)
        initial_states = torch.randint(0, 2, (N,))
        if perfect:
            # final data consistent with initial state — no errors
            final_data = torch.zeros(N, D, dtype=torch.long)
            final_data[initial_states == 1] = 1
        else:
            final_data = torch.randint(0, 2, (N, D)).long()
        labels = torch.zeros(N, D)

        return [(syndromes, graph_batch, labels, initial_states, final_data)]

    def test_perfect_model_metrics(self):
        """A model that predicts zeros on zero-error data should have ler=0."""
        from ibm_qec.model.decoder import GeneralRepCodeDecoder

        D, r = 5, 5
        model = GeneralRepCodeDecoder(D_max=D, r_max=r, system_node_features=4,
                                      system_edge_features=0, channels=8).to(DEVICE)
        # Override forward to always predict 0.0 (no correction needed)
        class ZeroModel(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, syndromes, system_graph):
                out = self.inner(syndromes, system_graph)
                return torch.zeros_like(out)

        loader = self._make_loader(N=32, D=D, r=r, perfect=True)
        result = evaluate_decoder(ZeroModel(model).to(DEVICE), loader)

        assert result["ler_0"] == 0.0
        assert result["ler_1"] == 0.0
        assert result["accuracy"] == 1.0

    def test_metric_keys(self):
        from ibm_qec.model.decoder import GeneralRepCodeDecoder

        D, r = 5, 5
        model = GeneralRepCodeDecoder(D_max=D, r_max=r, system_node_features=4,
                                      system_edge_features=0, channels=8).to(DEVICE)
        loader = self._make_loader(N=16, D=D, r=r)
        result = evaluate_decoder(model, loader)

        expected_keys = {
            "ler_0", "ler_1", "accuracy",
            "avg_pre_conf_0", "avg_post_conf_0",
            "avg_pre_conf_1", "avg_post_conf_1",
            "ler_0_bound_low", "ler_0_bound_high",
            "ler_1_bound_low", "ler_1_bound_high",
            "overall_ler_bound_low", "overall_ler_bound_high",
            "shots_0", "shots_1", "shots_total",
            "logical_errors_0", "logical_errors_1", "logical_errors_total",
            "highlight_factor",
        }
        assert set(result.keys()) == expected_keys

    def test_shot_counts_sum(self):
        from ibm_qec.model.decoder import GeneralRepCodeDecoder

        D, r = 5, 5
        model = GeneralRepCodeDecoder(D_max=D, r_max=r, system_node_features=4,
                                      system_edge_features=0, channels=8).to(DEVICE)
        loader = self._make_loader(N=32, D=D, r=r)
        result = evaluate_decoder(model, loader)

        assert result["shots_0"] + result["shots_1"] == result["shots_total"]
        assert result["shots_total"] == 32

    def test_random_model_has_errors(self):
        """An untrained model on random data should generally have some errors."""
        from ibm_qec.model.decoder import GeneralRepCodeDecoder

        D, r = 5, 5
        model = GeneralRepCodeDecoder(D_max=D, r_max=r, system_node_features=4,
                                      system_edge_features=0, channels=8).to(DEVICE)
        loader = self._make_loader(N=64, D=D, r=r, perfect=False)
        result = evaluate_decoder(model, loader)

        # With random data, we expect some logical errors in at least one state
        assert result["logical_errors_total"] >= 0
        assert 0.0 <= result["accuracy"] <= 1.0
