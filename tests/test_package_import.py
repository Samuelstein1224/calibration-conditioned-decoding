"""Verify that the ibm_qec package and all subpackages import correctly."""
import pytest


class TestSubpackageImports:
    def test_import_ibm_qec(self):
        import ibm_qec  # noqa: F401

    def test_import_model(self):
        import ibm_qec.model  # noqa: F401

    def test_import_model_decoder(self):
        import ibm_qec.model.decoder  # noqa: F401

    def test_import_data(self):
        import ibm_qec.data  # noqa: F401

    def test_import_data_dataset(self):
        import ibm_qec.data.dataset  # noqa: F401

    def test_import_evaluation(self):
        import ibm_qec.evaluation  # noqa: F401

    def test_import_evaluation_metrics(self):
        import ibm_qec.evaluation.metrics  # noqa: F401

    def test_import_baselines(self):
        import ibm_qec.baselines  # noqa: F401

    def test_import_baselines_mwpm_eval(self):
        import ibm_qec.baselines.mwpm_eval  # noqa: F401


class TestKeyClassImports:
    def test_import_general_rep_code_decoder(self):
        from ibm_qec.model.decoder import GeneralRepCodeDecoder  # noqa: F401

    def test_import_general_conditioned_rep_code_decoder(self):
        from ibm_qec.model.decoder import GeneralConditionedRepCodeDecoder  # noqa: F401

    def test_import_multi_job_qec_dataset(self):
        from ibm_qec.data.dataset import MultiJobQECDataset  # noqa: F401

    def test_import_variable_size_collate_fn(self):
        from ibm_qec.data.dataset import variable_size_collate_fn  # noqa: F401

    def test_import_evaluate_decoder(self):
        from ibm_qec.evaluation.metrics import evaluate_decoder  # noqa: F401

    def test_import_device(self):
        from ibm_qec.device import DEVICE
        assert DEVICE is not None
