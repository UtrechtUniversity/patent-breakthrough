"""Test analysis.py functionality"""
import io

from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np
from pytest import mark

from docembedder import DocAnalysis
from docembedder.models import D2VEmbedder
from docembedder.datamodel import DataModel
from scipy.sparse import csr_matrix


def create_dataset(dense=True):
    data_fp = io.BytesIO()
    if dense:
        mat_type = np.array
    else:
        mat_type = csr_matrix
    with DataModel(data_fp, read_only=False) as data:
        patents_1 = 2*np.arange(5)
        patents_2 = 10+2*np.arange(5)
        data.store_window("window_1", patents_1, np.arange(5//2))
        data.store_embeddings("window_1", "model_1", mat_type([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]]))
        data.store_embeddings("window_1", "model_2", mat_type([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1]]))
        data.store_window("window_2", patents_2, 5+np.arange(5//2))
        data.store_embeddings("window_2", "model_1", mat_type([[0, 1], [0, 1], [1, 0], [1, 0], [0, 1]]))
        data.store_embeddings("window_2", "model_2", mat_type([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1]]))
        cpc_cor_1 = {
            "i_patents": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "j_patents": [1, 1, 2, 2, 3, 3, 4, 4, 0, 0],
            "correlations": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        cpc_cor_2 = {
            "i_patents": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "j_patents": [1, 1, 2, 2, 3, 3, 4, 4, 0, 0],
            "correlations": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        data.store_cpc_correlations("window_1", cpc_cor_1)
        data.store_cpc_correlations("window_2", cpc_cor_2)
    return data_fp


@mark.parametrize(
    "dense", [True, False]
)
def test_cpc_correlations(dense):
    data_fp = create_dataset(dense)
    with DataModel(data_fp) as data:
        analysis = DocAnalysis(data)
        cpc_cor = analysis.cpc_correlations(["model_1", "model_2"])
        assert isinstance(cpc_cor, dict)
        for model_name in ["model_1", "model_2"]:
            assert len(cpc_cor[model_name]["year"]) == 2
            assert len(cpc_cor[model_name]["correlations"]) == 2
        cpc_cor_2 = analysis.cpc_correlations(["model_1", "model_2"])
        assert cpc_cor == cpc_cor_2


def test_auto_correlation():
    dense_data_fp = create_dataset(dense=True)
    sparse_data_fp = create_dataset(dense=False)
    with DataModel(dense_data_fp) as data_dense, DataModel(sparse_data_fp) as data_sparse:
        analysis_dense = DocAnalysis(data_dense)
        analysis_sparse = DocAnalysis(data_sparse)
        for window in ["window_1", "window_2"]:
            for model in ["model_1", "model_2"]:
                delta_year_dense, auto_correlations_dense = analysis_dense.auto_correlation(window, model)
                delta_year_sparse, auto_correlations_sparse = analysis_sparse.auto_correlation(window, model)
                assert isinstance(delta_year_dense, np.ndarray)
                assert isinstance(auto_correlations_dense, np.ndarray)
                assert isinstance(delta_year_sparse, np.ndarray)
                assert isinstance(auto_correlations_sparse, np.ndarray)
                assert np.all(np.isclose(delta_year_dense, delta_year_sparse))
                assert np.all(np.isclose(auto_correlations_dense, auto_correlations_sparse))

# TODO: Add novelty/impact tests
