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

# PATENTS = {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            'patent': [100, 202, 250, 320, 370, 385, 415, 440, 500, 550],
#            'contents': ['test sentence', 'test sentence', 'test sentence', 'test sentence',
#                         'test sentence', 'test sentence', 'test sentence', 'test sentence',
#                         'test sentence', 'test sentence'],
#            'year': [1997, 1998, 1998, 1999, 2000, 2001, 2001, 2002, 2003, 2003]
#            }

# PATENT_INDEX = 4
# BACKWARD_BLOCK_EXPECTED = pd.DataFrame({'index': [0, 1, 2, 3],
#                                         'patent': [100, 202, 250, 320],
#                                         'contents': ['test sentence', 'test sentence',
#                                                      'test sentence', 'test sentence'],
#                                         'year': [1997, 1998, 1998, 1999]
#                                         })
# FORWARD_BLOCK_EXPECTED = pd.DataFrame({'index': [5, 6, 7, 8, 9],
#                                        'patent': [385, 415, 440, 500, 550],
#                                        'contents': ['test sentence', 'test sentence',
#                                                     'test sentence', 'test sentence',
#                                                     'test sentence'],
#                                        'year': [2001, 2001, 2002, 2003, 2003]
#                                        })


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
#
# def process_data():
#     """ Set parameters for expected dataset"""
#     data_fp = io.BytesIO()
#     df_patents = pd.DataFrame(data=PATENTS)
#     documents = df_patents['contents']
#     model = D2VEmbedder()
#     model.fit(documents)
#     embeddings = model.transform(documents)
#     with DataModel(data_fp, read_only=False) as data:
#         data.store_embeddings("test", "d2v", embeddings)
#         data.store_window("test", PATENTS["patent"], PATENTS["year"])
#         analysis = DocAnalysis(data)
#         impacts = analysis.patent_impacts("test", "d2v")
#         novelties = analysis.patent_novelties("test", "d2v")
#     return impacts, novelties


# def test_collect_blocks():
#     """ Function to test collect_blocks() functionality"""
#     sim = process_data()
#
#     sim.collect_blocks(PATENT_INDEX)
#
#     assert_frame_equal(
#         sim.backward_block[['index', 'patent', 'contents', 'year']].reset_index(drop=True),
#         BACKWARD_BLOCK_EXPECTED.reset_index(drop=True))
#
#     assert_frame_equal(
#         sim.forward_block[['index', 'patent', 'contents', 'year']].reset_index(drop=True),
#         FORWARD_BLOCK_EXPECTED.reset_index(drop=True))


# @pytest.mark.parametrize("patent_index", [3, 4])
# def test_compute_impact():
#     """Function to test compute_impact() functionality"""
#     impacts, novelties = process_data()
#     # sim.collect_blocks(PATENT_INDEX)
#     # sim.compute_impact(PATENT_INDEX)
#
#     assert int(impacts[PATENT_INDEX]) == 1
#     assert int(novelties[PATENT_INDEX]) == 0


# def test_compute_novelty():
#     """Function to test compute_novelty functionality"""
#     sim = process_data()
#     sim.collect_blocks(PATENT_INDEX)
#     sim.compute_impact(PATENT_INDEX)
#     sim.compute_novelty(PATENT_INDEX)
#
#     assert int(sim.df_patents_embeddings.loc[PATENT_INDEX, 'novelty']) == 0
