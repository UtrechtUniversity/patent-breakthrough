import io
from pathlib import Path

from pytest import mark
import numpy as np

from docembedder.datamodel import DataModel
import pytest
import scipy
from docembedder.models.tfidf import TfidfEmbedder
from docembedder.models.bert import BERTEmbedder
from docembedder.models.bpemb import BPembEmbedder
from docembedder.models.countvec import CountVecEmbedder
from docembedder.models.doc2vec import D2VEmbedder
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.preprocessor.oldprep import OldPreprocessor


def get_file(real_file, extra=""):
    if real_file:
        hdf5_fp = Path("tests", "data", f"temp{extra}.h5")
        hdf5_fp.unlink(missing_ok=True)
    else:
        hdf5_fp = io.BytesIO()
    return hdf5_fp


@mark.parametrize("real_file", [True, False])
def test_window(real_file):
    hdf5_fp = get_file(real_file)
    patent_id = np.sort(np.random.choice(1000, size=100, replace=False))
    patent_id_2 = patent_id + 1
    year = np.arange(1800, 1900)
    year_2 = year + 1
    with DataModel(hdf5_fp, read_only=False) as data:
        data.store_window("test", patent_id, year)
        assert "test" in data.window_list
        assert data.has_window("test")

        new_patent, new_year = data.load_window("test")
        assert np.all(new_year == year)
        assert np.all(patent_id == new_patent)

        with pytest.raises(AssertionError):
            data.store_window("test", patent_id_2, year)

        with pytest.raises(ValueError):
            data.store_window("x", patent_id[:50], year)

        with pytest.raises(AssertionError):
            data.store_window("test", patent_id, year_2)


def assert_same(array_a, array_b, dense):
    if dense:
        assert np.all(array_a == array_b)
    else:
        assert ((array_a != array_b).nnz == 0)


@mark.parametrize("real_file", [True, False])
@mark.parametrize("dense", [True, False])
def test_embeddings(real_file, dense):
    hdf5_fp = get_file(real_file)
    hdf5_fp2 = get_file(real_file, extra="2")
    embeddings = 2*np.random.rand(200).reshape(10, -1).astype(int)
    embeddings_2 = embeddings + 1
    patent_id = np.sort(np.random.choice(1000, size=100, replace=False))
    year = np.arange(1800, 1900)

    if not dense:
        embeddings_2 = scipy.sparse.csr_matrix(embeddings+1)
        embeddings = scipy.sparse.csr_matrix(embeddings)

    with DataModel(hdf5_fp, read_only=False) as data:
        data.store_window("window", patent_id, year)
        with pytest.raises(ValueError):
            data.store_embeddings("window", "model", 10)

        data.store_embeddings("window", "model", embeddings)
        assert ("window", "model") in list(data.iterate_window_models())

        new_embeddings = data.load_embeddings("window", "model")
        assert_same(embeddings, new_embeddings, dense)

        data.store_embeddings("window", "model", embeddings_2, overwrite=False)
        new_embeddings = data.load_embeddings("window", "model")
        assert_same(embeddings, new_embeddings, dense)

        data.store_embeddings("window", "model", embeddings_2, overwrite=True)
        new_embeddings = data.load_embeddings("window", "model")
        assert_same(embeddings_2, new_embeddings, dense)

        data.store_cpc_spearmanr("window", "model", 0.5)
        assert data.load_cpc_spearmanr("window", "model") == 0.5

        with DataModel(hdf5_fp2, read_only=False) as data2:
            data2.store_window("window", patent_id, year)
            data2.store_embeddings("window", "model_2", patent_id, year)
        with pytest.raises(FileNotFoundError):
            data.add_data("test")
        data.add_data(hdf5_fp2, delete_copy=True)
        assert "model_2" in data.model_names


@mark.parametrize("real_file", [True, False])
def test_cpc_correlations(real_file):
    hdf5_fp = get_file(real_file)
    i_patents = np.arange(100)
    j_patents = 1 + np.arange(100)
    correlations = np.random.rand(100)
    all_cor = {"i_patents": i_patents, "j_patents": j_patents,
               "correlations": correlations}
    with DataModel(hdf5_fp, read_only=False) as data:
        data.store_cpc_correlations("test", all_cor)
        assert data.has_cpc("test")
        new_cpc = data.load_cpc_correlations("test")
        for key, arr in new_cpc.items():
            assert np.all(all_cor[key] == arr)
        data.store_cpc_correlations("test", 100)


@mark.parametrize("real_file", [True, False])
def test_impact_novelty(real_file):
    hdf5_fp = get_file(real_file)
    novelty = np.random.rand(100)
    impact = np.random.rand(100)
    patent_ids = np.arange(100)
    results = {"exponent": 1.0, "patent_ids": patent_ids, "impact": impact,
               "novelty": novelty, "focal_year": 1850}
    results_2 = {"exponent": 1.0, "patent_ids": patent_ids, "impact": impact+1,
                 "novelty": novelty+1, "focal_year": 1850}
    with DataModel(hdf5_fp, read_only=False) as data:
        data.store_impact_novelty("window", "model", results)
        new_results = data.load_impact_novelty("window", "model", 1.0)
        assert np.all(new_results["impact"] == impact) and np.all(new_results["novelty"] == novelty)

        data.store_impact_novelty("window", "model", results_2, overwrite=False)
        new_results = data.load_impact_novelty("window", "model", 1.0)
        assert np.all(new_results["impact"] == impact) and np.all(new_results["novelty"] == novelty)

        data.store_impact_novelty("window", "model", results_2, overwrite=True)
        new_results = data.load_impact_novelty("window", "model", 1.0)

        assert np.all(new_results["impact"] == impact+1) and np.all(new_results["novelty"] == novelty+1)


@mark.parametrize("real_file", [True, False])
@mark.parametrize("model",
                  [
                      TfidfEmbedder(ngram_max=2, stop_words=None),
                      BERTEmbedder("anferico/bert-for-patents"),
                      CountVecEmbedder("prop"),
                      BPembEmbedder(25, 1000),
                      D2VEmbedder(30, 1, 3, 4)
                  ])
def test_model_store(real_file, model):
    hdf5_fp = get_file(real_file)
    with DataModel(hdf5_fp, read_only=False) as data:
        data.store_model("tfidf", model)
        new_model = data.load_model("tfidf")
        assert model.settings == new_model.settings
        assert data.has_model("tfidf")


@mark.parametrize("real_file", [True, False])
@mark.parametrize("prep", [
    Preprocessor(keep_empty_patents=True),
    OldPreprocessor(keep_empty_patents=True, list_path=Path("tests", "data"))
])
def test_prep_store(real_file, prep):
    hdf5_fp = get_file(real_file)
    with DataModel(hdf5_fp, read_only=False) as data:
        data.store_preprocessor("prep", prep)
        new_prep = data.load_preprocessor("prep")
        assert new_prep.settings == prep.settings
        assert new_prep.__class__ == prep.__class__
        assert data.has_prep("prep")

