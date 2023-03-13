"""Testing for all the models."""
import pytest

import numpy as np
from pytest import mark
from scipy.sparse import csr_matrix

from docembedder.models import BPembEmbedder
from docembedder.models import D2VEmbedder
from docembedder.models import TfidfEmbedder
from docembedder.models import BERTEmbedder
from docembedder.models.countvec import CountVecEmbedder

documents = [
    "This is a very interesting sentence",
    "And here is another one",
    "We need this sentence too",
    "And let's repeat the sentence again",
]


def check_settings(model, param_names):
    for par_name in param_names:
        assert par_name in model.settings


@mark.parametrize(
    "param",
    [
        {"vector_size": 100, "min_count": 1, "epoch": 10, "workers": 4},
        {"vector_size": 30, "min_count": 2, "epoch": 20, "workers": 1},
        {"vector_size": 50, "min_count": 1, "epoch": 1, "workers": 2},
    ]
)
def test_doc2vec(param):
    """Function to test doc2vec.py functionality
    """
    embedder = D2VEmbedder(**param)
    embedder.fit(documents)
    embeddings = embedder.transform(documents)
    assert len(embeddings) == len(documents)
    assert len(embeddings[0]) == param["vector_size"]
    assert len(embeddings[1]) == param["vector_size"]

    with pytest.raises(ValueError):
        D2VEmbedder().transform(documents)

    param.pop("workers")
    check_settings(embedder, list(param))


@mark.parametrize(
    "param",
    [
        {},
        {"stop_words": "english", "stem": True, "sublinear_tf": False},
        {"stop_words": None, "stem": False, "sublinear_tf": True},
        {"norm": "l1", "sublinear_tf": False, "min_df": 1, "max_df": 1.0},
        {"norm": "l2", "sublinear_tf": True, "min_df": 2, "max_df": 0.9},
        {"norm": None, "sublinear_tf": False, "min_df": 3, "max_df": 1.0},
        {"norm": "None", "sublinear_tf": False, "min_df": 3, "max_df": 1.0},
    ])
def test_tfidf(param):
    embedder = TfidfEmbedder(**param)
    embedder.fit(documents)
    X = embedder.transform(documents)
    assert isinstance(X, csr_matrix)
    assert X.shape[0] == len(documents)
    check_settings(embedder, list(param))


@mark.parametrize(
    "pretrained_model",
    [
        "prithivida/bert-for-patents-64d",
        "anferico/bert-for-patents",
        "AI-Growth-Lab/PatentSBERTa"
    ]
)
def test_bert(pretrained_model):
    """ Function to test bert.py functionality
    """
    embedder = BERTEmbedder(pretrained_model)
    embedder.fit(documents)
    embeddings = embedder.transform(documents)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(documents)
    check_settings(embedder, ["pretrained_model"])


@mark.parametrize("vector_size", [100, 300])
@mark.parametrize("vocab_size", [1000, 3000])
def test_bpemb(vector_size, vocab_size):
    embedder = BPembEmbedder(vector_size=vector_size, vocab_size=vocab_size)
    embedder.fit(documents)
    X = embedder.transform(documents)
    assert X.shape == (len(documents), vector_size)
    check_settings(embedder, ["vector_size", "vocab_size"])


@mark.parametrize("method", ["sigmoid", "prop"])
def test_countvec(method):
    embedder = CountVecEmbedder(method)
    embedder.fit(documents)
    X = embedder.transform(documents)
    assert X.shape[0] == len(documents)
    assert isinstance(X, csr_matrix)
    check_settings(embedder, ["method"])
