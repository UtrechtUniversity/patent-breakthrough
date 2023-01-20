"""Module for creating models from dictionaries."""

from __future__ import annotations

from docembedder.models.base import BaseDocEmbedder
from docembedder.models.bert import BERTEmbedder
from docembedder.models.tfidf import TfidfEmbedder
from docembedder.models.doc2vec import D2VEmbedder
from docembedder.models.bpemb import BPembEmbedder
from docembedder.models.countvec import CountVecEmbedder
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.preprocessor.oldprep import OldPreprocessor


def create_model(model_type, model_dict: dict) -> BaseDocEmbedder:
    """Create a model from a dictionary.

    Arguments
    ---------
    model_type:
        Name of the class of the model.
    kwargs:
        Keyword arguments for the model initialization.

    Returns
    -------
    model:
        Initialized (but not trained) model.
    """
    all_models = [BERTEmbedder, TfidfEmbedder, D2VEmbedder, BPembEmbedder,
                  CountVecEmbedder]

    try:
        model_class = [x for x in all_models if x.__name__ == model_type][0]
    except IndexError as error:
        raise ValueError(f"Unknown model type: {model_type}.") from error
    model = model_class(**model_dict)

    return model


def create_preprocessor(prep_type, prep_dict: dict) -> Preprocessor:
    """Create a preprocessor from a dictionary.

    Arguments
    ---------
    prep_type:
        Name of the class of the preprocessor.
    kwargs:
        Keyword arguments for the model initialization.

    Returns
    -------
    preprocessor:
        Initialized preprocessor.
    """
    if prep_type == "OldPreprocessor":
        prep_model: Preprocessor = OldPreprocessor(**prep_dict)
    else:
        prep_model = Preprocessor(**prep_dict)
    return prep_model
