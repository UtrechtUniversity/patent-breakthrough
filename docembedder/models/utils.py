"""Module for creating models from dictionaries."""

from docembedder.models.base import BaseDocEmbedder
from docembedder.models.bert import BERTEmbedder
from docembedder.models.tfidf import TfidfEmbedder
from docembedder.models.doc2vec import D2VEmbedder
from docembedder.models.bpemb import BPembEmbedder
from docembedder.models.countvec import CountVecEmbedder


def create_model(model_type: str, **kwargs) -> BaseDocEmbedder:
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
    return model_class(**kwargs)
