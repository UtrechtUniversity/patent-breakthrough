"""Utilities for working with embeddings."""

from enum import IntEnum

import numpy as np
from docembedder.utils import np_save
from docembedder.typing import AllEmbedType, FileType
from scipy.sparse import csr_matrix


class EmbedType(IntEnum):
    ARRAY = 1
    CSR_MATRIX = 2


def store_embeddings(embeddings: AllEmbedType,
                     embedding_fp: FileType):
    """Store embeddings for a window/year.

    Arguments
    ---------
    window_name:
        Year or window name.
    model_name:
        Name of the model that generated the embeddings.
    overwrite:
        If True, overwrite embeddings if they exist.
    """
    if not isinstance(embeddings, (np.ndarray, csr_matrix)):
        raise ValueError(f"Not implemented datatype {type(embeddings)}")

    if isinstance(embeddings, np.ndarray):
        np_save(embedding_fp, EmbedType.ARRAY, embeddings)
    elif isinstance(embeddings, csr_matrix):
        np_save(embedding_fp, EmbedType.CSR_MATRIX, embeddings.data, embeddings.indices,
                embeddings.indptr, embeddings.shape)


def load_embeddings(embedding_fp):
    with open(embedding_fp, "rb") as handle:
        embedding_type = np.load(handle)
        if embedding_type == EmbedType.ARRAY:
            return np.load(handle)
        if embedding_type == EmbedType.CSR_MATRIX:
            data = np.load(handle)
            indices = np.load(handle)
            indptr = np.load(handle)
            shape = np.load(handle)
        return csr_matrix((data, indices, indptr), shape=shape)
    raise ValueError(f"Unknown embedding type detected in file {embedding_fp}")
