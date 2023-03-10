"""Module containing Byte pair embeddings."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Union, Dict, Optional, Sequence, Any

import numpy as np
from numpy import typing as npt
from hyperopt import hp
from bpemb import BPEmb

from docembedder.models.base import BaseDocEmbedder


def _get_prefac(model: BPEmb, documents: Union[Iterable[str], Sequence[str]]) -> Dict[str, float]:
    """Compute the prefactor for each (sub)word.

    See: https://radix.ai/blog/2021/3/a-guide-to-building-document-embeddings-part-1/

    The prefactor is equal to 1/(1+1e3*p[word]), where p[word]:= the probability
    of the word occuring in a document.
    """

    # First count the number of occurences of each subword
    counts: defaultdict = defaultdict(lambda: 0)
    n_documents = 0
    for pat in documents:
        tokens = model.encode(pat)
        unq_tokens = set(tokens)
        for token in unq_tokens:
            counts[token] += 1
        n_documents += 1

    # Create a dictionary of prefactors of each subword.
    prefacs = {}
    for token, count in counts.items():
        pre_fac = 1/(1+1e3*count/n_documents)
        prefacs[token] = pre_fac
    return prefacs


class BPembEmbedder(BaseDocEmbedder):
    """Class for Byte pair embeddings.

    More information on:
    https://bpemb.h-its.org/#introduction
    https://bpemb.h-its.org/en/

    Arguments
    ---------
    vector_size:
        Vector size for the embedding. Only a few choices are available, see link above.
    vocab_size:
        Vocabularly size used to create Byte Pair Embeddings, higher should be better.
        Available values are available through the links above.
    """
    def __init__(self, vector_size: int=300, vocab_size: int=200000):
        self.vector_size = vector_size
        self.vocab_size = vocab_size
        self._prefacs: Optional[Dict[str, float]] = None
        self._model = BPEmb(lang="en", vs=self.vocab_size, dim=self.vector_size)

    def fit(self, documents: Union[Iterable[str], Sequence[str]]) -> None:
        # Compute the prefactors for each word. Similar to TF-IDF
        self._prefacs = _get_prefac(self._model, documents)

    def transform(self, documents: Union[str, Iterable[str]]) -> npt.NDArray[np.float_]:
        if self._prefacs is None:
            raise ValueError("Error Byte Pair Embedding model not fitted.")

        # Compute the base vectors for each document.
        doc_vecs_list = []
        for pat in documents:
            embedding_matrix = self._model.embed(pat)
            prefac = []
            for token in self._model.encode(pat):
                prefac.append(self._prefacs.get(token, 1))
            embedding_matrix = embedding_matrix*np.array(prefac).reshape(-1, 1)
            doc_vecs_list.append(np.sum(embedding_matrix, axis=0)/np.sum(prefac))

        # Subtract singular value, see:
        # https://radix.ai/blog/2021/3/a-guide-to-building-document-embeddings-part-1/
        doc_vecs = np.array(doc_vecs_list)
        svd_u, _, _ = np.linalg.svd(doc_vecs)
        singular_vec = svd_u[0].reshape(-1, 1)
        mult_mat = singular_vec.dot(singular_vec.T)
        doc_vecs -= mult_mat.dot(doc_vecs)
        return doc_vecs

    @property
    def settings(self) -> Dict[str, Any]:
        return {
            "vector_size": 300,
            "vocab_size": 200000,
        }

    @classmethod
    def hyper_space(cls) -> Dict[str, Any]:
        """Parameter space for hyperopt."""
        return {
            "vector_size": hp.choice("vector_size", [25, 50, 100, 200, 300]),
            "vocab_size": hp.choice("vocab_size",
                                    [1000, 3000, 5000, 10000, 25000, 50000, 100000, 200000]),
        }
