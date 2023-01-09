"""Sklearn TF-IDF class."""  # pylint: skip-file

from typing import Iterable, Union

import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from docembedder.base import BaseDocEmbedder


class CountVecEmbedder(BaseDocEmbedder):
    """Sklearn TF-IDF class."""
    def __init__(self, method="sigmoid"):
        self.method = method
        self._model = CountVectorizer(analyzer=str.split)

    def fit(self, documents: Iterable[str]) -> None:
        X = self._model.fit_transform(documents)
        if self.method == "prop":
            weights = 1.0 / X.sum(axis=0)
        elif self.method == "sigmoid":
            freqs = X.sum(axis=0)
            c2 = freqs.max() / 2
            c1 = 0.5
            weights = 1 - (1 / (1 + np.exp(-c1 * (freqs - c2))))
        self.weights = weights

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.spmatrix]:
        X = self._model.transform(documents)
        X = X.multiply(self.weights)
        return normalize(X, norm="l2").tocsr()

    @property
    def embedding_size(self) -> int:
        return len(self._model.vocabulary_)
