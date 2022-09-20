"""Sklearn TF-IDF class."""

from typing import Iterable, Union

import numpy as np
import numpy.typing as npt
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer

from docembedder.base import BaseDocEmbedder


class TfidfEmbedder(BaseDocEmbedder):
    """Sklearn TF-IDF class."""
    def __init__(self, ngram_max: int=1, stop_words: str="english"):
        self.ngram_max = ngram_max
        self.stop_words = stop_words
        self._model = TfidfVectorizer(ngram_range=(1, ngram_max),
                                      stop_words=stop_words)

    def fit(self, documents: Iterable[str]) -> None:
        self._model.fit(documents)

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:
        return self._model.transform(documents).tocsr()

    @property
    def embedding_size(self) -> int:
        return len(self._model.vocabulary_)
