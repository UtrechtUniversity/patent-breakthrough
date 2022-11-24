"""Sklearn TF-IDF class."""

from typing import Iterable, Union, Optional

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import SnowballStemmer

from docembedder.base import BaseDocEmbedder


def _tokenizer(text):
    tokens = nltk.word_tokenize(text)
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(item) for item in tokens]


class TfidfEmbedder(BaseDocEmbedder):  # pylint: disable=too-many-instance-attributes
    """Sklearn TF-IDF class."""
    def __init__(  # pylint: disable=too-many-arguments
            self, ngram_max: int=1, stop_words: str="english", stem=False, min_df=3,
            norm="l1", sublinear_tf=False, max_df=1.0):
        self.ngram_max = ngram_max
        self.stop_words = stop_words
        self.min_df = min_df
        self.norm = norm
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        if stem:
            self.stem_tokenizer = _tokenizer
        else:
            self.stem_tokenizer = None

        self._model: Optional[TfidfVectorizer] = None

    def fit(self, documents: Iterable[str]) -> None:
        min_df = min(self.min_df, len(documents))
        self._model = TfidfVectorizer(
            ngram_range=(1, self.ngram_max),
            stop_words=self.stop_words,
            tokenizer=self.stem_tokenizer,
            min_df=min_df,
            norm=self.norm,
            sublinear_tf=self.sublinear_tf,
            max_df=self.max_df)
        self._model.fit(documents)

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.spmatrix]:
        return self._model.transform(documents).tocsr()

    @property
    def embedding_size(self) -> int:
        return len(self._model.vocabulary_)
