"""Sklearn TF-IDF class."""

from typing import Sequence, Union, Optional, Callable, Dict, Any
import warnings

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import SnowballStemmer

from hyperopt import hp

from docembedder.models.base import BaseDocEmbedder


def _tokenizer(text):
    tokens = nltk.word_tokenize(text)
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(item) for item in tokens]


class TfidfEmbedder(BaseDocEmbedder):  # pylint: disable=too-many-instance-attributes
    """Sklearn TF-IDF class.

    Based on:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    Arguments
    ---------
    ngram_max:
        Maximum n-gram, higher numbers mean bigger embeddings.
    stop_words:
        Remove stop words from a certain langauge.
    stem:
        Whether to stem the words. Has a negative impact on performance.
    norm:
        Which kind of normalization is used: "l1", "l2" or None.
    sublinear_tf:
        Apply sublinear term-frequency scaling.
    min_df:
        Minimum document frequency of word to be included in the embedding.
    max_df:
        Maximum document frequency of word to be included in the embedding.
    """
    sparse = True

    def __init__(  # pylint: disable=too-many-arguments
            self, ngram_max: int=1,
            stop_words: Optional[str]="english",
            stem: bool=False,
            norm: Optional[str]="l1",
            sublinear_tf: bool=False,
            min_df: int=3,
            max_df: float=1.0):
        self.ngram_max = ngram_max
        self.stop_words = stop_words
        self.norm = norm
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df
        self.stem = stem
        self.stem_tokenizer: Optional[Callable] = None
        if stem:
            self.stem_tokenizer = _tokenizer

        if self.norm == "None":
            self.norm = None
        if self.stop_words == "None":
            self.stop_words = None
        self._model: Optional[TfidfVectorizer] = None

    def fit(self, documents: Sequence[str]) -> None:
        min_df = min(self.min_df, len(documents))
        max_df = max(min_df/len(documents), self.max_df)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._model = TfidfVectorizer(
                ngram_range=(1, self.ngram_max),
                stop_words=self.stop_words,
                tokenizer=self.stem_tokenizer,
                min_df=min_df,
                norm=self.norm,
                sublinear_tf=self.sublinear_tf,
                max_df=max_df)
            self._model.fit(documents)

    def transform(self, documents: Union[str, Sequence[str]]) -> Union[
            scipy.sparse.spmatrix]:
        if self._model is None:
            raise ValueError("Fit TF-IDF model before transforming data.")
        return self._model.transform(documents).tocsr()

    @property
    def settings(self) -> Dict[str, Any]:
        return {
            "ngram_max": self.ngram_max,
            "stop_words": str(self.stop_words),
            "stem": self.stem,
            "norm": str(self.norm),
            "sublinear_tf": self.sublinear_tf,
            "min_df": self.min_df,
            "max_df": self.max_df,
        }

    @classmethod
    def hyper_space(cls) -> Dict[str, Any]:
        """Parameter space for hyperopt."""

        return {
            "ngram_max": hp.choice("ngram_max", [1, 2, 3]),
            "stop_words": hp.choice("stop_words", ["english", None]),
            "stem": hp.choice("stem", [True, False]),
            "norm": hp.choice("norm", ["l1", "l2", None]),
            "sublinear_tf": hp.choice("sublinear_tf", [True, False]),
            "min_df": 1 + hp.randint("min_df", 0, 9),
            "max_df": hp.uniform("max_df", 0.2, 1)
        }
