""" Gensim Doc2vec class."""

import ssl
import logging
from typing import Iterable, Union

import nltk
import pandas as pd
import scipy
import numpy.typing as npt
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import gensim

from docembedder.base import BaseDocEmbedder


try:
    _create_unverified_https_context = ssl._create_unverified_context # pylint: disable=W0212
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context # pylint: disable=W0212

nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt='%H:%M:%S', level=logging.INFO)


class D2VEmbedder(BaseDocEmbedder):
    """ class for calculating Document vectors
    """
    def __init__(self, vector_size=100, min_count=2, epoch=10, workers=4):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epoch = epoch
        self.workers = 4
        self._tagged_data = []

        self._d2v_model = gensim.models.doc2vec.Doc2Vec(
            vector_size=vector_size, min_count=min_count, epochs=epoch, workers=workers)

    def fit(self, documents: Iterable[str]) -> None:
        self._tagged_data = [
            TaggedDocument(words=word_tokenize(_d.lower()),
                           tags=[str(i)]) for i, _d in enumerate(documents)]
        self._d2v_model.build_vocab(self._tagged_data)

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:

        self._d2v_model.train(
            self._tagged_data, total_examples=self._d2v_model.corpus_count, epochs=self.epoch)
        return self._d2v_model

    @property
    def embedding_size(self) -> int:
        pass


if __name__ == "__main__":
    a = D2VEmbedder()

    patent_df = pd.read_csv('../data/tst_sample.csv')
    doc = patent_df['contents'].tolist()
    vectors = a.transform(doc)
    print(vectors)
