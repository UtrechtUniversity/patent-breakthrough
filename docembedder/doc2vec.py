""" Gensim Doc2vec class."""

import ssl
import logging
from typing import Iterable, Union, List

import nltk
import numpy.typing as npt
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import gensim
import scipy
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
    """ Doc2Vec
    class for representing each Document as a Vector using Genism Doc2Vec model

    useful links:
    https://radimrehurek.com/gensim/models/doc2vec.html
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py


    Arguments
    ----------
    vector_size: Dimensionality of the feature vectors
    min_count: Ignores all words with total frequency lower than this
    epoch: Number of iterations (epochs) over the corpus
    workers: Number of worker threads to train the model (faster training with multicore machines)
    """
    def __init__(self,
                 vector_size: int = 100,
                 min_count: int = 2,
                 epoch: int = 10,
                 workers: int = 4):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epoch = epoch
        self.workers = 4
        self._tagged_data: List = []

        self._d2v_model = gensim.models.doc2vec.Doc2Vec(
            vector_size=vector_size, min_count=min_count, epochs=epoch, workers=workers)

    def fit(self, documents: Iterable[str]) -> None:
        logging.info("Building Doc2Vec vocabulary:")
        self._tagged_data = [
            TaggedDocument(words=word_tokenize(_d.lower()),
                           tags=[str(i)]) for i, _d in enumerate(documents)]
        self._d2v_model.build_vocab(self._tagged_data)

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:

        pass

    def train(self) -> None:
        """ Train on tagged_documents"""
        logging.info("Training Doc2Vec model:")
        self._d2v_model.train(
            self._tagged_data, total_examples=self._d2v_model.corpus_count, epochs=self.epoch)

    def get_vectors(self, corpus_size: int) -> npt.NDArray:
        """
        Get vectors from trained doc2vec model

        arguments
        ---------
        corpus_size: Size of the documents

        return
        ------
         Document vectors
        """
        vectors = np.zeros((corpus_size, self.vector_size))
        for i in range(0, corpus_size):
            vectors[i] = self._d2v_model.dv[i]
        return vectors

    @property
    def embedding_size(self) -> int:
        return self.vector_size
