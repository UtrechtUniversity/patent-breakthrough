""" Gensim Doc2vec class."""

import logging
from typing import Iterable, Union, List

import numpy.typing as npt
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import gensim
import nltk

from docembedder.base import BaseDocEmbedder


class D2VEmbedder(BaseDocEmbedder):
    """ Doc2Vec
    class for representing each Document as a Vector using Genism Doc2Vec model

    useful links:
    https://radimrehurek.com/gensim/models/doc2vec.html
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py


    Arguments
    ---------
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
        nltk.download('punkt')

        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                            datefmt='%H:%M:%S', level=logging.INFO)

        self._d2v_model = gensim.models.doc2vec.Doc2Vec(
            vector_size=vector_size, min_count=min_count, epochs=epoch, workers=workers)

    def fit(self, documents: Iterable[str]) -> None:
        logging.info("Building Doc2Vec vocabulary:")
        self._tagged_data = [
            TaggedDocument(words=word_tokenize(_d.lower()),
                           tags=[str(i)]) for i, _d in enumerate(documents)]
        self._d2v_model.build_vocab(self._tagged_data)
        self._d2v_model.train(
            self._tagged_data, total_examples=self._d2v_model.corpus_count, epochs=self.epoch)

    def transform(self, documents: Union[str, Iterable[str]]) -> npt.NDArray[np.float_]:
        logging.info("Extracting Document vectors:")
        vectors = [
            self._d2v_model.infer_vector(
                doc_words=word_tokenize(_d.lower())) for i, _d in enumerate(documents)]

        return vectors

    @property
    def embedding_size(self) -> int:
        return self.vector_size
