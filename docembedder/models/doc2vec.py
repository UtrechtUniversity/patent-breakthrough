""" Gensim Doc2vec class."""

import logging
from typing import Iterable, Union, List, Optional, Dict, Any
from urllib.error import URLError

import numpy as np
from numpy import typing as npt
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import gensim
import nltk

from hyperopt import hp
from hyperopt.pyll.base import scope

from docembedder.models.base import BaseDocEmbedder


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
        self.workers = workers
        self._tagged_data: List = []
        self._d2v_model: Optional[gensim.models.doc2vec.Doc2Vec] = None

        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                            datefmt='%H:%M:%S')
        try:
            nltk.download('punkt', quiet=True)
        except URLError as exc:
            raise ValueError(
                """
                You need to install Python certificates to download files needed for
                the doc2vec model.
                See https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-"""
                """verify-failed-error.

                For a quick workaround, use:
                ssl._create_default_https_context = ssl._create_unverified_context

                However, this is not safe, so instead you are advised to install the certifi package
                and install the certificates for you Python system."""
            ) from exc

    def fit(self, documents: Iterable[str]) -> None:
        logging.info("Building Doc2Vec vocabulary:")
        self._d2v_model = gensim.models.doc2vec.Doc2Vec(
            vector_size=self.vector_size, min_count=self.min_count, epochs=self.epoch,
            workers=self.workers)

        self._tagged_data = [
            TaggedDocument(words=word_tokenize(_d.lower()),
                           tags=[str(i)]) for i, _d in enumerate(documents)]
        self._d2v_model.build_vocab(self._tagged_data)
        self._d2v_model.train(
            self._tagged_data, total_examples=self._d2v_model.corpus_count, epochs=self.epoch)

    def transform(self, documents: Union[str, Iterable[str]]) -> npt.NDArray[np.float_]:
        if self._d2v_model is None:
            raise ValueError("Error: Doc2Vec model not yet trained.")
        logging.info("Extracting Document vectors:")
        vectors = [
            self._d2v_model.infer_vector(doc_words=word_tokenize(d.lower()))
            for d in documents]

        return np.array(vectors)

    @property
    def settings(self) -> Dict[str, Any]:
        return {
            "vector_size": self.vector_size,
            "min_count": self.min_count,
            "epoch": self.epoch,
        }

    @classmethod
    def hyper_space(cls) -> Dict[str, Any]:
        """Parameter space for hyperopt."""
        return {
            "vector_size": scope.int(hp.quniform("vector_size", 100, 300, 1)),
            "min_count": scope.int(hp.quniform("min_count", 1, 15, 1)),
            "epoch": scope.int(hp.quniform("epoch", 8, 15, 1)),
        }
