from docembedder.base import BaseDocEmbedder
import gensim
from typing import Iterable, Union
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np
import numpy.typing as npt
import scipy
import pandas as pd

import logging  # Setting up the loggings to monitor gensim
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


class D2VEmbedder(BaseDocEmbedder):
    """ class for calculating Document vectors
    """
    def __init__(self, vector_size=30, min_count=2, epoch=80):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epoch = epoch
        self._d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epoch)

    def fit(self, documents: Iterable[str]) -> None:
        pass

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(documents)]
        self._d2v_model.build_vocab(tagged_data)
        self._d2v_model.train(tagged_data, total_examples=self._d2v_model.corpus_count, epochs=80)
        return self._d2v_model

    @property
    def embedding_size(self) -> int:
        return len(self._d2v_model)


if __name__ == "__main__":
    a = D2VEmbedder()

    patent_df = pd.read_csv('../data/tst_sample.csv')
    documents = patent_df['contents'].tolist()
    vectors = a.transform(documents)
    print(vectors[0])

