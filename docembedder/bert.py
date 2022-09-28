""" Huggingface BERT class."""

from typing import Iterable, Union

import numpy as np
import numpy.typing as npt
import scipy

from sentence_transformers import SentenceTransformer

from docembedder.base import BaseDocEmbedder


class BERTEmbedder(BaseDocEmbedder):
    """BERT embedding based on Hugging face pre-trained models.
    There are a number of pre-trained models on the patent data available.
    - prithivida/bert-for-patents-64d : https://huggingface.co/prithivida/bert-for-patents-64d
    - anferico/bert-for-patents : https://huggingface.co/anferico/bert-for-patents
    - AI-Growth-Lab/PatentSBERTa : https://huggingface.co/AI-Growth-Lab/PatentSBERTa
     """

    def __init__(self,

                 pretrained_model: str = "prithivida/bert-for-patents-64d",
                 text_column: str = "contents",
                 embedding_vectors: np.ndarray = None,
                 model_path: str = "./models/document_embeddings_tst.dill"
                 ):
        self.pretrained_model = pretrained_model
        self.text_column = text_column
        self.embedding_vectors = embedding_vectors
        self._sbert_model = SentenceTransformer(pretrained_model)
        self.model_path = model_path

    def fit(self, documents: Iterable[str]) -> None:
        pass

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:
        return self._sbert_model.encode(documents)

    @property
    def embedding_size(self) -> int:
        pass
