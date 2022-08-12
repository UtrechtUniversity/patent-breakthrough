""" Huggingface BERT class."""

from typing import Iterable, Union

from pathlib import Path

import numpy as np
import numpy.typing as npt
import scipy
import dill

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
                 pretrained_model: str = "sentence-transformers/stsb-distilbert-base",
                 text_column: str = "contents"):
        self.pretraiend_model = pretrained_model
        self.text_column = text_column
        self._sbert_model = SentenceTransformer(pretrained_model)

    def fit(self, documents: Iterable[str]) -> None:
        embedding_vectors = self._sbert_model.encode(documents)

        path_model = Path(__file__).parent / "../models/document_embeddings.model"
        with open(path_model, 'wb') as file:
            dill.dump(embedding_vectors, file)

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:
        pass

    @property
    def embedding_size(self) -> int:
        pass
