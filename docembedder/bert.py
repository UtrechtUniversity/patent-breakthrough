""" Huggingface BERT class."""

from typing import Iterable, Union

import numpy as np
import numpy.typing as npt

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
                 ):
        self.pretrained_model = pretrained_model
        self._sbert_model = SentenceTransformer(pretrained_model)

    def fit(self, documents: Iterable[str]) -> None:
        pass

    def transform(self, documents: Union[str, Iterable[str]]) -> npt.NDArray[np.float_]:
        return self._sbert_model.encode(documents)

    @property
    def embedding_size(self) -> int:
        pass
