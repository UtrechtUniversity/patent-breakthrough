""" Huggingface BERT class."""

from typing import Iterable, Union, Dict, Any, Optional

import numpy as np
import numpy.typing as npt

from sentence_transformers import SentenceTransformer

from hyperopt import hp

from docembedder.models.base import BaseDocEmbedder


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
        self._sbert_model: Optional[SentenceTransformer] = None

    def fit(self, documents: Iterable[str]) -> None:
        pass

    def transform(self, documents: Union[str, Iterable[str]]) -> npt.NDArray[np.float_]:
        if self._sbert_model is None:
            self._sbert_model = SentenceTransformer(self.pretrained_model)

        return self._sbert_model.encode(documents)

    @property
    def settings(self) -> Dict[str, Any]:
        return {
            "pretrained_model": self.pretrained_model
        }

    @classmethod
    def hyper_space(cls) -> Dict[str, Any]:
        """Parameter space for hyperopt."""
        return {
            "pretrained_model": hp.choice("pretrained_model",
                                          ["prithivida/bert-for-patents-64d",
                                           "anferico/bert-for-patents",
                                           "AI-Growth-Lab/PatentSBERTa"]),
        }
