"""Base class for document embeddings."""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Dict, Any

import scipy
from numpy import typing as npt
import numpy as np


class BaseDocEmbedder(ABC):
    sparse = False

    """Base class for creating document embeddings."""
    @abstractmethod
    def fit(self, documents: Sequence[str]) -> None:
        """Train the model on documents."""

    @abstractmethod
    def transform(self, documents: Union[str, Sequence[str]]) -> Union[
            scipy.sparse.spmatrix, npt.NDArray[np.float_]]:
        """Get the embedding for a document."""

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Vector size of the embedding."""

    @property
    def settings(self) -> Dict[str, Any]:
        return {}
