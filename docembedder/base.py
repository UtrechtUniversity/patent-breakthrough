"""Base class for document embeddings."""

from abc import ABC, abstractmethod
from typing import Iterable, Union

import scipy
from numpy import typing as npt
import numpy as np


class BaseDocEmbedder(ABC):
    """Base class for creating document embeddings."""
    @abstractmethod
    def fit(self, documents: Iterable[str]) -> None:
        """Train the model on documents."""

    @abstractmethod
    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:
        """Get the embedding for a document."""

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Vector size of the embedding."""
