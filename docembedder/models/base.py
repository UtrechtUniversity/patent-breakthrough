"""Base class for document embeddings."""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Dict, Any, TypeAlias

import scipy
from numpy import typing as npt
import numpy as np


AllEmbedType: TypeAlias = Union[scipy.sparse.spmatrix, npt.NDArray[np.float_]]


class BaseDocEmbedder(ABC):
    sparse = False

    """Base class for creating document embeddings."""
    @abstractmethod
    def fit(self, documents: Sequence[str]) -> None:
        """Train the model on documents."""

    @abstractmethod
    def transform(self, documents: Union[str, Sequence[str]]) -> AllEmbedType:
        """Get the embedding for a document."""

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Vector size of the embedding."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings of the document embedder."""
        return {}
