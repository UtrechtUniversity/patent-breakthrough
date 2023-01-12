"""Base class for document embeddings."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Sequence, Dict, Any
from typing_extensions import TypeAlias

import scipy
from numpy import typing as npt
import numpy as np


AllEmbedType: TypeAlias = Union[scipy.sparse.spmatrix, npt.NDArray[np.float_]]
PathType: TypeAlias = Union[Path, str]


class BaseDocEmbedder(ABC):
    """Base class for creating document embeddings."""
    sparse = False

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
    @abstractmethod
    def settings(self) -> Dict[str, Any]:
        """Settings of the document embedder."""
        return {}
