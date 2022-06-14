"""Base class for document embeddings."""

from abc import ABC, abstractmethod
from typing import Iterable, Union

import numpy as np
import numpy.typing as npt


class BaseDocEmbedder(ABC):
    """Base class for creating document embeddings."""
    @abstractmethod
    def fit(self, documents: Iterable[str]) -> None:
        """Train the model on documents."""

    @abstractmethod
    def predict(self, document: Union[str, Iterable[str]]) -> npt.NDArray[np.float_]:
        """Get the embedding for a document."""

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Vector size of the embedding."""
