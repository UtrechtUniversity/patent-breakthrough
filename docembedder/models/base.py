"""Base class for document embeddings."""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Dict, Any

from docembedder.typing import AllEmbedType


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
    def settings(self) -> Dict[str, Any]:
        """Settings of the document embedder."""

    @classmethod
    @abstractmethod
    def hyper_space(cls) -> Dict[str, Any]:
        """Parameter space for hyperopt."""
