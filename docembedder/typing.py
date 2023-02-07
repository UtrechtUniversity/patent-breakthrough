"""Type Aliases for static typing."""

from __future__ import annotations
import io
from pathlib import Path
from typing import Union, Sequence

from typing_extensions import TypeAlias

import scipy
from numpy import typing as npt
import numpy as np

AllEmbedType: TypeAlias = Union[scipy.sparse.spmatrix, npt.NDArray[np.float_]]
PathType: TypeAlias = Union[Path, str]
IntSequence: TypeAlias = Union[Sequence[int], npt.NDArray[np.int_]]
FileType: TypeAlias = Union[PathType, io.BytesIO]
