"""Type Aliases for static typing."""

from pathlib import Path
from typing import Union
from collections.abc import Sequence

from typing_extensions import TypeAlias

import scipy
from numpy import typing as npt
import numpy as np

AllEmbedType: TypeAlias = Union[scipy.sparse.spmatrix, npt.NDArray[np.float_]]
PathType: TypeAlias = Union[Path, str]
IntSequence: TypeAlias = Union[Sequence[int], npt.NDArray[np.int_]]
