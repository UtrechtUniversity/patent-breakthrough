"""Document embedding package."""

from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.analysis import DOCSimilarity
from docembedder.preprocessor.parser import read_xz
from docembedder.datamodel import DataModel
from docembedder.models.base import AllEmbedType

from . import _version
__version__ = _version.get_versions()['version']
