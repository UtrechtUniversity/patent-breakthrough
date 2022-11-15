from pathlib import Path
from numpy import savez_compressed
import numpy as np


class FileIO:
    """Class for formatting/reading and writing datafiles"""

    def __init__(self):
        self.embeddings_dir = Path(".", "data/processed/embeddings")

    def save_embeddings(self,
                        file_name,
                        train_id,
                        patent_id,
                        embeddings,
                        preprocessing_setting,
                        model_setting,
                        version
                        ):

        savez_compressed(self.embeddings_dir/file_name,
                         train_id=train_id,
                         patent_id=patent_id,
                         embeddings=embeddings,
                         preprocessing_setting=preprocessing_setting,
                         model_setting=model_setting,
                         version=version
                         )

    def load_embeddings(self, file_name):
        return np.load(self.embeddings_dir/file_name)







