"""Module containing patent classifications"""

from typing import Union, Sequence, Dict, List
from pathlib import Path

import polars as pl
import numpy as np


class PatentClassification():
    """Class to find similarities between patents from classifications.

    Arguments
    ---------
    classification_file: Text file (GPCPCs.txt) containing the classifications for all patents.
    similarity_exponent: Value that determines which level of classification is most important.
        Higher values mean more weight to finer levels, lower values mean more weight to more
        coarse levels. Should be between 0 and 1.
    """
    def __init__(self, classification_file: Union[str, Path], similarity_exponent=2./3.):
        self.class_df = pl.read_csv(classification_file, sep="\t")
        self.similarity_exponent = similarity_exponent
        self.lookup: Dict[str, List[int]] = {}

    def get_similarity(self, i_patent_id: int, j_patent_id: int) -> float:
        """Get the similarity between two patents.

        Arguments
        ---------
        i_patent_id: First patent ID to compare.
        j_patent_id: Second patent ID to compare.

        Returns
        -------
        Similarity score for the two patents.
        """
        if i_patent_id == j_patent_id:
            return 1.0

        i_patent_class = self._get_pat_classifications(i_patent_id)
        j_patent_class = self._get_pat_classifications(j_patent_id)
        corr_matrix = np.zeros((len(i_patent_class), len(j_patent_class)))

        for iter_class, class_i in enumerate(i_patent_class):
            # class_i = i_patent_df["CPC"].iloc[iter_class]
            for jter_class, class_j in enumerate(j_patent_class):
                # class_j = j_patent_df["CPC"].iloc[jter_class]
                corr_matrix[iter_class, jter_class] = self.get_classification_similarity(class_i,
                                                                                         class_j)
        return np.mean(np.append(np.max(corr_matrix, axis=0), np.max(corr_matrix, axis=1)))

    def _get_pat_classifications(self, patent_id) -> List:
        try:
            return self.lookup[patent_id]
        except KeyError as exc:
            raise ValueError("Cannot find patent with id '{patent_id}'") from exc

    def get_classification_similarity(self, class_a: str, class_b: str) -> float:
        """Get the similarity between two classifications.

        Arguments
        ---------
        class_a: String containing the first classification.
        class_b: String containing the second classification.

        Returns
        -------
        Similarity score between two classifications.
        """
        dissimilarity = 1.0
        for _ in range(4):
            if class_a[0] != class_b[0]:
                return 1-dissimilarity
            dissimilarity *= self.similarity_exponent
            class_a = class_a[1:]
            class_b = class_b[1:]

        left_a, right_a = class_a.split("/")
        left_b, right_b = class_b.split("/")
        if left_a != left_b:
            return 1-dissimilarity
        dissimilarity *= self.similarity_exponent
        if right_a != right_b:
            return 1-dissimilarity
        return 1

    def set_patent_ids(self, patent_ids: Sequence[int]):
        """Initialize the look-up table for a subset of the patent_ids.

        Arguments
        ---------
        patent_ids: Patent ID's to initialize the lookup table for.
        """
        pat_df = pl.DataFrame({"pat": patent_ids})
        df_filtered = (
            self.class_df.lazy()
            .groupby("pat")
            .agg(
                [
                    pl.col("CPC").list()
                ]
            )
            .join(pat_df.lazy(), on="pat")
            .collect()
        )
        self.lookup = dict(zip(df_filtered["pat"], df_filtered["CPC"].to_list()))  # type: ignore
