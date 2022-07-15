from typing import Union
from pathlib import Path

import pandas as pd
import numpy as np


class PatentClassification():
    def __init__(self, fp: Union[str, Path], similarity_exponent=2./3.):
        self.df = pd.read_csv(fp, delimiter="\t")
        self.by_patent = self.df.groupby("pat")
        self.similarity_exponent = similarity_exponent

    def get_correlation(self, i_patent_id: int, j_patent_id: int) -> float:
        if i_patent_id == j_patent_id:
            return 1.0

        try:
            i_patent_df = self.by_patent.get_group(i_patent_id)
        except KeyError as exc:
            raise ValueError(f"Cannot find patent with id '{i_patent_id}'") from exc

        try:
            j_patent_df = self.by_patent.get_group(j_patent_id)
        except KeyError as exc:
            raise ValueError(f"Cannot find patent with id '{j_patent_id}'") from exc
        corr_matrix = np.zeros((len(i_patent_df), len(j_patent_df)))

        for iter_class in range(len(i_patent_df)):
            class_i = i_patent_df["CPC"].iloc[iter_class]
            for jter_class in range(len(j_patent_df)):
                class_j = j_patent_df["CPC"].iloc[jter_class]
                corr_matrix[iter_class, jter_class] = self._get_classification_correlation(class_i,
                                                                                           class_j)
        return np.mean(np.append(np.max(corr_matrix, axis=0), np.max(corr_matrix, axis=1)))

    def _get_classification_correlation(self, class_a: str, class_b: str) -> float:
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
