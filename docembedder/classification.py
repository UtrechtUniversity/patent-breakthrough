"""Module containing patent classifications"""

from typing import Dict, List, Optional

import polars as pl
import numpy as np
from docembedder.typing import PathType, IntSequence


class PatentClassification():
    """Class to find similarities between patents from classifications.

    Arguments
    ---------
    classification_file: Text file (GPCPCs.txt) containing the classifications for all patents.
    similarity_exponent: Value that determines which level of classification is most important.
        Higher values mean more weight to finer levels, lower values mean more weight to more
        coarse levels. Should be between 0 and 1.
    """
    def __init__(self, classification_file: PathType, similarity_exponent=2./3.):
        self.class_df = pl.read_csv(classification_file, sep="\t")
        self.similarity_exponent = similarity_exponent
        self._lookup: Dict[int, List[str]] = {}
        self._initialized = False

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
            for jter_class, class_j in enumerate(j_patent_class):
                corr_matrix[iter_class, jter_class] = self.get_classification_similarity(class_i,
                                                                                         class_j)
        return np.mean(np.append(np.max(corr_matrix, axis=0), np.max(corr_matrix, axis=1)))

    def _get_pat_classifications(self, patent_id) -> List:
        if not self._initialized:
            self.set_patent_ids()
            self._initialized = True
        try:
            return self._lookup[patent_id]  # type: ignore
        except KeyError as exc:
            raise ValueError(f"Cannot find patent with id '{patent_id}'") from exc

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

    def set_patent_ids(self, patent_ids: Optional[IntSequence]=None) -> None:
        """Initialize the look-up table for a subset of the patent_ids.

        Arguments
        ---------
        patent_ids: Patent ID's to initialize the lookup table for.
        """
        pat_df = pl.DataFrame({"pat": patent_ids})
        query = (
            self.class_df.lazy()
            .groupby("pat")
            .agg(
                [
                    pl.col("CPC")
                ]
            )
        )
        if patent_ids is not None:
            query = query.join(pat_df.lazy(), on="pat", how="inner")
        df_filtered = query.collect()
        self._lookup = dict(zip(df_filtered["pat"], df_filtered["CPC"].to_list()))  # type: ignore

    def sample_cpc_correlations(self,  # pylint: disable=too-many-locals
                                patent_ids: IntSequence,
                                samples_per_patent: Optional[int]=None):
        """Sample/compute CPC correlations.

        Since it is costly to compute the CPC correlations between all patents O(n^2),
        this method is able to only take a few samples per patent.

        Arguments
        ---------
        patent_ids:
            Patent numbers to compute the correlations for.
        samples_per_patent:
            Number of correlation values per patent. If None, compute the full
            correlation matrix

        Returns
        -------
        cpc_correlations:
            Dictionary containing three arrays that together from the tuples (i, j, correlation).
        """
        self.set_patent_ids(patent_ids)
        index_used_mask = np.array([pid in self._lookup for pid in patent_ids])
        index_used = np.where(index_used_mask)[0]
        n_index_used = len(index_used)
        if n_index_used < 2:
            raise ValueError("Not enough patents to sample CPC correlations, need 2.")
        if samples_per_patent is None or samples_per_patent >= n_index_used-1:
            i_patents, j_patents = np.where((np.tri(len(patent_ids), k=-1).T*index_used_mask)
                                            * index_used_mask.reshape(1, -1))
        else:
            i_patent_list: List[int] = []
            j_patent_list: List[int] = []
            n_sample = min(n_index_used-1, samples_per_patent)

            rng = np.random.default_rng()
            for i_index_pat, index_pat in enumerate(index_used):
                j_pat = rng.choice(n_index_used-1, size=n_sample, replace=False)
                j_pat += (j_pat >= i_index_pat)
                j_pat = index_used[np.array(j_pat, dtype=np.int_)]
                i_pat = np.full(n_sample, index_pat)
                i_patent_list.extend(i_pat)
                j_patent_list.extend(j_pat)
            i_patents = np.array(i_patent_list)
            j_patents = np.array(j_patent_list)
        correlations = [self.get_similarity(patent_ids[i_pat], patent_ids[j_pat])
                        for i_pat, j_pat in zip(i_patents, j_patents)]
        return {
            "i_patents": i_patents,
            "j_patents": j_patents,
            "correlations": correlations,
        }
