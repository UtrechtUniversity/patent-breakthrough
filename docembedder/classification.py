"""Module containing patent classifications"""

from __future__ import annotations

import multiprocessing
import re
from typing import Dict, List, Optional, Any

import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm

from docembedder.typing import PathType, IntSequence
from docembedder.simspec import SimulationSpecification
from docembedder.embedding_utils import _gather_results


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
        self.class_df = pl.read_csv(classification_file, separator="\t")
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
            query = query.join(pat_df.lazy(), on="pat", how="inner")  # type: ignore
        df_filtered = query.collect()  # type: ignore
        self._lookup = dict(zip(df_filtered["pat"], df_filtered["CPC"].to_list()))  # type: ignore

    def sample_cpc_correlations(self,  # pylint: disable=too-many-locals
                                patent_ids: IntSequence,
                                samples_per_patent: Optional[int]=None,
                                seed=None):
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
        seed:
            Seed used in case of sampling.

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

            rng = np.random.default_rng(seed=seed)
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
            "correlations": np.array(correlations),
        }


similarity_levels = 1-(2./3.)**np.arange(6)


def _cpc_inproduct(vec, cpc_vectors):
    inprod = np.empty(cpc_vectors.shape[0], dtype=float)
    indices = np.arange(cpc_vectors.shape[0])

    for i in range(0, cpc_vectors.shape[1]):
        disable = (vec[i] != cpc_vectors[indices, i])
        inprod[indices[disable]] = similarity_levels[i]
        indices = indices[~disable]
    inprod[indices] = 1
    return inprod


def _max_avg_inproduct(cpc_vectors, pat_ids, idx_back, idx_focal):
    n_focal = len(idx_focal)
    n_back = len(idx_back)
    inprod_cache = np.empty((n_focal, n_back), dtype=float)
    for i_focal in range(n_focal):
        inprod_cache[i_focal] = _cpc_inproduct(cpc_vectors[idx_focal[i_focal]],
                                               cpc_vectors[idx_back])

    pat_lines = np.where(pat_ids[idx_back[1:]]-pat_ids[idx_back[:-1]] > 0)[0] + 1
    max_inprod_cache = np.empty((n_focal, len(np.unique(pat_ids[idx_back]))), dtype=float)
    prev_idx = 0
    for i_dest, idx in enumerate(pat_lines):
        max_inprod_cache[:, i_dest] = np.max(inprod_cache[:, prev_idx:idx], axis=1)
        prev_idx = idx

    max_inprod_cache[:, -1] = np.max(inprod_cache[:, prev_idx:], axis=1)

    avg_inprod_cache = np.empty((len(np.unique(pat_ids[idx_focal])), max_inprod_cache.shape[1]),
                                dtype=float)

    pat_lines = np.where(pat_ids[idx_focal[1:]] - pat_ids[idx_focal[:-1]] > 0)[0] + 1
    prev_idx = 0
    for i_dest, idx in enumerate(pat_lines):
        avg_inprod_cache[i_dest] = np.mean(max_inprod_cache[prev_idx:idx], axis=0)
        prev_idx = idx
    avg_inprod_cache[-1] = np.mean(max_inprod_cache[prev_idx:], axis=0)
    return avg_inprod_cache


def _vectorize_classification(cpc_class, regex):
    """Convert the CPC class to a vector using a regex (reused)."""
    vector = np.empty(6, dtype=int)
    mat = regex.match(cpc_class)
    vector[0] = ord(mat.group(1))
    vector[1] = mat.group(2)
    vector[2] = mat.group(3)
    vector[3] = ord(mat.group(4))
    vector[4] = mat.group(5)
    vector[5] = mat.group(6)
    return vector


def get_cpc_data(year_fp: PathType, cpc_fp: PathType,
                 progress_bar: bool=True) -> dict[str, Any]:
    """Get CPC data for usage in determining novelty and impact.

    Arguments
    ---------
    year_fp:
        Data file that contains the year of issue for each patent.
    cpc_fp:
        Data file containing the CPC classifications for each patent.
    progress_bar:
        Whether to enable a progress bar.

    Returns
    -------
    cpc_data:
        Dictionary with numpy arrays containing "year", "pat_ids", and vectorized
        cpc classifications "cpc_vectors".
    """
    year_df = pd.read_csv(year_fp, sep="\t")
    pat_class = PatentClassification(cpc_fp)
    combined_df = pat_class.class_df.to_pandas().merge(year_df, on="pat", how="left")
    combined_df = combined_df.dropna()
    combined_df = combined_df.astype({"year": int}).sort_values(by=["year", "pat", "progr"])

    # Create a matrix that has all the vectorized versions of CPC codes
    regex = re.compile(r"([A-Z])(\d)(\d)([A-Z])(\d+)\/(\d+)")
    all_vectors = np.empty((len(pat_class.class_df), 6), dtype=int)
    for i_class, classification in enumerate(tqdm(combined_df["CPC"].values,
                                                  disable=not progress_bar)):
        all_vectors[i_class] = _vectorize_classification(classification, regex)

    return {"year": combined_df["year"].values,
            "pat_ids": combined_df["pat"].values,
            "cpc_vectors": all_vectors}


def _separate_focal_idx(idx_focal, pat_ids, max_forw_back, max_mat_size):
    # Seperate focal indices so that we can use parallel processing
    # but also to use less memory at the same time.
    n_blocks = round(max_forw_back*len(idx_focal)/max_mat_size)
    if n_blocks <= 1:
        return [idx_focal]

    blocks = []
    start_block = 0
    for i_block in range(n_blocks-1):
        # Partition it into equal parts
        end_block = round(((i_block+1)/n_blocks) * len(idx_focal))
        # We must ensure that patent ids are kept together.
        while pat_ids[idx_focal[end_block]] == pat_ids[idx_focal[end_block-1]]:
            end_block -= 1
        blocks.append(idx_focal[start_block:end_block])
        start_block = end_block

    # Add the last block
    blocks.append(idx_focal[start_block:])
    return blocks


def _compute_similarity(job: tuple):
    # Compute the impact and novelty for the provided focal indices.
    cpc_vectors, pat_ids, idx_back, idx_forw, idx_focal, exponents = job
    inproduct_back = _max_avg_inproduct(cpc_vectors, pat_ids, idx_back, idx_focal)
    inproduct_forw = _max_avg_inproduct(cpc_vectors, pat_ids, idx_forw, idx_focal)

    results = []
    for expon in exponents:
        similarity_back = np.mean(np.array(inproduct_back)**expon, axis=1)**(1/expon)
        similarity_forw = np.mean(np.array(inproduct_forw)**expon, axis=1)**(1/expon)
        results.append({
            "novelty": similarity_back,
            "impact": similarity_forw/(similarity_back+1e-12),
            "patent_ids": np.unique(pat_ids[idx_focal]),
            "exponent": expon,
        })

    return results


def cpc_nov_impact(cpc_data: dict[str, Any],  # pylint: disable=too-many-locals
                   sim_spec: SimulationSpecification,
                   exponents: list[float],
                   max_mat_size: int=100000000,
                   n_jobs: int=10):
    """Compute the novelty and impact using CPC codes.

    Parameters
    ----------
    cpc_data:
        Dictionary with the data containing the CPC codes, patent ids and year of issue.
    sim_spec:
        Simulation specification that determines which years/window sizes are used.
    exponents:
        List of exponents for the kind of averaging that is being used.
        Generally we use (sum_ij (x_i*x_j)**exponent)**(1/exponent), where x_i*x_j is the
        similarity between two classifications of patent i and j.
    max_mat_size:
        Determines how the problem is split up, preventing using too much memory and enabling
        parallel processing. Higher numbers mean using more memory, but potentially being more
        efficient. Take care not to make it too small. A size of 1e7 seems to use about 2GB per
        process.
    n_jobs:
        Number of parallel jobs.
    """
    year, pat_ids, cpc_vectors = cpc_data["year"], cpc_data["pat_ids"], cpc_data["cpc_vectors"]
    all_results = []

    for all_years in sim_spec.year_ranges:
        start_year = min(all_years)
        end_year = max(all_years)+1
        focal_year = (end_year - 1 + start_year)//2

        # Get the forward/backward/focal year CPC codes (not patent_ids)
        idx_back = np.where(year < focal_year)[0]
        idx_forw = np.where((year > focal_year) &
                            (year < end_year))[0]
        idx_focal = np.where(year == focal_year)[0]

        # Split the problem in multiple pieces,
        # The split will seperate the focal patents into multiple groups.
        max_forw_back = max(len(idx_back), len(idx_forw))
        split_idx_focal = _separate_focal_idx(idx_focal, pat_ids, max_forw_back, max_mat_size)

        # Create the jobs
        jobs = [(cpc_vectors, pat_ids, idx_back, idx_forw, sub_idx_focal, exponents)
                for sub_idx_focal in split_idx_focal]

        # Use multiprocessing pooling to create the results.
        with multiprocessing.get_context('spawn').Pool(processes=n_jobs) as pool:
            for data_part in tqdm(pool.imap_unordered(_compute_similarity, jobs), total=len(jobs)):
                all_results.extend(data_part)

    # Gather/reorganize the results before returning them.
    return _gather_results(all_results)
