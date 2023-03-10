"""Analysis functions and classes for embedding results."""

from __future__ import annotations
from collections import defaultdict
from typing import List, Union, Dict, Any, Optional, DefaultDict

import numpy as np
from numpy import typing as npt
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from docembedder.datamodel import DataModel
from docembedder.models.base import AllEmbedType


def _compute_cpc_cor(embeddings: AllEmbedType,
                     cpc_res: Dict[str, Any],
                     chunk_size: int=10000) -> float:
    """Compute correlation for a set of embeddings."""
    n_split = max(1, len(cpc_res["correlations"]) // chunk_size)
    i_pat_split = np.array_split(cpc_res["i_patents"], n_split)
    j_pat_split = np.array_split(cpc_res["j_patents"], n_split)

    model_cor: List[float] = []

    for i_split in range(n_split):
        i_embeddings = embeddings[i_pat_split[i_split]]
        j_embeddings = embeddings[j_pat_split[i_split]]
        if isinstance(i_embeddings, np.ndarray):
            mcor = np.sum(i_embeddings*j_embeddings, axis=1)
        elif isinstance(i_embeddings, csr_matrix):
            mul_cor = i_embeddings.multiply(j_embeddings)
            mcor = np.asarray(mul_cor.sum(axis=1)).reshape(-1)
        else:
            raise ValueError(f"Unsupported embeddings type: {type(i_embeddings)}")
        model_cor.extend(mcor)

    return spearmanr(model_cor, cpc_res["correlations"]).correlation


def _auto_cor(delta, embeddings):
    start = delta
    end = embeddings.shape[0] - delta
    if isinstance(embeddings, np.ndarray):
        return (embeddings[:end]*embeddings[start:]).sum(axis=1).flatten().mean()
    return np.array(embeddings[:end].multiply(embeddings[start:]).sum(axis=1)).flatten().mean()


class DocAnalysis():  # pylint: disable=too-few-public-methods
    """Analysis class that can analyse embeddings.

    Arguments
    ---------
    data: Data to analyze (class that handles hdf5 files).
    """
    def __init__(self, data: DataModel):
        self.data = data

    def _compute_impact_novelty(self, window_name, model_name):  # pylint: disable=too-many-locals
        patent_ids, patent_years = self.data.load_window(window_name)
        embeddings = self.data.load_embeddings(window_name, model_name)
        patent_indices = np.array(range(len(patent_ids)))
        min_year = np.amin(patent_years)
        max_year = np.amax(patent_years)
        focal_year = int((min_year+max_year)/2)

        impact_arr: npt.NDArray[np.float_] = np.full(len(patent_years), np.nan)
        novelty_arr: npt.NDArray[np.float_] = np.full(len(patent_years), np.nan)

        for cur_index in range(len(patent_ids)):
            if patent_years[cur_index] != focal_year:
                continue
            cur_embedding = embeddings[cur_index]
            if len(cur_embedding.shape) == 1:
                cur_embedding = cur_embedding.reshape(1, -1)
            cur_year = patent_years[cur_index]

            other_indices = np.delete(patent_indices, cur_index)
            other_embeddings = embeddings[other_indices, :]
            other_years = np.delete(patent_years, cur_index)

            embeddings_backward = other_embeddings[other_years < cur_year]
            embeddings_forward = other_embeddings[other_years > cur_year]

            if embeddings_backward.size:
                backward_similarity = cosine_similarity(cur_embedding, embeddings_backward)
                backward_dissimilarity = 1 - cosine_similarity(cur_embedding, embeddings_backward)
            else:
                backward_similarity = np.nan
                backward_dissimilarity = np.nan

            average_backward_similarity = np.mean(backward_similarity)
            average_backward_dissimilarity = np.mean(backward_dissimilarity)

            novelty_arr[cur_index] = average_backward_dissimilarity

            if embeddings_forward.size:
                forward_similarity = cosine_similarity(cur_embedding, embeddings_forward)
            else:
                forward_similarity = np.nan
            average_forward_similarity = np.mean(forward_similarity)

            if average_forward_similarity and average_backward_similarity:
                impact_arr[cur_index] = average_backward_similarity / average_forward_similarity

        impact_arr = impact_arr[~np.isnan(impact_arr)]
        novelty_arr = novelty_arr[~np.isnan(novelty_arr)]
        return impact_arr, novelty_arr, focal_year

    def auto_correlation(self, window_name, model_name):
        """Compute autocorrelations for embeddings."""
        embeddings = normalize(self.data.load_embeddings(window_name, model_name))
        patent_ids, _ = self.data.load_window(window_name)
        delta_count = np.unique((1+0.3*np.arange(5000)**2+0.5).astype(int))
        delta_count = delta_count[delta_count < len(patent_ids)]
        delta_year = 4*delta_count/len(patent_ids)
        auto_correlations = np.array([_auto_cor(i, embeddings) for i in delta_count])
        return delta_year, auto_correlations

    def patent_impacts(self, window_name, model_name):
        """Compute impact using cosine similarity between document vectors
        """
        try:
            impacts = self.data.load_impacts(window_name, model_name)
        except KeyError:
            impacts, novelties, focal_year = self._compute_impact_novelty(window_name, model_name)
            self.data.store_impact_novelty(window_name, model_name, focal_year, impacts, novelties)
        return impacts

    def patent_novelties(self, window_name, model_name):
        """Compute novelty using cosine similarity between document vectors
        """
        try:
            novelties = self.data.load_novelties(window_name, model_name)
        except KeyError:
            impacts, novelties, focal_year = self._compute_impact_novelty(window_name, model_name)
            self.data.store_impact_novelty(window_name, model_name, focal_year, impacts, novelties)
        return novelties

    def cpc_correlations(self, models: Optional[Union[str, List[str]]]=None
                         ) -> Dict[str, Dict[str, Any]]:
        """Compute the correlations with the CPC classifications.

        It computes the correlations for each window/year in which the embeddings
        are trained on the same patents.

        Argumentssi
        ---------
        models: Model names to use for computation. If None, use all models available.

        Returns
        -------
        results: Tuple with the average years of the windows and a dictionary
                 containing the correlations for each of the models.
        """
        if models is None:
            models = self.data.model_names
        elif isinstance(models, str):
            models = [models]
        elif not isinstance(models, list):
            raise TypeError("models argument must be a string or a list of strings.")

        correlations: DefaultDict[str, Dict[str, Any]] = defaultdict(
            lambda: {"year": [], "correlations": []})

        for window, model_name in self.data.iterate_window_models():
            if model_name not in models:
                continue
            try:
                correlation = self.data.load_cpc_spearmanr(window, model_name)
            except KeyError:
                embeddings = self.data.load_embeddings(window, model_name)
                cpc_cor = self.data.load_cpc_correlations(window)
                correlation = _compute_cpc_cor(embeddings, cpc_cor)
                if not self.data.read_only:
                    self.data.store_cpc_spearmanr(window, model_name, correlation)
            try:
                year: Union[float, str] = float(window)
            except ValueError:
                year_list = window.split("-")
                if len(year_list) == 2:
                    year = float(np.mean([float(x) for x in year_list]))
                else:
                    year = window

            correlations[model_name]["year"].append(year)
            correlations[model_name]["correlations"].append(correlation)
        return dict(correlations)
