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


def _auto_cor(delta: int, embeddings: AllEmbedType):
    start = delta
    end = embeddings.shape[0] - delta
    if isinstance(embeddings, np.ndarray):
        return (embeddings[:end]*embeddings[start:]).sum(axis=1).flatten().mean()
    return np.array(embeddings[:end].multiply(embeddings[start:]).sum(axis=1)).flatten().mean()


class DocAnalysis():
    """Analysis class that can analyse embeddings.

    Arguments
    ---------
    data: Data to analyze (class that handles hdf5 files).
    """
    def __init__(self, data: DataModel):
        self.data = data

    def compute_impact_novelty(  # pylint: disable=too-many-locals
            self,
            window_name: str,
            model_name: str,
            window: Optional[Union[int, tuple[int, int]]] = None
            ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], int]:
        """Compute the impact and novelty for a window/model name.

        Arguments
        ---------
        window_name:
            Name of the window to compute.
        model_name:
            Name of the model.
        window:
            Size in years of the window to use. If an integer, range will be
            [-window, window].
        """

        _patent_ids, patent_years = self.data.load_window(window_name)
        min_year = np.amin(patent_years)
        max_year = np.amax(patent_years)
        focal_year = int((min_year+max_year)/2)
        if window is None:
            window = (1, int(max(focal_year-min_year, max_year - focal_year)))
        elif isinstance(window, int):
            window = (1, window)
        embeddings = self.data.load_embeddings(window_name, model_name)
        focal_idx = np.where(patent_years == focal_year)[0]
        back_idx = np.where((patent_years <= focal_year - window[0])
                            & (patent_years >= focal_year-window[1]))
        forw_idx = np.where((patent_years >= focal_year + window[0])
                            & (patent_years <= focal_year+window[1]))

        impact_arr: npt.NDArray[np.float_] = np.zeros(len(focal_idx))
        novelty_arr: npt.NDArray[np.float_] = np.zeros(len(focal_idx))
        embeddings_backward = embeddings[back_idx]
        embeddings_forward = embeddings[forw_idx]

        for i_cur_index, cur_index in enumerate(focal_idx):
            cur_embedding = embeddings[cur_index]
            if len(cur_embedding.shape) == 1:
                cur_embedding = cur_embedding.reshape(1, -1)

            backward_similarity = np.mean(cosine_similarity(cur_embedding, embeddings_backward))
            forward_similarity = np.mean(cosine_similarity(cur_embedding, embeddings_forward))
            novelty_arr[i_cur_index] = 1-backward_similarity
            impact_arr[i_cur_index] = forward_similarity / (backward_similarity+1e-12)

        return impact_arr, novelty_arr, focal_year

    def auto_correlation(self,
                         window_name: str,
                         model_name: str) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Compute autocorrelations for embeddings."""
        embeddings = normalize(self.data.load_embeddings(window_name, model_name))
        patent_ids, _ = self.data.load_window(window_name)
        delta_count = np.unique((1+0.3*np.arange(5000)**2+0.5).astype(int))
        delta_count = delta_count[delta_count < len(patent_ids)]
        delta_year = 4*delta_count/len(patent_ids)
        auto_correlations = np.array([_auto_cor(i, embeddings) for i in delta_count])
        return delta_year, auto_correlations

    def patent_impacts(self, window_name: str, model_name: str) -> npt.NDArray[np.float_]:
        """Compute impact using cosine similarity between document vectors
        """
        try:
            impacts = self.data.load_impacts(window_name, model_name)
        except KeyError:
            impacts, novelties, focal_year = self.compute_impact_novelty(window_name, model_name)
            self.data.store_impact_novelty(window_name, model_name, focal_year, impacts, novelties)
        return impacts

    def patent_novelties(self, window_name: str, model_name: str) -> npt.NDArray[np.float_]:
        """Compute novelty using cosine similarity between document vectors
        """
        try:
            novelties = self.data.load_novelties(window_name, model_name)
        except KeyError:
            impacts, novelties, focal_year = self.compute_impact_novelty(window_name, model_name)
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
