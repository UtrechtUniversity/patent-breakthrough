"""Analysis functions and classes for embedding results."""

from collections import defaultdict
from typing import List, Union, Dict, Any, Optional, DefaultDict

import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

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

# pylint: disable=R0904
# pylint: disable=R0914

class DocAnalysis():  # pylint: disable=too-few-public-methods
    """Analysis class that can analyse embeddings.

    Arguments
    ---------
    data: Data to analyze (class that handles hdf5 files).
    """
    def __init__(self, data: DataModel):
        self.data = data

    def _compute_impact(self, model_name, window_name):
        patent_ids, patent_years = self.data.load_window(window_name)
        embeddings = self.data.load_embeddings(window_name, model_name)
        patent_indices = np.array(range(len(patent_ids)))
        # impact_list: List[float] = []
        impact_arr: np.ndarray = np.empty(len(patent_years)) * np.nan

        for cur_index in range(len(patent_ids)):
            cur_embedding = embeddings[cur_index]
            cur_year = patent_years[cur_index]

            # other_indices = []
            # for i in [x for x in range(len(patent_ids)) if x != cur_index]:
            #     other_indices.append(i)

            other_indices = np.delete(patent_indices, cur_index)

            other_embeddings = embeddings[other_indices,:]
            other_years = np.delete(patent_years, cur_index)

            embeddings_backward = other_embeddings[other_years < cur_year]
            embeddings_forward = other_embeddings[other_years > cur_year]

            if embeddings_backward.size:
                backward_similarity = cosine_similarity(cur_embedding, embeddings_backward)
            else:
                backward_similarity = np.nan
            average_backward_similarity = np.mean(backward_similarity)

            if embeddings_forward.size:
                forward_similarity = cosine_similarity(cur_embedding, embeddings_forward)
            else:
                forward_similarity = np.nan
            average_forward_similarity = np.mean(forward_similarity)

            if average_forward_similarity and average_backward_similarity:
                # impact = average_backward_similarity / average_forward_similarity
                impact_arr[cur_index] = average_backward_similarity / average_forward_similarity
            # else:
            #     impact = np.nan

            # try:
            #     impact = average_backward_similarity / average_forward_similarity
            # except (ZeroDivisionError, TypeError):
            #     impact = np.nan

            # impact_arr.append(impact)

        return impact_arr

    def patent_impacts(self):
        """ Compute impact using cosine similarity between document vectors
        """

        for window_name, model_name in tqdm(self.data.iterate_window_models()):
            impact_array = self._compute_impact(model_name, window_name)
            # impact_arr = np.array(impact_list)
            self.data.store_impacts(window_name, model_name, impact_array)



    def cpc_correlations(self, models: Optional[Union[str, List[str]]]=None
                         ) -> Dict[str, Dict[str, List[float]]]:
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
        else:
            raise TypeError("models argument must be a string or a list of strings.")

        correlations: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"year": [], "correlations": []})

        for window, model_name in tqdm(self.data.iterate_window_models()):
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
                year = float(window)
            except ValueError:
                year = float(np.mean([float(x) for x in window.split("-")]))

            correlations[model_name]["year"].append(year)
            correlations[model_name]["correlations"].append(correlation)
        return dict(correlations)
