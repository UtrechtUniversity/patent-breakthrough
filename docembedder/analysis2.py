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


class DocAnalysis():  # pylint: disable=too-few-public-methods
    """Analysis class that can analyse embeddings.

    Arguments
    ---------
    data: Data to analyze (class that handles hdf5 files).
    """
    def __init__(self, data: DataModel):
        self.data = data

    def compute_impact(self):
        """ Compute impact using cosine similarity between document vectors

        Function to calculate the impact of the focused patent for a window of the years.
        Impact score is calculated as:
        the average of the backward similarity / the average of the forward similarity.
        Note that a negative value for impact implies that there is no valid impact for that
        patent available.

        """
        list_models = self.data.model_names
        list_windows = self.data.windows_list

        for model in list_models:
            for year in list_windows:
                patent_ids, patent_years = self.data.load_window(year)
                embs = self.data.load_embeddings(year, model)

                ids_years_embs = np.array(list(zip(patent_ids, patent_years, embs)))
                impact_list = []
                impact = 0

                for item in ids_years_embs:
                    backward_similarity = 0
                    forward_similarity = 0
                    len_backward = 0
                    len_forward = 0
                    focus_index = item[0]
                    focus_year = item[1]
                    focus_emb = item[2]

                    for patent_index, patent_year, patent_emb in ids_years_embs:
                        if patent_index == focus_index:
                            continue
                        if patent_year <= focus_year:
                            backward_similarity += cosine_similarity(focus_emb, patent_emb)
                            len_backward += 1
                        elif patent_year > focus_year:
                            forward_similarity += cosine_similarity(focus_emb, patent_emb)
                            len_forward += 1

                    if len_backward == 0:
                        average_backward_similarity = -1
                    else:
                        average_backward_similarity = backward_similarity / len_backward
                    if len_forward == 0:
                        average_forward_similarity = -1
                    else:
                        average_forward_similarity = forward_similarity / len_forward

                    if average_forward_similarity == 0:
                        impact = -1
                    else:
                        impact = average_backward_similarity / average_forward_similarity
                        impact = impact.tolist()
                        impact_list.append(impact[0][0])

                impact_arr = np.array(impact_list)
                self.data.store_impacts(year, model, impact_arr)

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
