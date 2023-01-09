from collections import defaultdict
from typing import List

import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix


def _compute_cpc_cor(model_res, cpc_res, patent_id, chunk_size=10000):
    n_split = max(1, len(cpc_res["correlations"]) // chunk_size)
    i_pat_split = np.array_split(cpc_res["i_patents"], n_split)
    j_pat_split = np.array_split(cpc_res["j_patents"], n_split)

    model_cor: List[float] = []
    embeddings = model_res

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


class DocAnalysis():
    def __init__(self, data):
        self.data = data

    def cpc_correlations(self, models=None):
        if models is None:
            models = self.data.model_names
        elif isinstance(models, str):
            models = [models]

        correlations = defaultdict(lambda: [])
        years = []
        for res in self.data.iterate_embeddings(return_cpc=True, model_names=models):
            patent_id = res["patent_id"]
            cpc_res = res["cpc"]
            years.append(res["year"])
            for model_name, model_res in res["embeddings"].items():
                correlations[model_name].append(_compute_cpc_cor(model_res, cpc_res, patent_id))
        return years, dict(correlations)
