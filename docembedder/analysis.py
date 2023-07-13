"""Analysis functions and classes for embedding results."""

from __future__ import annotations
from collections import defaultdict
import multiprocessing
from typing import List, Union, Dict, Any, Optional, DefaultDict

import numpy as np
from numpy import typing as npt
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm

from docembedder.datamodel import DataModel
from docembedder.models.base import AllEmbedType
from docembedder.embedding_utils import _gather_results


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


def _multi_compute_impact(job):
    return _compute_impact(*job)


def _compute_impact(embedding_back, embedding_focal, embedding_forw, exponent=1.0):
    if isinstance(embedding_focal, csr_matrix):
        sim_back = (embedding_focal.dot(embedding_back.T).toarray()+1)/2
        sim_forw = (embedding_focal.dot(embedding_forw.T).toarray()+1)/2
    else:
        sim_back = (np.dot(embedding_focal, embedding_back.T)+1)/2
        sim_forw = (np.dot(embedding_focal, embedding_forw.T)+1)/2
    avg_sim_back = (sim_back**exponent).mean(axis=1)**(1/exponent)
    avg_sim_forw = (sim_forw**exponent).mean(axis=1)**(1/exponent)
    novelty = 1-avg_sim_back
    impact = avg_sim_forw / (avg_sim_back+1e-12)
    return {
        "novelty": novelty,
        "impact": impact,
        "exponent": exponent,
    }


def compute_impact_novelty(  # pylint: disable=too-many-arguments, too-many-locals
        embeddings: AllEmbedType,
        back_idx: npt.NDArray[np.int_], focal_idx: npt.NDArray[np.int_],
        forw_idx: npt.NDArray[np.int_],
        n_jobs: int = 10,
        max_mat_size: int = int(1e7),
        exponents: Union[float, list[float]] = 1.0,
        progress_bar: bool = False):
    """Compute the impact and novelty scores from embeddings.

    Arguments
    ---------
    embeddings:
        Document embeddings to compute the novelty and impact for.
    back_idx:
        Indices containing the backward patents.
    focal_idx:
        Indices containing the focal patents.
    forw_idx:
        Indices containing the forward patents (in the future).
    n_jobs:
        Parallelize over this many jobs.
    exponents:
        Exponents to use for computing the (weighted) average similarity.
    """
    if isinstance(exponents, float):
        exponents = [exponents]

    # Normalize the embeddings.
    embeddings_backward = normalize(embeddings[back_idx])
    embeddings_forward = normalize(embeddings[forw_idx])

    # Figure out over how many jobs the focal embeddings should be split.
    max_back_forw_len = max(embeddings_backward.shape[0], embeddings_forward.shape[0])
    mem_split = round((len(focal_idx)*max_back_forw_len)/max_mat_size)
    n_split = min(n_jobs, len(focal_idx))
    n_split = max(n_split, mem_split)

    # Split the jobs on the focal indices and the exponents.
    split_focal_idx = np.array_split(focal_idx, n_split)
    jobs = [
        (embeddings_backward, normalize(embeddings[cur_focal_idx]), embeddings_forward, expon)
        for cur_focal_idx in split_focal_idx
        for expon in exponents]

    # Compute the results
    if n_jobs == 1:
        results = [_compute_impact(*job) for job in jobs]
    else:
        results = []
        with multiprocessing.get_context('spawn').Pool(processes=n_jobs) as pool:
            for res in tqdm(pool.imap(_multi_compute_impact, jobs),
                            disable=not progress_bar, total=len(jobs)):
                # for res in pool.starmap(_compute_impact, jobs):
                results.append(res)
    return _gather_results(results)


class DocAnalysis():
    """Analysis class that can analyse embeddings.

    Arguments
    ---------
    data: Data to analyze (class that handles hdf5 files).
    """
    def __init__(self, data: DataModel):
        self.data = data

    def compute_impact_novelty(  # pylint: disable=too-many-locals,too-many-arguments
            self,
            window_name: str,
            model_name: str,
            window: Optional[Union[int, tuple[int, int]]] = None,
            exponents: Union[float, list[float]] = 1.0,
            n_jobs: int = 10,
            max_mat_size: int = int(1e8),
            ) -> dict[float, dict]:
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
        if isinstance(exponents, float):
            exponents = [exponents]

        patent_ids, patent_years = self.data.load_window(window_name)
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
                            & (patent_years >= focal_year-window[1]))[0]
        forw_idx = np.where((patent_years >= focal_year + window[0])
                            & (patent_years <= focal_year+window[1]))[0]

        results = compute_impact_novelty(
            embeddings, back_idx, focal_idx, forw_idx, n_jobs, max_mat_size,
            exponents)
        for expon in exponents:
            results[expon]["focal_year"] = focal_year
            results[expon]["patent_ids"] = patent_ids[focal_idx]
            results[expon]["exponent"] = expon
        return results

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

    def impact_novelty_results(self, window_name: str, model_name: str,
                               exponents: Union[float, list[float]],
                               cache: bool = True,
                               **kwargs) -> dict:
        """Get the impact and novelty results for a window/model.

        Arguments
        ---------
        window_name:
            Name of the window to get the data for.
        model_name:
            Name of the model to get the data for.
        exponents:
            Exponents to compute the impact/novelty with.
        kwargs:
            Extra keyword arguments for computing the novelty/impact if necessary.

        Results
        -------
        results:
            Dictionary with the impact/novelties and other properties.
        """
        if isinstance(exponents, float):
            exponents = [exponents]
        try:
            results = {}
            for expon in exponents:
                results[expon] = self.data.load_impact_novelty(window_name, model_name, expon)
        except KeyError:
            results = self.compute_impact_novelty(window_name, model_name, exponents=exponents,
                                                  **kwargs)
            if cache:
                for expon in exponents:
                    self.data.store_impact_novelty(window_name, model_name, results[expon],
                                                   overwrite=True)
        return results

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
