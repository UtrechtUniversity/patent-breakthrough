"""Functions to run models for hyperparameter optimization in parallel.
"""

from __future__ import annotations
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from tqdm import tqdm
import numpy as np

from docembedder.analysis import _compute_cpc_cor
from docembedder.classification import PatentClassification
from docembedder.utils import SimulationSpecification
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.typing import PathType


def _prep_worker(job) -> tuple[int, list[str], dict]:
    sim_spec, prep, patent_fp, year, cpc_fp = job
    pat_class = PatentClassification(cpc_fp)
    patents = prep.preprocess_file(
        patent_fp, max_patents=sim_spec.debug_max_patents)
    documents = [pat["contents"] for pat in patents]
    patent_ids = [pat["patent"] for pat in patents]
    cpc_cor = pat_class.sample_cpc_correlations(
        patent_ids,
        samples_per_patent=10,
        seed=12345)

    return year, documents, cpc_cor


def get_patent_data_multi(sim_spec: SimulationSpecification,  # pylint: disable=too-many-locals,too-many-arguments
                          prep: Preprocessor,
                          patent_dir: PathType,
                          cpc_fp: PathType,
                          n_jobs: int=10,
                          progress_bar=True) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Get all documents and cpc correlations in parallel.

    Arguments
    ---------
    sim_spec:
        Simulation specifications to get the patent data for.
    prep:
        Preprocessor to use.
    patent_dir:
        Directory where the patents are stored.
    cpc_fp:
        File where the CPC classifications are stored.
    n_jobs:
        Number of cores to use.

    Returns
    -------
    documents, cpc_cors:
        Documents and CPC correlations for each of the windows/year ranges.
    """
    all_years = []
    for year_list in sim_spec.year_ranges:
        all_years.extend(year_list)
    all_years = list(set(all_years))
    all_jobs = [(sim_spec, prep, Path(patent_dir, f"{year}.xz"), year, cpc_fp)
                for year in all_years if Path(patent_dir, f"{year}.xz").is_file()]
    patent_dict = {}
    if n_jobs == 1:
        for job in tqdm(all_jobs, total=len(all_jobs), disable=not progress_bar):
            year, cur_doc, cur_cpc_cors = _prep_worker(job)
            patent_dict[year] = (cur_doc, cur_cpc_cors)
    else:
        with Pool(processes=n_jobs) as pool:
            for year, cur_doc, cur_cpc_cors in tqdm(pool.imap_unordered(_prep_worker, all_jobs),
                                                    total=len(all_jobs),
                                                    disable=not progress_bar):
                patent_dict[year] = (cur_doc, cur_cpc_cors)

    documents: list[list[str]] = []
    cpc_cors: list[dict] = []

    def extend(docs, cpc, year):
        if year not in patent_dict:
            return
        cur_len = len(docs)
        docs.extend(patent_dict[year][0])
        cur_cpc = patent_dict[year][1]
        cpc["i_patents"].extend([x + cur_len for x in cur_cpc["i_patents"]])
        cpc["j_patents"].extend([x + cur_len for x in cur_cpc["j_patents"]])
        cpc["correlations"].extend(cur_cpc["correlations"])

    for year_list in sim_spec.year_ranges:
        new_docs: list[str] = []
        new_cpc: dict[str, Any] = defaultdict(list)
        for year in year_list:
            extend(new_docs, new_cpc, year)
        if len(new_docs) == 0:
            continue
        documents.append(new_docs)
        for key in new_cpc:
            new_cpc[key] = np.array(new_cpc[key])
        cpc_cors.append(dict(new_cpc))
    return documents, cpc_cors


def run_jobs(model, documents, cpc_cor, n_jobs=10):
    """Get the mean correlation from model, documents and CPC correlations."""
    jobs = [(model, documents[i], cpc_cor[i]) for i in range(len(documents))]
    if n_jobs == 1:
        correlations = []
        for job in jobs:
            correlations.append(_pool_worker(job))
    else:
        with Pool(processes=n_jobs) as pool:
            correlations = list(pool.imap_unordered(_pool_worker, jobs))
    return np.mean(correlations)


def _pool_worker(job):
    model, patents, cpc_correlations = job
    model.fit(patents)
    embeddings = model.transform(patents)
    cpc_cor = _compute_cpc_cor(embeddings, cpc_correlations)
    return cpc_cor
