"""General utilities for performing runs."""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.models.base import BaseDocEmbedder
from docembedder.typing import PathType, FileType
from docembedder.utils import SimulationSpecification, insert_models, create_jobs


def _compute_embeddings(jobs, progress_bar=True):  # pylint: disable=too-many-locals
    patent_cache = defaultdict(dict)
    embed_cache = defaultdict(dict)
    model_cache = {}

    def _update_cache(job):
        year_list = job.job_data["year_list"]
        for year in year_list:
            for prep_name, prep in job.preps.items():
                if year in patent_cache[prep_name]:
                    continue
                patent_cache[prep_name][year] = prep.preprocess_file(
                    Path(job.job_data["patent_dir"]) / (str(year) + ".xz"),
                    job.sim_spec.debug_max_patents,
                    return_stats=False
                )
        for patent_data in patent_cache.values():
            year_delete = set(list(patent_data))-set(year_list)
            for year in year_delete:
                del patent_data[year]

    def _transform(documents, patent_id, prep_model_name):
        new_idx = []
        for i_pat, pat_id in enumerate(patent_id):
            if pat_id not in embed_cache[prep_model_name]:
                new_idx.append(i_pat)
        new_docs = [documents[i_pat] for i_pat in new_idx]
        new_embeddings = model.transform(new_docs)
        all_embeddings = np.zeros((len(documents), new_embeddings.shape[1]))
        all_embeddings[new_idx] = new_embeddings
        for i_pat, pat_id in enumerate(patent_id):
            if pat_id in embed_cache[prep_model_name]:
                all_embeddings[i_pat] = embed_cache[prep_model_name][pat_id]
            else:
                embed_cache[prep_model_name][pat_id] = all_embeddings[i_pat]
        return all_embeddings

    for job in tqdm(jobs, disable=not progress_bar):
        _update_cache(job)
        for prep_name in job.preps:
            patents = []
            for pats in patent_cache[prep_name].values():
                patents.extend(pats)
            patent_id, patent_year = job.get_patent_ids(patents, job.job_data["name"])
            documents = [pat["contents"] for pat in patents if pat["patent"] in patent_id]
            for model_name, model in job.models.items():
                if model_name in model_cache:
                    model = model_cache[model_name]
                else:
                    model_cache[model_name] = model
                prep_model_name = f"{prep_name}-{model_name}"
                embeddings = _transform(documents, patent_id, prep_model_name)
                cpc_cor = job.compute_cpc(patent_id)
                job.store_results(job.job_data["output_fp"], job.job_data["name"], patent_id,
                                  patent_year,
                                  cpc_cor, {prep_model_name: embeddings}, unlink=False)


def pretrained_run_models(  # pylint: disable=too-many-arguments
        preprocessors: Optional[dict[str, Preprocessor]],
        models: dict[str, BaseDocEmbedder],
        sim_spec: SimulationSpecification,
        patent_dir: PathType,
        output_fp: FileType,
        cpc_fp: PathType,
        progress_bar: bool=True):
    """Run models with the simulation specifications.

    Arguments
    ---------
    models:
        Dictionary containing the models to be run.
    sim_spec:
        Specification for the run(s).
    patent_dir:
        Directory where the patents are stored. (*.xz)
    output_fp:
        File to store the results in. If it exists, add only non-existing results.
    cpc_fp:
        File that contains the CPC classifications (GPCPCs.txt).
    n_jobs:
        Number of jobs to be run in parallel.
    progress_bar:
        Whether to show a progress bar.
    """  # pylint: disable=duplicate-code
    if not sim_spec.check_file(output_fp):
        raise ValueError("Simulation specifications do not match existing specifications.")

    if preprocessors is None:
        preprocessors = {"default": Preprocessor()}
    insert_models(models, preprocessors, output_fp)
    jobs = create_jobs(sim_spec, output_fp, models, preprocessors, cpc_fp, patent_dir)
    if len(jobs) == 0:
        return
    _compute_embeddings(jobs, progress_bar=progress_bar)
