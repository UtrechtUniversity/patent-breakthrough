from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
import numpy as np

from docembedder.analysis import _compute_cpc_cor
from docembedder.classification import PatentClassification
from collections import defaultdict


def _prep_worker(job) -> tuple[int, list[str], dict]:
    sim_spec, prep, patent_fp, year, cpc_fp = job
    pat_class = PatentClassification(cpc_fp)
    empty_cpc = {"i_patents": [], "j_patents": [], "correlations": []}
    try:
        patents = prep.preprocess_file(
            patent_fp, max_patents=sim_spec.debug_max_patents)
    except FileNotFoundError:
        return year, [], empty_cpc
    if len(patents) < 10:
        return year, [], empty_cpc
    documents = [pat["contents"] for pat in patents]
    patent_ids = [pat["patent"] for pat in patents]
    cpc_cor = pat_class.sample_cpc_correlations(
        patent_ids,
        samples_per_patent=10,
        seed=12345)

    return year, documents, cpc_cor


def get_patent_data_multi(sim_spec, prep, patent_dir, cpc_fp, n_jobs=10):
    all_years = []
    for year_list in sim_spec.year_ranges:
        all_years.extend(year_list)
    all_years = list(set(all_years))
    all_jobs = [(sim_spec, prep, Path(patent_dir, f"{year}.xz"), year, cpc_fp) for year in all_years]
    patent_dict = {}
    with Pool(processes=n_jobs) as pool:
        for year, documents, cpc_cors in tqdm(pool.imap_unordered(_prep_worker, all_jobs),
                                              total=len(all_jobs)):
            patent_dict[year] = (documents, cpc_cors)

    documents = [[] for _ in range(len(list(sim_spec.year_ranges)))]
    cpc_cors = [defaultdict(list) for _ in range(len(list(sim_spec.year_ranges)))]

    def extend(i_list, year):
        cur_len = len(documents[i_list])
        documents[i_list].extend(patent_dict[year][0])
        cur_cpc = patent_dict[year][1]
        cpc_cors[i_list]["i_patents"].extend([x + cur_len for x in cur_cpc["i_patents"]])
        cpc_cors[i_list]["j_patents"].extend([x + cur_len for x in cur_cpc["j_patents"]])
        cpc_cors[i_list]["correlations"].extend(cur_cpc["correlations"])

    for i_list, year_list in enumerate(sim_spec.year_ranges):
        for year in year_list:
            extend(i_list, year)
        for key in cpc_cors[i_list]:
            cpc_cors[i_list][key] = np.array(cpc_cors[i_list][key])
        cpc_cors[i_list] = dict(cpc_cors[i_list])
            # documents[i_list].extend(patent_dict[year][0])
            # cpc_cors[i_list].extend(patent_dict[year][1])
    return documents, cpc_cors


def get_patent_data(sim_spec, prep, patent_dir):
    patent_dict = {}
    documents = []
    patent_ids = []
    for year_list in sim_spec.year_ranges:
        cur_documents = []
        cur_patent_ids = []
        for year in year_list:
            try:
                if year in patent_dict:
                    patents = patent_dict[year]
                else:
                    patent_fp = Path(patent_dir, f"{year}.xz")
                    patents = prep.preprocess_file(patent_fp,
                                                   max_patents=sim_spec.debug_max_patents)
                cur_documents.extend([pat["contents"] for pat in patents])
                cur_patent_ids.extend(pat["patent"] for pat in patents)
            except FileNotFoundError:
                pass
        documents.append(cur_documents)
        patent_ids.append(cur_patent_ids)
    return documents, patent_ids


def get_cpc_data(patent_ids, cpc_fp, seed=12345):
    pat_class = PatentClassification(cpc_fp)

    cpc_cor = [
        pat_class.sample_cpc_correlations(
            cur_patent_ids,
            samples_per_patent=10,
            seed=seed)
        for cur_patent_ids in patent_ids
    ]
    return cpc_cor


def run_jobs(model, documents, cpc_cor, n_jobs=10):
    jobs = [(model, documents[i], cpc_cor[i]) for i in range(len(documents))]
    with Pool(processes=n_jobs) as pool:
        correlations = list(pool.imap_unordered(_pool_worker, jobs))
    return np.mean(correlations)


def _pool_worker(job):
    model, patents, cpc_correlations = job
    model.fit(patents)
    embeddings = model.transform(patents)
    cpc_cor = _compute_cpc_cor(embeddings, cpc_correlations)
    return cpc_cor
