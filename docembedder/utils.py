import itertools
from collections import defaultdict
from time import sleep

from docembedder import DataModel


from multiprocessing import Pool
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.classification import PatentClassification
from copy import deepcopy


def init_year_jobs(years, output_fp):
    with DataModel(output_fp, read_only=True) as data:
        year_jobs = {year: None for year in years if not data.has_year(year)}
    return year_jobs


def init_cpc_jobs(years, output_fp):
    with DataModel(output_fp, read_only=True) as data:
        cpc_jobs = [year for year in years if not data.has_cpc(year)]
    return cpc_jobs


def init_run_jobs(years, models, output_fp):
    run_jobs = defaultdict(lambda: [])
    with DataModel(output_fp, read_only=True) as data:
        for year, model in itertools.product(years, models):
            # print(year, model, data.has_run(model, year))
            if not data.has_run(model, year):
                run_jobs[year].append(model)
    return dict(run_jobs)


def get_all_jobs(models, years, patent_dir, output_fp, cpc_fp, max_patents, samples_per_patent):
    year_jobs = init_year_jobs(years, output_fp)
    cpc_jobs = init_cpc_jobs(years, output_fp)
    run_jobs = init_run_jobs(years, models, output_fp)

    all_jobs = []
    for year, r_job in run_jobs.items():
        all_jobs.append(
            {
                "year": year,
                "init_year": year in year_jobs,
                "cpc": year in cpc_jobs,
                "run": r_job,
                "patent_dir": patent_dir,
                "cpc_fp": cpc_fp,
                "max_patents": max_patents,
                "samples_per_patent": samples_per_patent,
                "output_fp": output_fp,
            }
        )
    return all_jobs


def insert_models(models, output_fp):
    with DataModel(output_fp, read_only=False) as data:
        for model_name, model in models.items():
            if not data.has_model(model_name):
                data.store_model(model_name, model)


def pool_worker(job):
    year = job["year"]
    cpc = job["cpc"]
    model_names = job["run"]
    init_year = job["init_year"]
    prep = Preprocessor()

    patents = prep.preprocess_file(job["patent_dir"] / (str(year) + ".xz"),
                                   job["max_patents"])
    if init_year:
        train_id = [pat["patent"] for pat in patents]
        test_id = train_id
    else:
        with DataModel(job["output_fp"], read_only=True) as data:
            train_id, test_id = data.get_train_test_id(year)

    train_documents = [pat["contents"] for pat in patents if pat["patent"] in train_id]
    test_documents = [pat["contents"] for pat in patents if pat["patent"] in test_id]

    all_embeddings = {}
    for model_name in model_names:
        with DataModel(job["output_fp"], read_only=True) as data:
            model = data.load_model(model_name)
        model.fit(train_documents)
        all_embeddings[model_name] = model.transform(test_documents)

    if cpc:
        pat_class = PatentClassification(job["cpc_fp"])
        cpc_cor = pat_class.sample_cpc_correlations(test_id, samples_per_patent=10)

    results = {
        "embeddings": all_embeddings,
    }
    if cpc:
        results["cpc"] = cpc_cor

    if init_year:
        results["test_id"] = test_id
        results["train_id"] = train_id

    results["job"] = deepcopy(job)
    return results


def run_jobs_multi(jobs, output_fp, n_jobs=10):
    all_return = []
    with Pool(processes=n_jobs) as pool:
        for ret in pool.imap_unordered(pool_worker, jobs):
            all_return.append(ret)

    with DataModel(output_fp, read_only=False) as data:
        for result in all_return:
            year = result["job"]["year"]
            if "test_id" in result:
                data.store_year(year, result["test_id"], result["train_id"])
            if "cpc" in result:
                data.store_cpc_correlations(result["cpc"], year)
            for model_name, embeddings in result["embeddings"].items():
                data.store_embeddings(model_name, embeddings, year)


def run_models(models, years, patent_dir, output_fp, cpc_fp, max_patents=None,
               samples_per_patent=10, n_jobs=10):
    with DataModel(output_fp, read_only=False):
        pass
    insert_models(models, output_fp)
    jobs = get_all_jobs(models, years, patent_dir, output_fp, cpc_fp, max_patents,
                        samples_per_patent)
    run_jobs_multi(jobs, output_fp, n_jobs=n_jobs)
