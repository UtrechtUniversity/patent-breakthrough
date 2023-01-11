import itertools
from collections import defaultdict
from pathlib import Path

from docembedder import DataModel
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.classification import PatentClassification


STARTING_YEAR = 1838


class SimulationSpecification():
    def __init__(self, year_start, year_end, window_size=1, cpc_samples_per_patent=1000,
                 debug_max_patents=None, n_patents_per_window=None):
        self.year_start = year_start
        self.year_end = year_end
        self.window_size = window_size
        self.cpc_samples_per_patent = cpc_samples_per_patent
        self.debug_max_patents = debug_max_patents
        self.n_patents_per_window = n_patents_per_window

    def create_jobs(self, output_fp, models, cpc_fp, patent_dir):
        jobs = []
        cur_start = self.year_start - ((self.year_start-STARTING_YEAR) % self.window_size)
        cur_end = self.year_start + self.window_size
        delta = (self.window_size+1)//2
        with DataModel(output_fp, read_only=True) as data:
            while cur_start < self.year_end:
                year_list = list(range(cur_start, cur_end))
                name = f"{cur_start}-{cur_end-1}"
                compute_year = not data.has_year(name)
                compute_cpc = not data.has_cpc(name)
                compute_models = [model for model in models if not data.has_run(model, name)]
                if compute_year or compute_cpc or compute_models:
                    jobs.append(Job(
                        job_data={
                            "name": name,
                            "output_fp": output_fp, "cpc_fp": cpc_fp,
                            "temp_fp": Path(output_fp.parent, f"temp_{name}.h5"),
                            "year_list": year_list,
                            "patent_dir": patent_dir,
                        },
                        compute_year=compute_year,
                        compute_cpc=compute_cpc,
                        compute_models=compute_models,
                        sim_spec=self,
                    ))
                cur_start += delta
                cur_end += delta
        return jobs

    @property
    def name(self):
        return (f"s{self.year_start}-e{self.year_end}-w{self.window_size}-"
                f"c{self.cpc_samples_per_patent}-d{self.debug_max_patents}-"
                f"n{self.n_patents_per_window}")


class Job():
    def __init__(self, job_data, compute_year, compute_cpc, compute_models, sim_spec):
        self.job_data = job_data
        self.compute_year = compute_year
        self.compute_cpc = compute_cpc
        self.compute_models = compute_models
        self.sim_spec = sim_spec

    def run(self):
        year_list = self.job_data["year_list"]
        temp_fp = self.job_data["temp_fp"]
        name = self.job_data["name"]

        prep = Preprocessor()

        patents = []
        for year in year_list:
            patents.extend(prep.preprocess_file(
                self.job_data["patent_dir"] / (str(year) + ".xz"),
                self.sim_spec.debug_max_patents)
            )

        if self.compute_year:
            if self.sim_spec.n_patents_per_window is None:
                train_id = [pat["patent"] for pat in patents]
            else:
                train_id = np.random.choice([pat["patent"] for pat in patents],
                                            size=min(len(patents), self.sim_spec.n_patents))
            test_id = train_id
        else:
            with DataModel(self.job_data["output_fp"], read_only=True) as data:
                train_id, test_id = data.get_train_test_id(name)

        train_documents = [pat["contents"] for pat in patents if pat["patent"] in train_id]
        test_documents = [pat["contents"] for pat in patents if pat["patent"] in test_id]

        all_embeddings = {}
        for model_name in self.compute_models:
            with DataModel(self.job_data["output_fp"], read_only=True) as data:
                model = data.load_model(model_name)
            model.fit(train_documents)
            all_embeddings[model_name] = model.transform(test_documents)

        if self.compute_cpc:
            pat_class = PatentClassification(self.job_data["cpc_fp"])
            cpc_cor = pat_class.sample_cpc_correlations(test_id, samples_per_patent=10)

        temp_fp.unlink(missing_ok=True)
        with DataModel(temp_fp, read_only=False) as data:
            if self.compute_year:
                data.store_year(name, test_id, train_id)
            if self.compute_cpc:
                data.store_cpc_correlations(name, cpc_cor)
            for model_name, embeddings in all_embeddings.items():
                data.store_embeddings(name, model_name, embeddings)
        return temp_fp

    def __str__(self):
        return (f"{self.job_data['name']}, year: {self.compute_year}, cpc: {self.compute_cpc}"
                f", models: {self.compute_models}")


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
            if not data.has_run(model, year):
                run_jobs[year].append(model)
    return dict(run_jobs)


def get_all_jobs(models, years, patent_dir, output_fp, cpc_fp, max_patents, samples_per_patent):
    year_jobs = init_year_jobs(years, output_fp)
    cpc_jobs = init_cpc_jobs(years, output_fp)
    run_jobs = init_run_jobs(years, models, output_fp)

    all_jobs = []
    for year, r_job in run_jobs.items():
        all_jobs.append({
                "year": year,
                "init_year": year in year_jobs,
                "cpc": year in cpc_jobs,
                "run": r_job,
                "patent_dir": patent_dir,
                "cpc_fp": cpc_fp,
                "max_patents": max_patents,
                "samples_per_patent": samples_per_patent,
                "output_fp": output_fp,
                "temp_fp": Path(output_fp.parent, f"temp_{year}.h5"),
        })
    return all_jobs


def insert_models(models, output_fp):
    with DataModel(output_fp, read_only=False) as data:
        for model_name, model in models.items():
            if not data.has_model(model_name):
                data.store_model(model_name, model)


def pool_worker(job):
    return job.run()

# def pool_worker(job):
#     year = job["year"]
#     cpc = job["cpc"]
#     model_names = job["run"]
#     init_year = job["init_year"]
#     temp_fp = job["temp_fp"]
#
#     prep = Preprocessor()
#
#     patents = prep.preprocess_file(job["patent_dir"] / (str(year) + ".xz"),
#                                    job["max_patents"])
#     if init_year:
#         train_id = [pat["patent"] for pat in patents]
#         test_id = train_id
#     else:
#         with DataModel(job["output_fp"], read_only=True) as data:
#             train_id, test_id = data.get_train_test_id(year)
#
#     train_documents = [pat["contents"] for pat in patents if pat["patent"] in train_id]
#     test_documents = [pat["contents"] for pat in patents if pat["patent"] in test_id]
#
#     all_embeddings = {}
#     for model_name in model_names:
#         with DataModel(job["output_fp"], read_only=True) as data:
#             model = data.load_model(model_name)
#         model.fit(train_documents)
#         all_embeddings[model_name] = model.transform(test_documents)
#
#     if cpc:
#         pat_class = PatentClassification(job["cpc_fp"])
#         cpc_cor = pat_class.sample_cpc_correlations(test_id, samples_per_patent=10)
#
#     temp_fp.unlink(missing_ok=True)
#     with DataModel(temp_fp, read_only=False) as data:
#         if init_year:
#             data.store_year(year, test_id, train_id)
#         if cpc:
#             data.store_cpc_correlations(cpc_cor, year)
#         for model_name, embeddings in all_embeddings.items():
#             data.store_embeddings(model_name, embeddings, year)
#
#     return temp_fp


def check_sim_specification(output_fp, sim_spec):
    with DataModel(output_fp, read_only=False) as data:
        try:
            return data.handle.attrs["sim_spec"] == sim_spec.name
        except KeyError:
            data.handle.attrs["sim_spec"] = sim_spec.name
        return True


def run_jobs_multi(jobs, output_fp, n_jobs=10):
    if len(jobs) == 0:
        return

    all_files = []
    with Pool(processes=n_jobs) as pool:
        for temp_data_fp in tqdm(pool.imap_unordered(pool_worker, jobs), total=len(jobs)):
            all_files.append(temp_data_fp)

    with DataModel(output_fp, read_only=False) as data:
        for temp_data_fp in all_files:
            data.add_data(temp_data_fp, delete_copy=True)


def run_models(models, sim_spec, patent_dir, output_fp, cpc_fp, n_jobs=10):

    # Open the file in read/write mode so that it is initialized.
    # with DataModel(output_fp, read_only=False):
    #     pass

    if not check_sim_specification(output_fp, sim_spec):
        raise ValueError("Simulation specifications do not match existing specifications.")

    insert_models(models, output_fp)
    jobs = sim_spec.create_jobs(output_fp, models, cpc_fp, patent_dir)
    run_jobs_multi(jobs, output_fp, n_jobs=n_jobs)
