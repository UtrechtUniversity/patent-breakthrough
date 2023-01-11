"""General utilities for performing runs."""

from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from docembedder import DataModel
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.classification import PatentClassification


STARTING_YEAR = 1838  # First year of the patents dataset


class SimulationSpecification():
    """Specification for doing runs.

    It uses the starting year so that windows start regularly and always
    from the same years, independent of where the run itself starts. All years
    between the year_start and year_end will be present in at least one of the windows.
    It is possible that some years will be included that are outside this interval.

    Arguments
    ---------
    year_start:
        Start of the windows to run the models on.
    year_end:
        End of the windows to run the models on.
    window_size:
        Number of years in each window. Each consecutive window is shifted by
        the window_size divided by 2 rounded up.
    cpc_samples_per_patent:
        Number of CPC correlation samples per patent.
    debug_max_patents:
        Only read the first x patents from the file to speed up computation.
        Leave at None for not skipping anything.
    n_patents_per_window:
        Number of patents to be drawn for each window. If None, all patents
        are used.
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                 year_start: int, year_end: int, window_size: int=1,
                 cpc_samples_per_patent: int=1000,
                 debug_max_patents: Optional[int]=None,
                 n_patents_per_window: Optional[int]=None):
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
            try:
                patents.extend(prep.preprocess_file(
                    self.job_data["patent_dir"] / (str(year) + ".xz"),
                    self.sim_spec.debug_max_patents)
                )
            except FileNotFoundError:
                pass

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


def insert_models(models, output_fp):
    with DataModel(output_fp, read_only=False) as data:
        for model_name, model in models.items():
            if not data.has_model(model_name):
                data.store_model(model_name, model)


def _pool_worker(job):
    return job.run()


def run_jobs_multi(jobs, output_fp, n_jobs=10):
    if len(jobs) == 0:
        return

    all_files = []
    with Pool(processes=n_jobs) as pool:
        for temp_data_fp in tqdm(pool.imap_unordered(_pool_worker, jobs), total=len(jobs)):
            all_files.append(temp_data_fp)

    with DataModel(output_fp, read_only=False) as data:
        for temp_data_fp in all_files:
            data.add_data(temp_data_fp, delete_copy=True)


def check_sim_specification(output_fp, sim_spec):
    with DataModel(output_fp, read_only=False) as data:
        try:
            return data.handle.attrs["sim_spec"] == sim_spec.name
        except KeyError:
            data.handle.attrs["sim_spec"] = sim_spec.name
        return True


def run_models(models, sim_spec, patent_dir, output_fp, cpc_fp, n_jobs=10):
    if not check_sim_specification(output_fp, sim_spec):
        raise ValueError("Simulation specifications do not match existing specifications.")

    insert_models(models, output_fp)
    jobs = sim_spec.create_jobs(output_fp, models, cpc_fp, patent_dir)
    run_jobs_multi(jobs, output_fp, n_jobs=n_jobs)
