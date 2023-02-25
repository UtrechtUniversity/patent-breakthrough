"""General utilities for performing runs."""

from __future__ import annotations
import io
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, List, Tuple, Iterable
from time import time

import numpy as np
from tqdm import tqdm
from numpy import typing as npt

from docembedder.datamodel import DataModel
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.classification import PatentClassification
from docembedder.models.base import BaseDocEmbedder
from docembedder.typing import PathType, AllEmbedType, IntSequence, FileType


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
                 year_start: int,
                 year_end: int,
                 window_size: int=1,
                 cpc_samples_per_patent: int=10,
                 debug_max_patents: Optional[int]=None,
                 n_patents_per_window: Optional[int]=None):
        self.year_start = year_start
        self.year_end = year_end
        self.window_size = window_size
        self.cpc_samples_per_patent = cpc_samples_per_patent
        self.debug_max_patents = debug_max_patents
        self.n_patents_per_window = n_patents_per_window

    def create_jobs(self, output_fp: FileType,  # pylint: disable=too-many-arguments, too-many-locals
                    models: Dict[str, BaseDocEmbedder],
                    preprocessors: Dict[str, Preprocessor],
                    cpc_fp: PathType,
                    patent_dir: PathType):
        """Create jobs to run the simulation specification.

        Arguments
        ---------
        output_fp:
            HDF5 File to store/load embedding data to/from.
        models:
            Dictionary containing the models to be run.
        cpc_fp:
            File that contains the CPC classifcation data (GPCPCs.txt).
        patent_dir:
            Directory that contains the zipped (*.xz) patent files.
        """
        jobs = []
        # cur_start = self.year_start - ((self.year_start-STARTING_YEAR) % self.window_size)
        # cur_end = self.year_start + self.window_size
        # delta = (self.window_size+1)//2
        with DataModel(output_fp, read_only=True) as data:
            # while cur_start < self.year_end:
            for year_list in self.year_ranges:
                job_name = f"{year_list[0]}-{year_list[-1]}"
                compute_window = not data.has_window(job_name)
                compute_cpc = not data.has_cpc(job_name)
                compute_models = [(prep, model)
                                  for prep in preprocessors
                                  for model in models
                                  if not data.has_run(prep, model, job_name)]
                if compute_window or compute_cpc or len(compute_models) > 0:
                    if isinstance(output_fp, io.BytesIO):
                        temp_fp: FileType = io.BytesIO()
                    else:
                        temp_fp = Path(output_fp).parent / f"temp_{job_name}.h5"
                    jobs.append(Job(
                        job_data={
                            "name": job_name,
                            "output_fp": output_fp, "cpc_fp": cpc_fp,
                            "temp_fp": temp_fp,
                            "year_list": year_list,
                            "patent_dir": patent_dir,
                        },
                        need_window=compute_window,
                        need_cpc=compute_cpc,
                        need_models=compute_models,
                        sim_spec=self,
                    ))
        return jobs

    @property
    def year_ranges(self) -> Iterable[list[int]]:
        cur_start = self.year_start - ((self.year_start-STARTING_YEAR) % self.window_size)
        cur_end = self.year_start + self.window_size
        delta = (self.window_size+1)//2
        while cur_start < self.year_end:
            yield list(range(cur_start, cur_end))
            cur_start += delta
            cur_end += delta

    @property
    def name(self) -> str:
        """Identifier of the simulation specifications.

        This is mainly used to check whether a new run is compatible.
        Different values for `year_start` and `year_end` should be compatible.
        """
        return (f"s{STARTING_YEAR}-w{self.window_size}-"
                f"c{self.cpc_samples_per_patent}-d{self.debug_max_patents}-"
                f"n{self.n_patents_per_window}")

    def check_file(self, output_fp: FileType) -> bool:
        """Check whether the output file has a simulation specification.

        If it doesn't have one, insert the supplied specification.

        Arguments
        ---------
        output_fp:
            File to check.
        sim_spec:
            Specification to check.

        Returns
        -------
            Whether the file now has the same specification.
        """
        with DataModel(output_fp, read_only=False) as data:
            try:
                return data.handle.attrs["sim_spec"] == self.name
            except KeyError:
                data.handle.attrs["sim_spec"] = self.name
        return True


class Job():
    """Job to run embedding tasks.

    The results of the run are stored in a temporary file. It needs to merged with
    the main file after the job has run.

    Arguments
    ---------
    job_data:
        Dictionary containing information such as file/directory names, the window and the name
        of the job.
    compute_year:
        Whether the window needs to be created in the file. Otherwise it is read.
    compute_cpc:
        Whether the CPC correlations need to be created in the file.
    compute_models:
        List of models that need to be run.
    sim_spec:
        Specifications used for the run.
    """
    def __init__(self, job_data: Dict[str, Any],  # pylint: disable=too-many-arguments
                 need_window: bool, need_cpc: bool,
                 need_models: list[tuple[str, str]],
                 sim_spec: SimulationSpecification):
        self.job_data = job_data
        self.need_window = need_window
        self.need_cpc = need_cpc
        self.need_models = need_models
        self.sim_spec = sim_spec

        all_prep_names = list(set(x[0] for x in need_models))
        all_model_names = list(set(x[1] for x in need_models))

        with DataModel(self.job_data["output_fp"], read_only=True) as data:
            self.models = {model_name: data.load_model(model_name)
                           for model_name in all_model_names}
            self.preps = {prep_name: data.load_preprocessor(prep_name)
                          for prep_name in all_prep_names}
            # self.model, self.prep = data.load_model(self.job_data["name"])

    def get_patents(self, prep_name) -> List[Dict]:
        """Get the preprocessed patents.

        Returns
        -------
        patents:
            Preprocessed patents that are in the window.
        """
        # with DataModel(self.job_data["output_fp"], read_only=True) as data:
        # prep = data.load_preprocessor(prep_name)

        patents: List[Dict] = []
        for year in self.job_data["year_list"]:
            try:
                patents.extend(self.preps[prep_name].preprocess_file(
                    Path(self.job_data["patent_dir"]) / (str(year) + ".xz"),
                    self.sim_spec.debug_max_patents,
                    return_stats=False)
                )
            except FileNotFoundError:
                pass
        return patents

    def compute_patent_year(self, patents: List[Dict[str, Any]]) -> Tuple[npt.NDArray[np.int_],
                                                                          npt.NDArray[np.int_]]:
        """Compute the train/test patent numbers.

        Arguments
        ---------
        patents:
            Patents in the window.

        Returns
        -------
        Patent_id, year:
            Training and testing patent numbers.
        """
        # Get the train/test id's if they haven't been computed.
        if self.sim_spec.n_patents_per_window is None:
            patent_id = [pat["patent"] for pat in patents]
            year = [pat["year"] for pat in patents]
        else:
            idx = np.random.choice(
                len(patents),
                size=min(len(patents), self.sim_spec.n_patents_per_window))
            patent_id = [patents[i]["patent"] for i in idx]
            year = [patents[i]["year"] for i in idx]
        return np.array(patent_id), np.array(year)

    def compute_embeddings(self, model_name: str, documents: Sequence[str]) -> AllEmbedType:
        """Compute the embeddings.

        Arguments
        ---------
        documents:
            Documents for training/testing the models.

        Returns
        -------
        all_embeddings:
            Dictionary containing all embeddings for each model.
        """
        # all_embeddings = {}
        # for model_name in self.need_models:
        self.models[model_name].fit(documents)
        return self.models[model_name].transform(documents)
        # return all_embeddings

    def compute_cpc(self, test_id: IntSequence) -> Dict[str, Any]:
        """Compute the CPC classification correlations.

        Arguments
        ---------
        test_id:
            Patent numbers to compute the correlations for.

        Returns
        -------
        cpc_cor:
            Correlation of patents in the window.
        """
        avg_year = round(np.mean(self.job_data["year_list"]))
        test_id = np.array(test_id)
        pat_class = PatentClassification(self.job_data["cpc_fp"])
        cpc_cor = pat_class.sample_cpc_correlations(
            test_id, samples_per_patent=self.sim_spec.cpc_samples_per_patent,
            seed=avg_year)
        return cpc_cor

    def run(self) -> FileType:  # pylint: disable=too-many-locals
        """Run the job.

        Returns
        -------
        temp_fp:
            Path to the temporary file with the results.
        """
        temp_fp: FileType = self.job_data["temp_fp"]
        window_name = self.job_data["name"]

        # print("Do the run", [prep.logger.level for prep in self.preps.values()])

        last_prep = list(self.preps)[0]
        start_time = time()
        patents = self.get_patents(last_prep)
        print("read_patents", time()-start_time, self.job_data["name"])
        if self.need_window:
            patent_id, year = self.compute_patent_year(patents)
        else:
            # Retrieve the train/test id's if the have been computed.
            with DataModel(self.job_data["output_fp"], read_only=True) as data:
                patent_id, year = data.load_window(window_name)

        documents = [pat["contents"] for pat in patents if pat["patent"] in patent_id]
        print("compute_window", time()-start_time, self.job_data["name"])

        all_embeddings = {}
        for cur_prep, cur_model in self.need_models:
            if cur_prep != last_prep:
                patents = self.get_patents(cur_prep)
                documents = [pat["contents"] for pat in patents if pat["patent"] in patent_id]
                last_prep = cur_prep
            combi_name = f"{cur_prep}-{cur_model}"
            all_embeddings[combi_name] = self.compute_embeddings(cur_model, documents)

        print("get embeddings", time()-start_time, self.job_data["name"])

        # Compute the CPC correlations
        if self.need_cpc:
            cpc_cor = self.compute_cpc(patent_id)

        print("compute cpc", time()-start_time), self.job_data["name"]

        # Store the computed results to the temporary file.
        if not isinstance(temp_fp, io.BytesIO):
            Path(temp_fp).unlink(missing_ok=True)
        with DataModel(temp_fp, read_only=False) as data:
            if self.need_window:
                data.store_window(window_name, patent_id, year)
            if self.need_cpc:
                data.store_cpc_correlations(window_name, cpc_cor)
            for model_name, embeddings in all_embeddings.items():
                data.store_embeddings(window_name, model_name, embeddings)
        return temp_fp

    def __str__(self):
        return (f"{self.job_data['name']}, year: {self.need_window}, cpc: {self.need_cpc}"
                f", models: {self.need_models}")


def insert_models(models: Dict[str, BaseDocEmbedder],
                  preprocessors: dict[str, Preprocessor],
                  output_fp: FileType):
    """Store the information of models in the data file.

    Arguments
    ---------
    models:
        Model to store into the file.
    output_fp:
        File to store the model in.
    """
    with DataModel(output_fp, read_only=False) as data:
        for model_name, model in models.items():
            if not data.has_model(model_name):
                data.store_model(model_name, model)
        for prep_name, prep in preprocessors.items():
            if not data.has_prep(prep_name):
                data.store_preprocessor(prep_name, prep)


def _pool_worker(job):
    return job.run()


def run_jobs_multi(jobs: Sequence[Job],
                   output_fp: FileType,
                   n_jobs: int=10,
                   progress_bar: bool=True):
    """Run jobs with multiprocessing.

    Arguments
    ---------
    jobs:
        Jobs to be run in parallel.
    output_fp:
        File to store/load the results to/from.
    n_jobs:
        Number of jobs to be run simultaneously.
    progress_bar:
        Whether to display a progress bar.
    """
    if len(jobs) == 0:
        return

    # Process all jobs.
    all_files = []
    with Pool(processes=n_jobs) as pool:
        for temp_data_fp in tqdm(pool.imap_unordered(_pool_worker, jobs),
                                 total=len(jobs),
                                 disable=not progress_bar):
            all_files.append(temp_data_fp)

    # Merge the files with the main output file.
    with DataModel(output_fp, read_only=False) as data:
        for temp_data_fp in all_files:
            data.add_data(temp_data_fp, delete_copy=True)


def run_jobs_single(jobs: Sequence[Job],
                    output_fp: FileType,
                    progress_bar: bool=True):
    """Run jobs using a single thread/process.

    Arguments
    ---------
    jobs:
        Jobs to be run in parallel.
    output_fp:
        File to store/load the results to/from.
    n_jobs:
        Number of jobs to be run simultaneously.
    progress_bar:
        Whether to display a progress bar.
    """
    if len(jobs) == 0:
        return

    # Process all jobs.
    all_files = []
    for job in tqdm(jobs, disable=not progress_bar):
        temp_data_fp = job.run()
        all_files.append(temp_data_fp)

    # Merge the files with the main output file.
    with DataModel(output_fp, read_only=False) as data:
        for temp_data_fp in all_files:
            data.add_data(temp_data_fp, delete_copy=True)


def run_models(preprocessors: Optional[dict[str, Preprocessor]],  # pylint: disable=too-many-arguments
               models: dict[str, BaseDocEmbedder],
               sim_spec: SimulationSpecification,
               patent_dir: PathType,
               output_fp: FileType,
               cpc_fp: PathType,
               n_jobs: int=10,
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
    """
    if not sim_spec.check_file(output_fp):
        raise ValueError("Simulation specifications do not match existing specifications.")

    if preprocessors is None:
        preprocessors = {"default": Preprocessor()}
    insert_models(models, preprocessors, output_fp)
    jobs = sim_spec.create_jobs(output_fp, models, preprocessors, cpc_fp, patent_dir)
    if n_jobs == 1:
        run_jobs_single(jobs, output_fp, progress_bar=progress_bar)
    else:
        run_jobs_multi(jobs, output_fp, n_jobs=n_jobs, progress_bar=progress_bar)
