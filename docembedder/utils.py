"""General utilities for performing runs."""

from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, List, Tuple

import numpy as np
from tqdm import tqdm

from docembedder import DataModel
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.classification import PatentClassification
from docembedder.models.base import PathType, BaseDocEmbedder, AllEmbedType


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

    def create_jobs(self, output_fp: PathType,
                    models: Dict[str, BaseDocEmbedder],
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
        cur_start = self.year_start - ((self.year_start-STARTING_YEAR) % self.window_size)
        cur_end = self.year_start + self.window_size
        delta = (self.window_size+1)//2
        with DataModel(output_fp, read_only=True) as data:
            while cur_start < self.year_end:
                year_list = list(range(cur_start, cur_end))
                job_name = f"{cur_start}-{cur_end-1}"
                compute_year = not data.has_year(job_name)
                compute_cpc = not data.has_cpc(job_name)
                compute_models = [model for model in models if not data.has_run(model, job_name)]
                if compute_year or compute_cpc or compute_models:
                    jobs.append(Job(
                        job_data={
                            "name": job_name,
                            "output_fp": output_fp, "cpc_fp": cpc_fp,
                            "temp_fp": Path(output_fp).parent / f"temp_{job_name}.h5",
                            "year_list": year_list,
                            "patent_dir": patent_dir,
                        },
                        need_window=compute_year,
                        need_cpc=compute_cpc,
                        need_models=compute_models,
                        sim_spec=self,
                    ))
                cur_start += delta
                cur_end += delta
        return jobs

    @property
    def name(self) -> str:
        """Identifier of the simulation specifications.

        This is mainly used to check whether a new run is compatible.
        Different values for `year_start` and `year_end` should be compatible.
        """
        return (f"s{STARTING_YEAR}-w{self.window_size}-"
                f"c{self.cpc_samples_per_patent}-d{self.debug_max_patents}-"
                f"n{self.n_patents_per_window}")

    def check_file(self, output_fp: PathType) -> bool:
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
                 need_window: bool, need_cpc: bool, need_models: Sequence[str],
                 sim_spec: SimulationSpecification):
        self.job_data = job_data
        self.need_window = need_window
        self.need_cpc = need_cpc
        self.need_models = need_models
        self.sim_spec = sim_spec

    def get_patents(self) -> List[Dict]:
        """Get the preprocessed patents.

        Returns
        -------
        patents:
            Preprocessed patents that are in the window.
        """
        prep = Preprocessor()
        patents: List[Dict] = []
        for year in self.job_data["year_list"]:
            try:
                patents.extend(prep.preprocess_file(
                    Path(self.job_data["patent_dir"]) / (str(year) + ".xz"),
                    self.sim_spec.debug_max_patents,
                    return_stats=False)
                )
            except FileNotFoundError:
                pass
        return patents

    def compute_train_test(self, patents: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
        """Compute the train/test patent numbers.

        Arguments
        ---------
        patents:
            Patents in the window.

        Returns
        -------
        train_id, test_id:
            Training and testing patent numbers.
        """
        # Get the train/test id's if they haven't been computed.
        if self.sim_spec.n_patents_per_window is None:
            train_id = [pat["patent"] for pat in patents]
        else:
            train_id = np.random.choice(
                [pat["patent"] for pat in patents],
                size=min(len(patents), self.sim_spec.n_patents_per_window)).tolist()
        test_id = train_id
        return train_id, test_id

    def compute_embeddings(self, train_documents, test_documents) -> Dict[str, AllEmbedType]:
        """Compute the embeddings.

        Arguments
        ---------
        train_documents:
            Documents for training the models.
        test_documents:
            Documents for testing the models.

        Returns
        -------
        all_embeddings:
            Dictionary containing all embeddings for each model.
        """
        all_embeddings = {}
        for model_name in self.need_models:
            with DataModel(self.job_data["output_fp"], read_only=True) as data:
                model = data.load_model(model_name)
            model.fit(train_documents)
            all_embeddings[model_name] = model.transform(test_documents)
        return all_embeddings

    def compute_cpc(self, test_id: Sequence[int]) -> Dict[str, Any]:
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
        pat_class = PatentClassification(self.job_data["cpc_fp"])
        cpc_cor = pat_class.sample_cpc_correlations(
            test_id, samples_per_patent=self.sim_spec.cpc_samples_per_patent)
        return cpc_cor

    def run(self) -> Path:
        """Run the job.

        Returns
        -------
        temp_fp:
            Path to the temporary file with the results.
        """
        temp_fp = self.job_data["temp_fp"]
        name = self.job_data["name"]

        patents = self.get_patents()

        if self.need_window:
            train_id, test_id = self.compute_train_test(patents)
        else:
            # Retrieve the train/test id's if the have been computed.
            with DataModel(self.job_data["output_fp"], read_only=True) as data:
                train_id, test_id = data.get_train_test_id(name)

        train_documents = [pat["contents"] for pat in patents if pat["patent"] in train_id]
        test_documents = [pat["contents"] for pat in patents if pat["patent"] in test_id]

        all_embeddings = self.compute_embeddings(train_documents, test_documents)

        # Compute the CPC correlations
        if self.need_cpc:
            cpc_cor = self.compute_cpc(test_id)

        # Store the computed results to the temporary file.
        temp_fp.unlink(missing_ok=True)
        with DataModel(temp_fp, read_only=False) as data:
            if self.need_window:
                data.store_year(name, test_id, train_id)
            if self.need_cpc:
                data.store_cpc_correlations(name, cpc_cor)
            for model_name, embeddings in all_embeddings.items():
                data.store_embeddings(name, model_name, embeddings)
        return temp_fp

    def __str__(self):
        return (f"{self.job_data['name']}, year: {self.need_window}, cpc: {self.need_cpc}"
                f", models: {self.need_models}")


def insert_models(models: Dict[str, BaseDocEmbedder], output_fp: PathType):
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


def _pool_worker(job):
    return job.run()


def run_jobs_multi(jobs: Sequence[Job],
                   output_fp: PathType,
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


def run_models(models: Dict[str, BaseDocEmbedder],  # pylint: disable=too-many-arguments
               sim_spec: SimulationSpecification,
               patent_dir: PathType,
               output_fp: PathType,
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

    insert_models(models, output_fp)
    jobs = sim_spec.create_jobs(output_fp, models, cpc_fp, patent_dir)
    run_jobs_multi(jobs, output_fp, n_jobs=n_jobs, progress_bar=progress_bar)
