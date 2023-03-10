"""Utils for model parameter optimization using hyperopt."""

from typing import Optional, Any, Dict, Callable, Type, Union, Iterable
from pathlib import Path
from collections import defaultdict
import pickle
import itertools

import numpy as np
from hyperopt import STATUS_OK, fmin, tpe, Trials, space_eval
from tqdm import tqdm
import pandas as pd

from docembedder.utils import SimulationSpecification
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.models.base import BaseDocEmbedder
from docembedder.typing import PathType
from docembedder.hyperopt.parallel import run_jobs, get_patent_data_multi


class ModelHyperopt():
    """
    ModelHyperopt class

    Objective and optimization functions, generalized for each model type.
    Class var 'best' contains the best result of each optimizing function.
    Class var 'trials' contains the results of optimization process. Each element contains:
    - trials - a list of dictionaries representing everything about the search
    - results - a list of dictionaries returned by 'objective' during the search
    - losses() - a list of losses (float for each 'ok' trial)
    - statuses() - a list of status strings

    """
    def __init__(self,  # pylint: disable=too-many-arguments
                 sim_spec: SimulationSpecification,
                 cpc_fp: Path,
                 patent_dir: Path,
                 preprocessor: Optional[Preprocessor] = None,
                 trials: Optional[Union[Dict[str, Trials], PathType]]=None,
                 ) -> None:
        self.sim_spec = sim_spec
        if preprocessor is None:
            preprocessor = Preprocessor()
        self.preprocessor = preprocessor
        self.cpc_fp = cpc_fp
        self.patent_dir = patent_dir
        self.best: Dict = defaultdict(fmin)
        self.pickle_fp = None
        if trials is None:
            self.trials: Dict = defaultdict(Trials)
        elif isinstance(trials, (str, Path)):
            self.pickle_fp = trials
            if Path(trials).is_file():
                with open(trials, "rb") as handle:
                    self.trials = pickle.load(handle)
            else:
                self.trials = defaultdict(Trials)
        else:
            self.trials = trials

    def optimize(  # pylint: disable=too-many-arguments
            self,
            label: str,
            model: Type[BaseDocEmbedder],
            max_evals: int = 10,
            n_jobs: int = 10) -> None:
        """Hyperopt optimization function"""

        if len(self.trials[label]) >= max_evals:
            return

        objective_function = self.get_objective_func(label=label, model=model, n_jobs=n_jobs)
        space = model.hyper_space()

        while len(self.trials[label]) < max_evals:
            new_evals = min(max_evals, len(self.trials[label])+10)
            self.best[label] = fmin(objective_function,
                                    space=space,
                                    algo=tpe.suggest,
                                    max_evals=new_evals,
                                    trials=self.trials[label])
            if self.pickle_fp is not None:
                with open(self.pickle_fp, "wb") as handle:
                    pickle.dump(self.trials, handle)

    def get_objective_func(self,
                           n_jobs: int=10,
                           **kwargs) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Creates general loss function"""
        prep = self.preprocessor

        documents, cpc_cor = get_patent_data_multi(self.sim_spec, prep, self.patent_dir,
                                                   self.cpc_fp, n_jobs=n_jobs)

        def objective_func(params: Dict[str, Any]) -> Dict[str, Any]:
            """Hyperopt objective function"""
            model = kwargs['model'](**params)
            correlation = run_jobs(model, documents, cpc_cor, n_jobs)
            return {"loss": -correlation, "status": STATUS_OK}
        return objective_func

    def dataframe(self, label, model_class):
        """Create a dataframe for the trial results with a label."""
        trial_results = defaultdict(list)
        for trial in self.trials[label]:
            if "loss" not in trial["result"]:
                continue
            new_trial = {key: val[0] for key, val in trial["misc"]["vals"].items()}
            conv_trial = space_eval(model_class.hyper_space(), new_trial)
            for key, val in conv_trial.items():
                trial_results[key].append(val)
            trial_results["loss"].append(trial["result"]["loss"])
        return pd.DataFrame(trial_results).sort_values("loss")

    def best_model(self, label, model_class):
        """Create the best model."""
        result_df = self.dataframe(label, model_class)
        best_prep_param = {
            key: list(value.values())[0] for key, value in result_df.head(1).to_dict().items()
            if key != "loss"}
        return model_class(**best_prep_param)


class PreprocessorHyperopt():
    """
    ModelHyperopt class

    Objective and optimization functions, generalized for each model type.
    Class var 'best' contains the best result of each optimizing function.
    Class var 'trials' contains the results of optimization process. Each element contains:
    - trials - a list of dictionaries representing everything about the search
    - results - a list of dictionaries returned by 'objective' during the search
    - losses() - a list of losses (float for each 'ok' trial)
    - statuses() - a list of status strings

    """
    def __init__(self,  # pylint: disable=too-many-arguments
                 sim_spec: SimulationSpecification,
                 cpc_fp: Path,
                 patent_dir: Path,
                 trials: Optional[Union[Dict, PathType]]=None,
                 ) -> None:
        self.sim_spec = sim_spec
        self.cpc_fp = cpc_fp
        self.patent_dir = patent_dir
        self.best: Dict = defaultdict(fmin)
        self.pickle_fp = None
        if trials is None:
            self.trials: Dict = defaultdict(list)
        elif isinstance(trials, dict):
            self.trials = trials
        elif isinstance(trials, (str, Path)):
            self.pickle_fp = trials
            if Path(trials).is_file():
                with open(trials, "rb") as handle:
                    self.trials = pickle.load(handle)
            else:
                self.trials = defaultdict(list)
        else:
            raise ValueError(f"Unknown trials type: '{type(trials)}'")

    def optimize(  # pylint: disable=too-many-locals
            self,
            label: str,
            model: BaseDocEmbedder,
            preprocessor: Type[Preprocessor],
            n_jobs: int = 10,
            **kwargs) -> None:
        """Optimize the preprocessor."""
        space = preprocessor.hyper_space()
        assert np.all([sub_space.name == "switch" for sub_space in space.values()])
        if len(space) > 0:
            all_settings: Iterable[tuple] = itertools.product(*[[True, False]
                                                                for _ in range(len(space))])
        else:
            all_settings = [()]  # type: ignore

        for setting in tqdm(all_settings, total=2**len(space)):
            params = {key: setting[i] for i, key in enumerate(space)}
            trial_done = False
            for trial in self.trials[label]:
                if params == trial["params"]:
                    trial_done = True
            if trial_done:
                continue
            prep = preprocessor(**params, **kwargs)
            documents, cpc_cor = get_patent_data_multi(
                self.sim_spec, prep,
                self.patent_dir, self.cpc_fp, n_jobs,
                progress_bar=False)
            correlation = run_jobs(model, documents, cpc_cor, n_jobs)
            self.trials[label].append({"loss": -correlation, "params": params})
            if self.pickle_fp is not None:
                with open(self.pickle_fp, "wb") as handle:
                    pickle.dump(self.trials, handle)

    def dataframe(self, label):
        """Create a dataframe for the trial results with a label."""
        data_dict = defaultdict(list)
        for trial in self.trials[label]:
            for key, value in trial["params"].items():
                data_dict[key].append(value)
            data_dict["loss"].append(trial["loss"])
        return pd.DataFrame(data_dict).sort_values("loss")

    def best_preprocessor(self, label, prep_class, **kwargs):
        """Create the best preprocessor."""
        result_df = self.dataframe(label)
        best_prep_param = {
            key: list(value.values())[0] for key, value in result_df.head(1).to_dict().items()
            if key != "loss"}
        return prep_class(**best_prep_param, **kwargs)
