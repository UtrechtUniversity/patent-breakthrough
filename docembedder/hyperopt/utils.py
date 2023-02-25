"""Utils for model parameter optimization using hyperopt."""

from typing import Optional, Any, Dict, Callable
from pathlib import Path
from collections import defaultdict
import pickle

from hyperopt import STATUS_OK, fmin, tpe, Trials

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
                 preprocessors: Optional[Dict[str, Preprocessor]],
                 cpc_fp: Path,
                 patent_dir: Path,
                 ) -> None:
        self.sim_spec = sim_spec
        self.preprocessors = preprocessors
        self.cpc_fp = cpc_fp
        self.patent_dir = patent_dir
        self.best: Dict = defaultdict(fmin)
        self.trials: Dict = defaultdict(Trials)

    def optimize(self,  # pylint: disable=too-many-arguments
                 label: str,
                 objective_function: Optional[Callable]=None,
                 space: Optional[Dict]=None,
                 model: Optional[BaseDocEmbedder]=None,
                 max_evals: int = 10,
                 n_jobs: int = 10,
                 pickle_fp: Optional[PathType] = None) -> None:
        """Hyperopt optimization function"""

        if objective_function is None:
            if model is None:
                raise ValueError("Either give objective function or model.")
            objective_function = self.get_objective_func(label=label, model=model, n_jobs=n_jobs)
        if space is None:
            if model is None:
                raise ValueError("Either give space or model.")
            space = model.hyper_space()

        if pickle_fp is None:
            self.best[label] = fmin(objective_function,
                                    space=space,
                                    algo=tpe.suggest,
                                    max_evals=max_evals,
                                    trials=self.trials[label])
        else:
            while len(self.trials[label]) < max_evals:
                new_evals = min(max_evals, len(self.trials[label])+10)
                self.best[label] = fmin(objective_function,
                                        space=space,
                                        algo=tpe.suggest,
                                        max_evals=new_evals,
                                        trials=self.trials[label])
                with open(pickle_fp, "wb") as handle:
                    pickle.dump(self, handle)

    def get_objective_func(self,
                           n_jobs: int=10,
                           **kwargs) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Creates general loss function"""
        prep = Preprocessor()

        documents, cpc_cor = get_patent_data_multi(self.sim_spec, prep, self.patent_dir,
                                                   self.cpc_fp, n_jobs=n_jobs)

        def objective_func(params: Dict[str, Any]) -> Dict[str, Any]:
            """Hyperopt objective function"""
            model = kwargs['model'](**params)
            correlation = run_jobs(model, documents, cpc_cor, n_jobs)
            return {"loss": -correlation, "status": STATUS_OK}
        return objective_func
