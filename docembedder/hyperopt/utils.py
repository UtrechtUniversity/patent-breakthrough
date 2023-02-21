"""Utils for model parameter optimization using hyperopt."""

from typing import Optional, Any, Dict, Callable
from pathlib import Path
import io
from collections import defaultdict
import numpy as np
from hyperopt import STATUS_OK, fmin, tpe, Trials
from docembedder.utils import run_models
from docembedder.utils import SimulationSpecification
from docembedder import DataModel
from docembedder.analysis2 import DocAnalysis
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.models.base import BaseDocEmbedder
from docembedder.typing import PathType
import pickle


class ModelHyperopt():  # pylint: disable=too-many-instance-attributes
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
                 year_start: int,
                 year_end: int,
                 window_size: int,
                 n_jobs: int,
                 preprocessors: Optional[Dict[str, Preprocessor]],
                 cpc_fp: Path,
                 patent_dir: Path,
                 debug_max_patents: Optional[int]=1000,
                 ) -> None:
        self.year_start = year_start
        self.year_end = year_end
        self.window_size = window_size
        self.n_jobs = n_jobs
        self.debug_max_patents = debug_max_patents
        self.preprocessors = preprocessors
        self.cpc_fp = cpc_fp
        self.patent_dir = patent_dir
        self.best: Dict = defaultdict(fmin)
        self.trials: Dict = defaultdict(Trials)

    def optimize(self,
                 label: str,
                 objective_function: Optional[Callable]=None,
                 space: Optional[Dict]=None,
                 model: Optional[BaseDocEmbedder]=None,
                 max_evals: int = 10,
                 pickle_fp: Optional[PathType] = None) -> None:
        """Hyperopt optimization function"""

        if objective_function is None:
            if model is None:
                raise ValueError("Either give objective function or model.")
            objective_function = self.get_objective_func(label=label, model=model)
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

    def get_objective_func(self, **kwargs) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Creates general loss function"""

        def objective_func(params: Dict[str, Any]) -> Dict[str, Any]:
            """Hyperopt objective function"""
            label = kwargs['label']

            sim_spec = SimulationSpecification(
                year_start=self.year_start,
                year_end=self.year_end,
                window_size=self.window_size,
                debug_max_patents=self.debug_max_patents)

            output_fp = io.BytesIO()

            run_models(preprocessors=self.preprocessors,
                       models={label: kwargs['model'](**params)},
                       sim_spec=sim_spec,
                       patent_dir=self.patent_dir,
                       output_fp=output_fp,
                       cpc_fp=self.cpc_fp,
                       n_jobs=self.n_jobs,
                       progress_bar=False)

            pp_prefix = "default-"

            with DataModel(output_fp, read_only=False) as data:
                analysis = DocAnalysis(data)
                correlations = analysis.cpc_correlations(models=f"{pp_prefix}{label}")

            return {'loss': -np.mean(correlations[f"{pp_prefix}{label}"]["correlations"]),
                    'status': STATUS_OK}

        return objective_func
