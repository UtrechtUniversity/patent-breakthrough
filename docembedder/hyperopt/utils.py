"""Utils for model parameter optimization using hyperopt."""

from typing import Optional, Any, Dict, Callable
from pathlib import Path
import io
from collections import defaultdict
import numpy as np
from docembedder.utils import run_models
from docembedder.utils import SimulationSpecification
from docembedder import DataModel
from docembedder.analysis2 import DocAnalysis
from docembedder.preprocessor.preprocessor import Preprocessor
from hyperopt import STATUS_OK, fmin, tpe, Trials


class ModelHyperopt():  # pylint: disable=too-many-instance-attributes
    """
    ModelHyperopt class
    contains objective and optimization functions for each model type
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

    def get_best(self):
        """
        returns best result of each optimizing function
        """
        return self.best

    def get_trials(self):
        """
        returns trial results of optimization process. eacht Trial contains:
        trials - a list of dictionaries representing everything about the search
        results - a list of dictionaries returned by 'objective' during the search
        losses() - a list of losses (float for each 'ok' trial)
        statuses() - a list of status strings
        """
        return self.trials

    def optimize(self,
                 label: str,
                 objective_function: Callable,
                 space: Dict,
                 max_evals: int = 10) -> None:
        """hyperopt optimization function"""        
        self.best[label] = fmin(objective_function,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=max_evals,
                                trials=self.trials[label])

    def get_objective_func(self, **kwargs) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """creates general loss function"""

        def objective_func(params: Dict[str, Any]) -> Dict[str, Any]:
            """hyperopt objective function"""
            label = kwargs['label']

            sim_spec = SimulationSpecification(year_start=self.year_start,
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
                       n_jobs=self.n_jobs)

            pp_prefix = "default-"

            with DataModel(output_fp, read_only=False) as data:
                analysis = DocAnalysis(data)
                correlations = analysis.cpc_correlations(models=f"{pp_prefix}{label}")

            return {'loss': (1 - np.mean(correlations[f"{pp_prefix}{label}"]["correlations"])),
                    'status': STATUS_OK}

        return objective_func
