"""Utils for model parameter optimization using hyperopt."""


from typing import Optional, Any, Dict
from pathlib import Path
import io
import numpy as np
from collections import defaultdict
from docembedder.utils import run_models
from docembedder.models import TfidfEmbedder, D2VEmbedder, CountVecEmbedder, BPembEmbedder, \
    BERTEmbedder
from docembedder.utils import SimulationSpecification
from docembedder import DataModel
from docembedder.analysis2 import DocAnalysis
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.models.base import BaseDocEmbedder
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

    def optimize_tfidf(self, max_evals: int = 10) -> None:
        """
        optimization function for Tfidf-model
        """
        self.best['tfidf'] = fmin(self._tfidf_objective_func,
                                  space=TfidfEmbedder.hyper_space(),
                                  algo=tpe.suggest,
                                  max_evals=max_evals,
                                  trials=self.trials['tfidf'])

    def optimize_d2v(self, max_evals: int = 10) -> None:
        """
        optimization function for Doc2Vec-model
        """
        self.best['d2v'] = fmin(self._d2v_objective_func,
                                space=D2VEmbedder.hyper_space(),
                                algo=tpe.suggest,
                                max_evals=max_evals,
                                trials=self.trials['d2v'])

    def optimize_countvec(self, max_evals: int = 10) -> None:
        """
        optimization function for CountVec-model
        """
        self.best['countvec'] = fmin(self._countvec_objective_func,
                                     space=CountVecEmbedder.hyper_space(),
                                     algo=tpe.suggest,
                                     max_evals=max_evals,
                                     trials=self.trials['countvec'])

    def optimize_bpemp(self, max_evals: int = 10) -> None:
        """
        optimization function for BPEmb-model
        """
        self.best['bpemp'] = fmin(self._bpemp_objective_func,
                                  space=BPembEmbedder.hyper_space(),
                                  algo=tpe.suggest,
                                  max_evals=max_evals,
                                  trials=self.trials['bpemp'])

    def optimize_bert(self, max_evals: int = 10) -> None:
        """
        optimization function for Bert-models
        """
        self.best['bert'] = fmin(self._bert_objective_func,
                                 space=BERTEmbedder.hyper_space(),
                                 algo=tpe.suggest,
                                 max_evals=max_evals,
                                 trials=self.trials['bert'])

    def _general_objective_func(self, label: str, model: BaseDocEmbedder) -> Dict[str, Any]:
        models = {label: model}
        sim_spec = SimulationSpecification(year_start=self.year_start,
                                           year_end=self.year_end, 
                                           window_size=self.window_size,
                                           debug_max_patents=self.debug_max_patents)

        output_fp = io.BytesIO()

        run_models(preprocessors=self.preprocessors, models=models, sim_spec=sim_spec,
                   patent_dir=self.patent_dir, output_fp=output_fp, cpc_fp=self.cpc_fp,
                   n_jobs=self.n_jobs)

        pp_prefix = "default-"

        with DataModel(output_fp, read_only=False) as data:
            analysis = DocAnalysis(data)
            correlations = analysis.cpc_correlations(models=f"{pp_prefix}{label}")

        return {'loss': (1 - np.mean(correlations[f"{pp_prefix}{label}"]["correlations"])),
                'status': STATUS_OK}

    def _tfidf_objective_func(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._general_objective_func(
            label="tfidf",
            model=TfidfEmbedder(
                max_df=params["max_df"],
                min_df=params["min_df"],
                ngram_max=params["ngram_max"],
                norm=params["norm"],
                stem=params["stem"],
                stop_words=params["stop_words"],
                sublinear_tf=params["sublinear_tf"]
            ))

    def _d2v_objective_func(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._general_objective_func(
            label="d2v",
            model=D2VEmbedder(
                vector_size=params["vector_size"],
                min_count=params["min_count"],
                epoch=params["epoch"]
            ))

    def _countvec_objective_func(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._general_objective_func(
            label="countvec",
            model=CountVecEmbedder(method=params["method"]
                                   ))

    def _bpemp_objective_func(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._general_objective_func(
            label="bpemp",
            model=BPembEmbedder(
                vector_size=params["vector_size"],
                vocab_size=params["vocab_size"]
            ))

    def _bert_objective_func(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._general_objective_func(
            label="bert",
            model=BERTEmbedder(pretrained_model=params["pretrained_model"]
                               ))