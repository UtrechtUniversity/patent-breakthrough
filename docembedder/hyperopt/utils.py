from typing import Optional, Any, Dict
from pathlib import Path
import json
from hashlib import md5
import io
import numpy as np
from docembedder.utils import run_models
from docembedder.models import TfidfEmbedder, D2VEmbedder, CountVecEmbedder, BPembEmbedder, \
    BERTEmbedder
from docembedder.utils import SimulationSpecification
from docembedder import DataModel
from docembedder.analysis2 import DocAnalysis
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.models.base import BaseDocEmbedder
from hyperopt import STATUS_OK, fmin, tpe


class ModelHyperopt():  # pylint: disable=too-many-instance-attributes

    def __init__(self,  # pylint: disable=too-many-arguments
                 year_start: int,
                 year_end: int,
                 window_size: int,
                 n_jobs: int,
                 preprocessors: Optional[Dict[str, Preprocessor]],
                 cpc_fp: Path,
                 patent_dir: Path
                 ) -> None:
        self.year_start = year_start
        self.year_end = year_end
        self.window_size = window_size
        self.n_jobs = n_jobs
        self.preprocessors = preprocessors
        self.cpc_fp = cpc_fp
        self.patent_dir = patent_dir
        self.best: Dict = {}

    def optimize_tfidf(self, max_evals: int = 10) -> None:
        self.best['tfidf'] = fmin(self._tfidf_objective_func,
                                  space=TfidfEmbedder.hyper_space(),
                                  algo=tpe.suggest,
                                  max_evals=max_evals)

    def optimize_d2v(self, max_evals: int = 10) -> None:
        self.best['d2v'] = fmin(self._d2v_objective_func,
                                space=D2VEmbedder.hyper_space(),
                                algo=tpe.suggest,
                                max_evals=max_evals)

    def optimize_countvec(self, max_evals: int = 10) -> None:
        self.best['countvec'] = fmin(self._countvec_objective_func,
                                     space=CountVecEmbedder.hyper_space(),
                                     algo=tpe.suggest,
                                     max_evals=max_evals)

    def optimize_bpemp(self, max_evals: int = 10) -> None:
        self.best['bpemp'] = fmin(self._bpemp_objective_func,
                                  space=BPembEmbedder.hyper_space(),
                                  algo=tpe.suggest,
                                  max_evals=max_evals)

    def optimize_bert(self, max_evals: int = 10) -> None:
        self.best['bert'] = fmin(self._bert_objective_func,
                                 space=BERTEmbedder.hyper_space(),
                                 algo=tpe.suggest,
                                 max_evals=max_evals)

    @classmethod
    def hash_params(cls, prefix: str, params: Dict[str, Any]) -> str:
        return prefix + md5(json.dumps(params).encode('utf-8')).hexdigest()

    def _general_objective_func(self, label: str, model: BaseDocEmbedder) -> Dict[str, Any]:
        models = {label: model}
        sim_spec = SimulationSpecification(year_start=self.year_start,
                                           year_end=self.year_end, window_size=self.window_size)

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
            label=self.hash_params(prefix="tfidf-", params=params),
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
            label=self.hash_params(prefix="d2v-", params=params),
            model=D2VEmbedder(
                vector_size=params["vector_size"],
                min_count=params["min_count"],
                epoch=params["epoch"]
            ))

    def _countvec_objective_func(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._general_objective_func(
            label=self.hash_params(prefix="countvec-", params=params),
            model=CountVecEmbedder(method=params["method"]
                                   ))

    def _bpemp_objective_func(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._general_objective_func(
            label=self.hash_params(prefix="bpemp-", params=params),
            model=BPembEmbedder(
                vector_size=params["vector_size"],
                vocab_size=params["vocab_size"]
            ))

    def _bert_objective_func(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._general_objective_func(
            label=self.hash_params(prefix="bert-", params=params),
            model=BERTEmbedder(pretrained_model=params["pretrained_model"]
                               ))
