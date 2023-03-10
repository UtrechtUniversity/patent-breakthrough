from pathlib import Path
import multiprocessing as mp

import io
from pytest import mark

from docembedder.utils import SimulationSpecification, run_models
from docembedder.models.tfidf import TfidfEmbedder
from docembedder.datamodel import DataModel
from docembedder.models.bpemb import BPembEmbedder
from docembedder.models.bert import BERTEmbedder
from docembedder.models.doc2vec import D2VEmbedder
from docembedder.models.countvec import CountVecEmbedder
from docembedder.hyperopt.utils import ModelHyperopt, PreprocessorHyperopt
from docembedder.hyperopt.parallel import get_patent_data_multi
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.preprocessor.oldprep import OldPreprocessor


@mark.parametrize(
    "model_class",
    [
        TfidfEmbedder,
        BPembEmbedder,
        BERTEmbedder,
        D2VEmbedder,
        CountVecEmbedder,
    ]
)
@mark.parametrize("n_jobs", [1, 2])
def test_run_models(model_class, n_jobs):
    sim_spec = SimulationSpecification(1850, 1880)
    patent_dir = Path("tests", "data", "input")
    cpc_fp = Path("tests", "data", "test_GPCPs.txt")
    label = "model"
    mp.set_start_method("spawn", force=True)
    hyper = ModelHyperopt(sim_spec, cpc_fp, patent_dir)
    hyper.optimize(label, model_class, max_evals=3, n_jobs=n_jobs)
    assert len(hyper.dataframe(label, model_class)) == 3
    assert isinstance(hyper.best_model(label, model_class), model_class)


@mark.parametrize(
    "prep_class,kwargs",
    [
        (Preprocessor, {}),
        (OldPreprocessor, {"list_path": Path("tests", "data")}),
    ]
)
@mark.parametrize("n_jobs", [1, 2])
def test_run_prep(prep_class, kwargs, n_jobs):
    sim_spec = SimulationSpecification(1850, 1880)
    patent_dir = Path("tests", "data", "input")
    cpc_fp = Path("tests", "data", "test_GPCPs.txt")
    label = "prep"
    mp.set_start_method("spawn", force=True)
    hyper = PreprocessorHyperopt(sim_spec, cpc_fp, patent_dir)
    hyper.optimize(label, TfidfEmbedder(), prep_class, n_jobs=n_jobs, **kwargs)
    n_param = len(prep_class.hyper_space())
    assert len(hyper.dataframe(label)) == 2**n_param
    assert isinstance(hyper.best_preprocessor(label, prep_class, **kwargs), prep_class)


def test_trials_models(tmp_path):
    trials_fp = Path(tmp_path, "test.pkl")
    sim_spec = SimulationSpecification(1850, 1880)
    patent_dir = Path("tests", "data", "input")
    cpc_fp = Path("tests", "data", "test_GPCPs.txt")
    label = "model"
    mp.set_start_method("spawn", force=True)
    hyper = ModelHyperopt(sim_spec, cpc_fp, patent_dir, trials=trials_fp)
    hyper.optimize(label, TfidfEmbedder, max_evals=1, n_jobs=1)
    assert len(hyper.dataframe(label, TfidfEmbedder)) == 1

    hyper = ModelHyperopt(sim_spec, cpc_fp, patent_dir, trials=trials_fp)
    assert len(hyper.dataframe(label, TfidfEmbedder)) == 1

    hyper.optimize(label, TfidfEmbedder, max_evals=2, n_jobs=1)
    assert len(hyper.dataframe(label, TfidfEmbedder)) == 2


def test_trials_prep(tmp_path):
    trials_fp = Path(tmp_path, "test.pkl")
    sim_spec = SimulationSpecification(1850, 1880)
    patent_dir = Path("tests", "data", "input")
    cpc_fp = Path("tests", "data", "test_GPCPs.txt")
    label = "model"
    mp.set_start_method("spawn", force=True)
    hyper = PreprocessorHyperopt(sim_spec, cpc_fp, patent_dir, trials=trials_fp)
    hyper.optimize(label, TfidfEmbedder(), OldPreprocessor, n_jobs=1,
                   list_path=Path("tests", "data"))
    assert len(hyper.dataframe(label)) == 1

    hyper = PreprocessorHyperopt(sim_spec, cpc_fp, patent_dir, trials=trials_fp)
    assert len(hyper.dataframe(label)) == 1
