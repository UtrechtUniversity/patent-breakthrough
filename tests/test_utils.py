from pathlib import Path
import multiprocessing as mp

import io
from pytest import mark

from docembedder.utils import SimulationSpecification, run_models
from docembedder.models.tfidf import TfidfEmbedder
from docembedder.datamodel import DataModel


@mark.parametrize("n_jobs", [1, 2])
def test_run_models(n_jobs):
    models = {"tfidf": TfidfEmbedder()}
    sim_spec = SimulationSpecification(1850, 1880)
    patent_dir = Path("tests", "data", "input")
    hdf5_file = io.BytesIO()
    cpc_fp = Path("tests", "data", "test_GPCPs.txt")
    mp.set_start_method("spawn", force=True)
    run_models(None, models, sim_spec, patent_dir, output_fp=hdf5_file, cpc_fp=cpc_fp,
               n_jobs=n_jobs)
    with DataModel(hdf5_file) as data:
        assert data.has_model("tfidf")
        assert len(data.window_list) == 1
        assert len(data.model_names) == 1
        assert len(list(data.iterate_window_models())) == 1
