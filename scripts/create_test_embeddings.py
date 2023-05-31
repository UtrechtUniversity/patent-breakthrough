#!/usr/bin/env python
from configparser import ConfigParser
from pathlib import Path
from docembedder.simspec import SimulationSpecification
from docembedder.models import TfidfEmbedder
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.models.doc2vec import D2VEmbedder
from docembedder.utils import run_models


if __name__ == "__main__":
    year_start = 1838
    year_end = 1844
    # set simulation specification
    sim_spec = SimulationSpecification(
        year_start=year_start,
        year_end=year_end,
        window_size=4,
        window_shift=1,
        debug_max_patents=100,
    )

    # read local config
    config = ConfigParser()
    _ = config.read("setup.ini")
    patent_dir = Path(config["DATA"]["patent_dir"])
    output_fp = Path("test.h5")
    cpc_fp = Path(config["DATA"]["cpc_file"])

    model_tfidf = {
        "tfidf": TfidfEmbedder(),
    }
    prep_tfidf = {
        "prep-tfidf": Preprocessor()
    }
    model_doc2vec = {
        "doc2vec": D2VEmbedder()
    }
    prep_doc2vec = {
        "prep-doc2vec": Preprocessor()
    }

    for year in range(year_start, year_end):
        if not (patent_dir / f"{year}.xz").is_file():
            raise ValueError(f"Please download patent file {year}.xz and put it in"
                             f"the right directory ({patent_dir})")

    run_models(prep_tfidf, model_tfidf, sim_spec, patent_dir, output_fp, cpc_fp, n_jobs=8)
    run_models(prep_doc2vec, model_doc2vec, sim_spec, patent_dir, output_fp, cpc_fp, n_jobs=8)
