#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

from docembedder.simspec import SimulationSpecification
from docembedder.models import TfidfEmbedder
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.models.doc2vec import D2VEmbedder
from docembedder.utils import run_models


def parse_arguments():
    parser = ArgumentParser(
        prog="create_novelty_impact.py",
        description="Create novelty and impact scores")
    parser.add_argument("--patent_dir", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--cpc_fp", required=True)
    # parser.add_argument("--year_start", required=True, type=int)
    # parser.add_argument("--year_end", required=True, type=int)
    return parser.parse_args()


def compute_embeddings(patent_dir, output_fp, cpc_fp):
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


if __name__ == "__main__":
    args = parse_arguments()
    compute_embeddings(Path(args.patent_dir), Path(args.embedding),
                       Path(args.cpc_fp))
