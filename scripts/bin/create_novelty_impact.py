#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np

from docembedder.analysis import compute_impact_novelty
from docembedder.embedding_utils import load_embeddings


def parse_arguments():
    parser = ArgumentParser(
        prog="create_novelty_impact.py",
        description="Create novelty and impact scores")
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--embedding_fp", required=True, type=Path)
    parser.add_argument("--window_fp", required=True, type=Path)
    return parser.parse_args()


def load_window(window_fp):
    with open(window_fp, "rb") as handle:
        patent_ids = np.load(handle)
        patent_years = np.load(handle)
    return patent_ids, patent_years


if __name__ == "__main__":
    args = parse_arguments()
    args.output_csv.parent.mkdir(exist_ok=True, parents=True)
    embeddings = load_embeddings(args.embedding_fp)
    patent_ids, patent_year = load_window(args.window_fp)
    results = compute_impact_novelty(embeddings, patent_ids, patent_year, exponents=[1.0, 2.0, 3.0],
                                     n_jobs=8)

    impact_novel = {}
    for expon, res in results.items():
        impact_novel["patent_ids"] = res["patent_ids"]
        impact_novel["year"] = res["year"]
        impact_novel[f"impact-{expon}"] = res["impact"]
        impact_novel[f"novelty-{expon}"] = res["novelty"]
    pd.DataFrame(impact_novel).to_csv(args.output_csv, index=False)
