#!/usr/bin/env python

from pathlib import Path
import pandas as pd

from docembedder.analysis import DocAnalysis
from docembedder.datamodel import DataModel
from collections import defaultdict
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser(
        prog="create_novelty_impact.py",
        description="Create novelty and impact scores")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding", required=True)
    return parser.parse_args()


def compute_impacts(embedding_fp: Path, output_dir: Path):
    exponents = [1.0, 2.0, 3.0]

    # Calculate impacts and novelties
    impact_novel = defaultdict(lambda: defaultdict(list))
    with DataModel(embedding_fp, read_only=False) as data:
        analysis = DocAnalysis(data)
        for window, model in data.iterate_window_models():
            results = analysis.impact_novelty_results(
                window, model, exponents, cache=False, n_jobs=8)
            for expon, res in results.items():
                if expon == exponents[0]:
                    impact_novel[model]["patent_ids"].extend(res["patent_ids"])
                impact_novel[model][f"impact-{expon}"].extend(res["impact"])
                impact_novel[model][f"novelty-{expon}"].extend(res["novelty"])

    output_dir.mkdir(exist_ok=True, parents=True)
    for model, data in impact_novel.items():
        classifier_name = model.split("-")[-1]
        impact_fp = Path(output_dir, f"impact-{classifier_name}.csv")
        pd.DataFrame(impact_novel[model]).sort_values("patent_ids").to_csv(
            impact_fp, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    compute_impacts(Path(args.embedding), Path(args.output_dir))
