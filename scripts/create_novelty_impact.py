#!/usr/bin/env python

from configparser import ConfigParser
from pathlib import Path
from docembedder.analysis import DocAnalysis
from docembedder.datamodel import DataModel
from collections import defaultdict
import pandas as pd

if __name__ == "__main__":
    year_start = 1838
    year_end = 1844
    exponents = [1.0, 2.0, 3.0]

    # read local config
    config = ConfigParser()
    _ = config.read("setup.ini")
    patent_dir = Path(config["DATA"]["patent_dir"])
    output_fp = Path("test.h5")
    impact_fp = Path("impact.csv")
    cpc_fp = Path(config["DATA"]["cpc_file"])

    # Calculate impacts and novelties
    impact_novel = defaultdict(lambda: defaultdict(list))
    with DataModel(output_fp, read_only=False) as data:
        analysis = DocAnalysis(data)
        for window, model in data.iterate_window_models():
            results = analysis.impact_novelty_results(
                window, model, exponents, n_jobs=8)
            for expon, res in results.items():
                if expon == exponents[0]:
                    impact_novel[model]["patent_ids"].extend(res["patent_ids"])
                impact_novel[model][f"impact-{expon}"].extend(res["impact"])
                impact_novel[model][f"novelty-{expon}"].extend(res["novelty"])

    for model, data in impact_novel.items():
        impact_fp = Path(f"impact-{model}.csv")
        pd.DataFrame(impact_novel[model]).sort_values("patent_ids").to_csv(impact_fp, index=False)
