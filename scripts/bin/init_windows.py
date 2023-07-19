#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser
from docembedder.simspec import SimulationSpecification
from docembedder.preprocessor.parser import read_xz
from docembedder.utils import np_save


def get_patent_ids(sim_spec, patent_dir):
    patent_id_by_year = {}
    patent_id_by_window = {}
    for year_list in sim_spec.year_ranges:
        patent_ids = []
        years = []
        for year in year_list:
            if year not in patent_id_by_year:
                patents = read_xz(Path(patent_dir, f"{year}.xz"))
                cur_ids = [pat["patent"] for pat in patents if len(pat["contents"]) > 0]
                if sim_spec.debug_max_patents is not None:
                    cur_ids = cur_ids[:sim_spec.debug_max_patents]
                patent_id_by_year[year] = cur_ids
            patent_ids.extend(patent_id_by_year[year])
            years.extend(len(patent_id_by_year[year])*[year])
        window_name = f"{year_list[0]}-{year_list[-1]}"
        patent_id_by_window[window_name] = (patent_ids, years)
    return patent_id_by_window


def parse_arguments():
    parser = ArgumentParser(
        prog="init.py",
        description="Initialize the run with the right parameters.")

    parser.add_argument("--settings", required=True)
    parser.add_argument("--patent_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sim_spec = SimulationSpecification.from_json(args.settings)
    patent_id_by_window = get_patent_ids(sim_spec, args.patent_dir)
    for window_name, (patent_ids, year) in patent_id_by_window.items():
        window_dir = Path(args.output_dir, "windows")
        window_dir.mkdir(exist_ok=True, parents=True)
        np_save(window_dir / f"{window_name}.npy", patent_ids, year)
