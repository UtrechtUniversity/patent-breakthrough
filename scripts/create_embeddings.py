#!/usr/bin/env python
from argparse import ArgumentParser
import json
from enum import IntEnum
from pathlib import Path

import numpy as np
from docembedder.models.utils import create_model, create_preprocessor
from docembedder.utils import np_save
from docembedder.typing import AllEmbedType, FileType
from scipy.sparse import csr_matrix


class EmbedType(IntEnum):
    ARRAY = 1
    CSR_MATRIX = 2


def store_embeddings(embeddings: AllEmbedType,
                     embedding_fp: FileType):
    """Store embeddings for a window/year.

    Arguments
    ---------
    window_name:
        Year or window name.
    model_name:
        Name of the model that generated the embeddings.
    overwrite:
        If True, overwrite embeddings if they exist.
    """
    if not isinstance(embeddings, (np.ndarray, csr_matrix)):
        raise ValueError(f"Not implemented datatype {type(embeddings)}")

    if isinstance(embeddings, np.ndarray):
        np_save(embedding_fp, EmbedType.ARRAY, embeddings)
    elif isinstance(embeddings, csr_matrix):
        np_save(embedding_fp, EmbedType.CSR_MATRIX, embeddings.data, embeddings.indices,
                embeddings.indptr, embeddings.shape)


def generate_documents(patent_dir, window_fp, prep):
    with open(window_fp, "rb") as handle:
        patent_ids = np.load(handle)
        years = np.load(handle)
    unique_years = np.unique(years)
    for year in unique_years:
        n_pats = len(patent_ids[years == year])
        patents = prep.preprocess_file(Path(patent_dir) / f"{year}.xz", n_pats)
        for pat in patents:
            yield pat["contents"]


def parse_model_file(model_fp):
    with open(model_fp, "r") as handle:
        model_dict = json.load(handle)
    model_dict["prep"] = create_preprocessor(model_dict["prep_type"],
                                             model_dict["prep_params"])
    model_dict["classifier"] = create_model(model_dict["classifier_type"],
                                            model_dict["classifier_params"])
    return model_dict


def parse_arguments():
    parser = ArgumentParser(
        prog="create_embeddings.py",
        description="Create embeddings for a window."
    )
    parser.add_argument("--patent_dir", required=True)
    parser.add_argument("--window_fp", required=True)
    parser.add_argument("--model_fp", required=True)
    parser.add_argument("--output_fp", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_dict = parse_model_file(args.model_fp)
    documents = generate_documents(args.patent_dir, args.window_fp, model_dict["prep"])
    embeddings = model_dict["classifier"].fit_transform(documents)
    Path(args.output_fp).parent.mkdir(exist_ok=True, parents=True)
    store_embeddings(embeddings, args.output_fp)
