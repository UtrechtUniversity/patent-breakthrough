"""Utilities for working with embeddings."""

import numpy as np


def _gather_results(raw_results):
    # Reformat list of results.
    all_keys = [key for key in raw_results[0] if key != "exponent"]
    gathered_results = {}
    all_exponents = np.unique([x["exponent"] for x in raw_results])
    for expon in all_exponents:
        new_res = {}
        for key in all_keys:
            new_res[key] = np.concatenate([x[key] for x in raw_results if x["exponent"] == expon])
        gathered_results[expon] = new_res
    return gathered_results
