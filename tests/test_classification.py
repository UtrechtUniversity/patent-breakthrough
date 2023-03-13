from pathlib import Path

import pytest
import numpy as np

from docembedder.classification import PatentClassification


data_dir = Path("tests", "data")
class_fp = data_dir / "test_GPCPs.txt"
raw_fp = data_dir / "raw_test_combined.txt"

year_lookup = {100001: 1960, 100002: 1961, 100003: 1962,
               100004: 1960, 100005: 1961, 100006: 1962}


def check_correlations(cor_dict, n_samples):
    assert isinstance(cor_dict["i_patents"][0], np.int_)
    assert isinstance(cor_dict["j_patents"][0], np.int_)
    assert isinstance(cor_dict["correlations"][0], np.float_)
    for data in cor_dict.values():
        assert len(data) == n_samples

    assert np.all((cor_dict["i_patents"] >= 0) & (cor_dict["i_patents"] < 6))
    assert np.all((cor_dict["j_patents"] >= 0) & (cor_dict["j_patents"] < 6))
    assert np.all((cor_dict["correlations"] >= 0) & (cor_dict["correlations"] <= 1))


def test_classification():
    pc = PatentClassification(class_fp)
    assert pc.get_similarity(100001, 100001) == 1
    assert pc.get_similarity(100001, 100002) == 0
    assert pc.get_similarity(100001, 100002) < 1
    assert pc.get_similarity(100002, 100004) > 0
    assert pc.get_similarity(100002, 100001) == pc.get_similarity(100001, 100002)

    # Raise value error if patent id's are not in the dataset.
    with pytest.raises(ValueError):
        print(pc.get_similarity(0, 1))

    cor_dict = pc.sample_cpc_correlations(list(year_lookup), None)
    check_correlations(cor_dict, 15)

    cor_dict = pc.sample_cpc_correlations(list(year_lookup), samples_per_patent=2)
    check_correlations(cor_dict, 12)

    # This is a bit awkward, but we don't want everything to be run again.
    cor_dict = pc.sample_cpc_correlations(list(year_lookup), samples_per_patent=3)
    check_correlations(cor_dict, 18)

    cor_dict = pc.sample_cpc_correlations(list(year_lookup), samples_per_patent=10)
    check_correlations(cor_dict, 15)

    with pytest.raises(ValueError):
        pc.sample_cpc_correlations([100001])
