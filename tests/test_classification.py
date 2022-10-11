from pathlib import Path

import pytest

from docembedder.classification import PatentClassification


data_dir = Path("tests", "data")
class_fp = data_dir / "test_GPCPs.txt"
raw_fp = data_dir / "raw_test_combined.txt"

year_lookup = {100001: 1960, 100002: 1961, 100003: 1962,
               100004: 1960, 100005: 1961, 100006: 1962}


def test_classification():
    pc = PatentClassification(class_fp)
    assert pc.get_similarity(100001, 100001) == 1
    assert pc.get_similarity(100001, 100002) == 0
    assert pc.get_similarity(100001, 100002) < 1
    assert pc.get_similarity(100002, 100004) > 0
    assert pc.get_similarity(100002, 100001) == pc.get_similarity(100001, 100002)

    with pytest.raises(ValueError):
        print(pc.get_similarity(0, 1)) 
