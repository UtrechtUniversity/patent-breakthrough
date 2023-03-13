import json
from pathlib import Path
from docembedder.preprocessor.preprocessor import Preprocessor
from pytest import mark
from docembedder.preprocessor.oldprep import OldPreprocessor

data_dir = Path("tests", "data")
lexicon_file = data_dir / "lexicon.txt"
doc_test = data_dir / "doc_test.json"
doc_test_cleaned = data_dir / "doc_test_cleaned.json"
doc_test_no_remains = data_dir / "doc_test_no_remains.json"
doc_test_no_start = data_dir / "doc_test_no_start.json"
doc_test_reassembled = data_dir / "doc_test_reassembled.json"

with open(doc_test, 'r') as file1:
    original_doc = json.load(file1)


def test_reassemble_words():
    with open(doc_test_reassembled, 'r') as file:
        reassembled = json.load(file)
    preprocessor = Preprocessor(lexicon_path=lexicon_file)
    assert original_doc['contents'] != reassembled['contents'] and \
        preprocessor.reassemble_words(original_doc['contents']) == \
        reassembled['contents']


def test_remove_start_section():
    with open(doc_test_no_start, 'r') as file:
        no_start_section = json.load(file)
    preprocessor = Preprocessor()
    assert original_doc['contents'] != no_start_section['contents'] and \
        preprocessor.remove_start_section(original_doc['contents']) == \
        no_start_section['contents']


def test_clean_document():
    with open(doc_test_cleaned, 'r') as file:
        cleaned = json.load(file)
    preprocessor = Preprocessor()
    assert original_doc['contents'] != cleaned['contents'] and \
        preprocessor.clean_document(original_doc['contents']) == \
        cleaned['contents']


def test_remove_unprintable():
    raw = '\u0080 A "test" \u0081\u00ddsentence\u00ff.\u00ff\n'
    clean = ' A "test" sentence.\n'
    preprocessor = Preprocessor()
    assert preprocessor.remove_unprintable(raw) == clean


def test_remove_remains():
    raw = 'A #$%^& test    sentence'
    clean = 'A test sentence'
    preprocessor = Preprocessor()
    assert preprocessor.remove_remains(raw) == clean


@mark.parametrize("patent_fp", [Path("tests", "data", "input", "test.jsonl"),
                                Path("tests", "data", "input", "1870.xz")])
@mark.parametrize("prep,n_args", [
    (Preprocessor(), 5),
    (OldPreprocessor(list_path=Path("tests", "data")), 3)])
def test_preprocess_file(patent_fp, prep, n_args):
    patents = prep.preprocess_file(patent_fp)
    assert isinstance(patents, list)
    assert len(patents) == 8

    patents = prep.preprocess_file(patent_fp, max_patents=5)
    assert len(patents) == 5

    patents, stats = prep.preprocess_file(patent_fp, return_stats=True)
    assert isinstance(stats, dict)
    assert len(patents) == 8

    assert len(prep.settings) == n_args
    assert isinstance(prep.hyper_space(), dict)


def test_preprocess_multiple():
    prep = Preprocessor(input_dir=Path("tests", "data", "input"))
    patents, stats = prep.preprocess_files()
    assert isinstance(patents, list)
    assert isinstance(patents[0], dict)
    assert len(patents) == 16
    assert isinstance(stats, dict)
