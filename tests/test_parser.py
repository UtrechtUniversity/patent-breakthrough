from pathlib import Path

from docembedder.preprocessor.parser import parse_raw, write_xz, read_xz, compress_raw,\
    compress_raw_dir

data_dir = Path("tests", "data")
raw_fp = data_dir / "raw_test.txt"
raw_fp_2 = data_dir / "raw_test_2.txt"
raw_fp_combined = data_dir / "raw_test_combined.txt"

year_lookup = {100001: 1960, 100002: 1961, 100003: 1962,
               100004: 1960, 100005: 1961, 100006: 1962}


def test_parser():
    patents = parse_raw(raw_fp, year_lookup)
    assert len(patents) == 3
    assert isinstance(patents, list)
    for pat in patents:
        assert isinstance(pat, dict)
        pat_id = pat["patent"]
        assert pat["year"] == year_lookup[pat_id]
        assert len(pat["contents"]) > 100
        assert pat["file"].endswith(f"US{pat_id}.txt")


def test_xz_io(tmp_path):
    tmp_file = tmp_path / "test.xz"
    patents = parse_raw(raw_fp, year_lookup)
    write_xz(tmp_file, patents)
    new_patents = read_xz(tmp_file)
    assert patents == new_patents


def test_compress_raw(tmp_path):
    compress_raw(raw_fp, year_lookup, tmp_path)
    patents = parse_raw(raw_fp, year_lookup)
    for year in year_lookup.values():
        year_fp = tmp_path / f"{year}.xz"
        assert year_fp.is_file()
        pat = read_xz(year_fp)[0]
        assert pat in patents
        assert pat["year"] == year


def test_compress_dir(tmp_path):
    compress_raw_dir(data_dir, year_lookup, tmp_path)
    created_files = list(Path(tmp_path).glob("*.xz"))
    assert len(created_files) == 3
    for xz_file in created_files:
        patents = read_xz(xz_file)
        assert len(patents) == 2


def test_combined():
    combined_patents = parse_raw(raw_fp_combined, year_lookup)
    patents_1 = parse_raw(raw_fp, year_lookup)
    patents_2 = parse_raw(raw_fp_2, year_lookup)

    for pat in combined_patents:
        assert pat in patents_1 or pat in patents_2
