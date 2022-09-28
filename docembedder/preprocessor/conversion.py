import re
from pathlib import Path
from collections import Counter, defaultdict
import lzma
import json
from typing import List, Union, Dict

import pandas as pd

# compile patterns for patent file format and whitepace
PATENT_FILE_PATTERN = re.compile(r'^\/Volumes\/(.*)?\d+\-\d+\/US\d+\.txt')
WHITESPACE = re.compile(r'\s+')


def parse_file_contents(contents: str):
    split_contents = contents.split('\n')
    patents: List[str] = []

    # this is for the current patent text
    current_patent = None
    current_file = None
    current_contents = ''

    # iterate over contents
    for line in split_contents:
        # check if current line is the start of a new patent
        head = PATENT_FILE_PATTERN.search(line)
        if head is not None:
            # add previous patent to data bucket
            if current_patent is not None:
                # transform patent name (remove US part) and verify with
                # filename
                patent_number = current_patent.upper()
                patent_verification = Path(current_file).stem
                if patent_number == patent_verification:
                    patent = int(patent_number.replace('US', ''))
                else:
                    raise f'Patent {patent_number} can\'t be verified'

                contents = current_contents or ''
                patents.append({
                    'patent': patent,
                    'file': current_file,
                    'contents': WHITESPACE.sub(' ', contents).strip()
                })
            # get the file
            current_file = head.group(0)
            # get the patent from the filepath
            current_patent = (current_file.split('/')[-1])[0:-4]
            # start contents
            current_contents = line.replace(current_file, '')
        else:
            # add line to contents
            current_contents += line

    return patents


def parse_raw(patent_input_fp: Union[Path, str], year_lookup: Counter):
    with open(patent_input_fp, mode='r', encoding='latin-1') as f:
        contents = f.read()

        # parse contents
        parsed = parse_file_contents(contents)
        # add year
        parsed = [
            {**patent, **{'year': year_lookup[patent['patent']]}}
            for patent in parsed
        ]
    return parsed


def read_xz(fp: Union[Path, str]):
    with lzma.open(fp, mode="rb") as comp_fp:
        patents = json.loads(comp_fp.read().decode(encoding="utf-8"))
    return patents


def write_xz(fp: Union[Path, str], patents: List[Dict]):
    with lzma.open(fp, mode="wb", preset=9) as comp_fp:
        comp_fp.write(str.encode(json.dumps(patents), encoding="utf-8"))


def compress_raw(patent_input_fp: Union[Path, str], year_fp: Union[Path, str],
                 output_dir: Union[Path, str]):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    year_df = pd.read_csv(year_fp, sep='\t')
    year_lookup = Counter(dict(zip(year_df["pat"], year_df["year"])))

    parsed_data = parse_raw(patent_input_fp, year_lookup)
    sorted_patents = sorted(parsed_data, key=lambda x: x["patent"])
    cat_patents = defaultdict(lambda: [])

    for pat in sorted_patents:
        cat_patents[pat["year"]].append(pat)

    for year, patents in cat_patents.items():
        compressed_fp = output_dir / Path(str(year) + ".xz")
        if compressed_fp.is_file():
            old_patents = read_xz(compressed_fp)
            old_patent_id = [pat["patent"] for pat in old_patents]
            added_patents = [pat for pat in patents if pat["patent"] not in old_patent_id]
            if len(added_patents) == 0:
                continue
            old_patents.extend(added_patents)
            patents = old_patents

        write_xz(compressed_fp, patents)


def compress_raw_dir(patent_input_dir: Union[Path, str], year_fp: Union[Path, str],
                     output_dir: Union[Path, str]):
    patent_input_dir = Path(patent_input_dir)
    processed_fp = Path(output_dir) / "processed_files.txt"
    try:
        with open(processed_fp, "r") as fp:
            processed = fp.read().split("\n")
    except FileNotFoundError:
        processed = []

    for patent_fp in patent_input_dir.glob("*.txt"):
        if patent_fp.name in processed:
            continue

        compress_raw(patent_fp, year_fp, output_dir)
        processed.append(patent_fp.name)

    with open(processed_fp, "w") as fp:
        fp.write("\n".join(processed))
