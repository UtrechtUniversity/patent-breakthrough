"""Functions to read and convert different patent files"""

import json
import lzma
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Union, Dict

import pandas as pd

# compile patterns for patent file format and whitepace
PATENT_FILE_PATTERN = re.compile(r'^\/Volumes\/(.*)?\d+\-\d+\/US\d+\.txt')
WHITESPACE = re.compile(r'\s+')


def _parse_file_contents(contents: str) -> List[Dict]:
    """Parse the contents of the raw file in JSONL format

    Arguments
    ---------
    contents:
        String containing a set of patents.

    Returns
    -------
    patents: List[Dict]
        A list of dictionaries, where each item is one patent.

    """
    split_contents = contents.split('\n')
    patents: List[Dict] = []

    # this is for the current patent text
    current_patent = None
    current_file = None
    current_contents = ''

    def swap_patents():
        """Swap out patents, storing them."""
        if current_patent is None:
            return
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

    # iterate over contents
    for line in split_contents:
        # check if current line is the start of a new patent
        head = PATENT_FILE_PATTERN.search(line)
        if head is not None:
            # add previous patent to data bucket
            swap_patents()
            # get the file
            current_file = head.group(0)
            # get the patent from the filepath
            current_patent = (current_file.split('/')[-1])[0:-4]
            # start contents
            current_contents = line.replace(current_file, '')
        else:
            # add line to contents
            current_contents += line

    # Add the last patent if it has content.
    if len(current_contents) > 0:
        swap_patents()

    return patents


def parse_raw(patent_input_fp: Union[Path, str], year_lookup: Union[Counter, Dict]) -> List[Dict]:
    """Parse a raw patent file into a structured list

    Arguments
    ---------
    patent_input_fp:
        Input file to process.
    year_lookup:
        Dictionary to lookup the year for each patent ID.

    Returns
    -------
    patents: List[Dict]
        A list of dictionaries, where each item is for one patent and includes
        the year of publication.
    """
    with open(patent_input_fp, mode='r', encoding='latin-1') as handle:
        contents = handle.read()

    # parse contents
    parsed = _parse_file_contents(contents)
    # add year
    parsed = [
        {**patent, **{'year': year_lookup[patent['patent']]}}
        for patent in parsed
    ]
    return parsed


def read_xz(compressed_fp: Union[Path, str]) -> List[Dict]:
    """Read an .xz file containing patents

    Arguments
    ---------
    compressed_fp:
        File to read the patents from.

    Results
    -------
    patents: List[Dict]
        Patents in the file.
    """
    with lzma.open(compressed_fp, mode="rb") as handle:
        patents = json.loads(handle.read().decode(encoding="utf-8"))
    return patents


def write_xz(compressed_fp: Union[Path, str], patents: List[Dict]) -> None:
    """Write a set of patents to a compressed file

    Arguments
    ---------
    fp:
        File to write to.
    patents:
        Patents to store.
    """
    with lzma.open(compressed_fp, mode="wb", preset=9) as handle:
        handle.write(str.encode(json.dumps(patents), encoding="utf-8"))


def compress_raw(patent_input_fp: Union[Path, str], year_fp: Union[Path, str, Dict],
                 output_dir: Union[Path, str]) -> None:
    """Compress a raw file into multiple compressed files by year

    Arguments
    ---------
    patent_input_fp:
        Raw file with patents.
    year_fp:
        CSV file with publication year for each patent.
    output_dir:
        Directory to write the compressed files to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create the year lookup so that we can add the year to the patents.
    if isinstance(year_fp, dict):
        year_lookup = year_fp
    else:
        year_df = pd.read_csv(year_fp, sep='\t')
        year_lookup = Counter(dict(zip(year_df["pat"], year_df["year"])))

    # Read the patent data from the raw files and sort them by patent id.
    parsed_data = parse_raw(patent_input_fp, year_lookup)
    sorted_patents = sorted(parsed_data, key=lambda x: x["patent"])

    # Split the patents by year.
    cat_patents = defaultdict(lambda: [])
    for pat in sorted_patents:
        cat_patents[pat["year"]].append(pat)

    # Write the patents to a files, numbered by year.
    for year, patents in cat_patents.items():
        compressed_fp = output_dir / Path(str(year) + ".xz")

        # If the patent year already exists read the existing patents.
        if compressed_fp.is_file():
            old_patents = read_xz(compressed_fp)
            old_patent_id = [pat["patent"] for pat in old_patents]

            # Prevent adding duplicates.
            added_patents = [pat for pat in patents if pat["patent"] not in old_patent_id]
            if len(added_patents) == 0:
                continue
            old_patents.extend(added_patents)
            patents = old_patents

        write_xz(compressed_fp, patents)


def compress_raw_dir(patent_input_dir: Union[Path, str], year_fp: Union[Path, str, dict],
                     output_dir: Union[Path, str]) -> None:
    """Compress all raw files in a directory.

    For efficiency, it stores which files have already been processed in
    a file called 'processed_files.txt' in the output directory.
    If somehow there is corruption, or re-runs are required, simply
    delete this file.

    This function is not thread-safe.

    Arguments
    ---------
    patent_input_dir:
        Directory containing all raw files with patents.
    year_fp:
        CSV file with publication year for each patent.
    output_dir:
        Directory to write the compressed files to.
    """
    patent_input_dir = Path(patent_input_dir)

    # Keep track of all processed files so that we can skip them next time
    processed_fp = Path(output_dir) / "processed_files.txt"
    try:
        with open(processed_fp, "r", encoding="utf-8") as handle:
            processed = handle.read().split("\n")
    except FileNotFoundError:
        processed = []

    # Compress all files in the patent_input_dir.
    for patent_fp in patent_input_dir.glob("*.txt"):
        if patent_fp.name in processed:
            continue

        compress_raw(patent_fp, year_fp, output_dir)
        processed.append(patent_fp.name)

    # Save the processed file.
    with open(processed_fp, "w", encoding="utf-8") as handle:
        handle.write("\n".join(processed))
