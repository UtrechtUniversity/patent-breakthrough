"""
Preprocessor for patent texts
"""
import argparse
import glob
import logging
import json
import re
import os
import lzma
from typing import List
from pathlib import Path


class Preprocessor:
    """
    Preprocessor class
    """

    def __init__(
            self,
            log_level: int = logging.INFO,
            log_file: str = None,
            log_format: str = '%(asctime)s [%(levelname)s] %(message)s',
            keep_empty_patents: bool = False,
            keep_missing_years: bool = False,
            keep_caps: bool = False,
            keep_start_section: bool = False,
            remove_non_alpha: bool = False,
            input_dir: str = None,
            output_dir: str = None):

        self.logger = logging.getLogger('preprocessor')
        self.logger.setLevel(log_level)
        slog = logging.StreamHandler()
        slog.setLevel(log_level)
        slog.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(slog)

        if log_file:
            flog = logging.FileHandler(log_file)
            flog.setLevel(log_level)
            flog.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(flog)

        self.keep_empty_patents = keep_empty_patents
        self.keep_missing_years = keep_missing_years
        self.keep_caps = keep_caps
        self.keep_start_section = keep_start_section
        self.remove_non_alpha = remove_non_alpha

        self.input_dir = input_dir
        self.output_dir = output_dir
        # if self.output_dir is not None:
            # os.makedirs(self.output_dir, exist_ok=True)
        # if 
        # self.logger.info(f'reading {input_dir}')
        # self.file_list = self.get_file_list(input_dir)

        #
        # self.logger = None
        # self.keep_empty_patents = False
        # self.keep_missing_years = False
        # self.keep_caps = False
        # self.keep_start_section = False
        # self.remove_non_alpha = False
        # self.output_dir = None
        # self.file_list = []
        # self.total_docs = {'processed': 0, 'empty': 0, 'no_year': 0}

    # def initialize(
    #         self,
    #         log_level: int = logging.INFO,
    #         log_file: str = None,
    #         log_format: str = '%(asctime)s [%(levelname)s] %(message)s',
    #         keep_empty_patents: bool = False,
    #         keep_missing_years: bool = False,
    #         keep_caps: bool = False,
    #         keep_start_section: bool = False,
    #         remove_non_alpha: bool = False,
    #         input_dir: str = None,
    #         output_dir: str = None,
    #         ):
    #     """Initializes all vars"""
    #
    #     self.logger = logging.getLogger('preprocessor')
    #     self.logger.setLevel(log_level)
    #     slog = logging.StreamHandler()
    #     slog.setLevel(log_level)
    #     slog.setFormatter(logging.Formatter(log_format))
    #     self.logger.addHandler(slog)
    #
    #     if log_file:
    #         flog = logging.FileHandler(log_file)
    #         flog.setLevel(log_level)
    #         flog.setFormatter(logging.Formatter(log_format))
    #         self.logger.addHandler(flog)
    #
    #     self.keep_empty_patents = keep_empty_patents
    #     self.keep_missing_years = keep_missing_years
    #     self.keep_caps = keep_caps
    #     self.keep_start_section = keep_start_section
    #     self.remove_non_alpha = remove_non_alpha
    #
    #     # self.input_dir = input_dir
    #     self.output_dir = output_dir
    #     os.makedirs(self.output_dir, exist_ok=True)
    #     self.logger.info(f'reading {input_dir}')
    #     self.file_list = self.get_file_list(input_dir)

    def from_arguments(self):
        """Parses command line parameters and initiates class"""
        parser = argparse.ArgumentParser(description="Cleans patents")

        # Path to the folder that contains the text files
        parser.add_argument(
            "--input",
            type=str,
            required=True,
            help="path to input plus extension e.g. './input/*.jsonl'"
        )
        # Path to output directory
        parser.add_argument(
            '--output',
            type=str,
            required=True,
            default=None,
            help='output directory string e.g. "../cleaned/"'
        )
        parser.add_argument('--remove_non_alpha', action='store_true')
        parser.add_argument('--keep_caps', action='store_true')
        parser.add_argument('--keep_start_section', action='store_true')
        parser.add_argument('--keep_empty_patents', action='store_true')
        parser.add_argument('--keep_missing_years', action='store_true')
        parser.add_argument('--log_file', type=str)
        args = vars(parser.parse_args())

        self.initialize(
            log_file=args['log_file'],
            keep_empty_patents=args['keep_empty_patents'],
            keep_missing_years=args['keep_missing_years'],
            keep_caps=args['keep_caps'],
            keep_start_section=args['keep_start_section'],
            remove_non_alpha=args['remove_non_alpha'],
            input_dir=args['input'],
            output_dir=args['output'],
        )

    @staticmethod
    def get_file_list(input_dir) -> list[str]:
        """Reads files from input directory"""
        return sorted(glob.glob(input_dir))

    def preprocess_files(self):
        """Iterates all input JSONL-files and calls preprocessing for each"""
        for file in self.file_list:
            self.logger.info(f'processing {file}')
            processed, empty, no_year = self.preprocess_file(file)
            self.logger.info(f'processed {file} ({processed:,} documents, ' +
                             f'skipped {empty:,} empty, {no_year:,} w/o year)')
            self.total_docs['processed'] += processed
            self.total_docs['empty'] += empty
            self.total_docs['no_year'] += no_year

        self.logger.info("done")
        self.logger.info(f"files: {len(self.file_list):,}")
        self.logger.info(f"docs processed: {self.total_docs['processed']:,}")
        self.logger.info(f"skipped empty docs: {self.total_docs['empty']:,}")
        self.logger.info("skipped docs w/o year: " +
                         f"{self.total_docs['no_year']:,}")

    # @staticmethod
    def yield_document(self, file: str):
        """Generator yielding single JSON-doc from input file"""
        if Path(file).suffix == ".json":
            return self.patent_get_jsonl(file)
        else:
            return self.patent_gen_xz(file)

    def patent_get_jsonl(self, file: str):
        with open(file) as handle:
            line = handle.readline()
            while line:
                yield json.loads(line)
                line = handle.readline()

    def patent_gen_xz(self, file: str):
        with lzma.open(file, mode="rb") as comp_fp:
            patents = json.loads(comp_fp.read().decode(encoding="utf-8"))
        for pat in patents:
            yield pat

    def preprocess_file(self, file: str):
        """Iterates individual JSON-docs in JSONL-file and calls preprocsseing
        for each"""
        parts = os.path.splitext(os.path.basename(file))
        processed = 0
        skipped_empty = 0
        skipped_no_year = 0
        # with open(new_file, "w") as new_file:
        processed_patents = []
        for patent in self.yield_document(file):
            if patent['year'] == 0 or patent['year'] is None:
                self.logger.warning(f'Patent #{patent["patent"]} has ' +
                                    'no year')
                if not self.keep_missing_years:
                    skipped_no_year += 1
                    continue

            body = patent['contents']

            if len(body) == 0:
                self.logger.warning(f'Patent #{patent["patent"]} has ' +
                                    'no content')
                if not self.keep_empty_patents:
                    skipped_empty += 1
                    continue

            body = self.remove_unprintable(body)
            body = self.remove_start_section(body)
            body = self.clean_document(body)
            body = self.remove_remains(body)

            patent['contents'] = body
            processed += 1
            processed_patents.append(patent)

        if self.output_dir is not None:
            new_file = os.path.join(self.output_dir, parts[0]+'_cleaned'+parts[1])
            self.write_document(new_file, processed_patents)

        stats = {
            "processed": processed,
            "skipped_empty": skipped_empty,
            "skipped_no_year": skipped_no_year,
        }

        return processed_patents, stats

    @staticmethod
    def remove_unprintable(content: str) -> str:
        """Removes unprintable characters"""
        return re.sub(r'[\u0080-\uffff]', "", content)

    @staticmethod
    def remove_remains(content: str) -> str:
        """Replaces multiple spaces with single ones"""
        cleaned = re.sub(r'(\B[^\w\s]+\B)', '', content)
        cleaned = re.sub(r'[ ]+', ' ', cleaned)
        return cleaned.strip()

    def clean_document(self, content: str) -> str:
        """Go through all words in document and call word processor."""
        word_pattern = re.compile(r'\w+')
        content_cleaned = word_pattern.sub(
            lambda word: self.process_word(word.group(0)),
            content
        )
        return content_cleaned

    def process_word(self, word: str) -> str:
        """Processes individual words"""
        lower_word = word.lower()
        contains_only_ident_chars = re.compile(r'\b([a-z])\1{1,}\b')
        contains_multiple_ident_chars = \
            re.compile(r'([a-z])\1{2,}')
        valid_single_letter_words = ['a', 'i']

        if (len(lower_word) < 2 and
                lower_word not in valid_single_letter_words):
            # print(f"x [single]: {word}")
            return ''

        if (self.remove_non_alpha and not lower_word.isalpha()):
            # print(f"x [non-alpha]: {word}")
            return ''

        if bool(contains_only_ident_chars.search(lower_word)):
            # print(f"x [ident]: {word}")
            return ''

        if bool(contains_multiple_ident_chars.search(lower_word)):
            # print(f"x [multi]: {word}")
            return ''

        # print(f"v: {word}")
        return word if self.keep_caps else lower_word

    @staticmethod
    def write_document(output_fp, all_patents: List[dict]):
        """Writes processed docs"""
        with open(output_fp, "w") as f:
            f.write("\n".join([json.dumps(patent) for patent in all_patents]))

    @staticmethod
    def count_upper_case_letters(str_obj: str) -> int:
        """Counts the number of uppercase letters in a string"""
        count = 0
        for elem in str_obj:
            if elem.isupper():
                count += 1
        return count

    @staticmethod
    def chunker(seq, size) -> list[str]:
        """Returns a chunk of a list of strings"""
        return list(seq[pos:pos + size] for pos in range(0, len(seq), size))

    def split_and_clean(self, content: str) -> List[str]:
        """Split content into words and clean them"""
        raw_words = content.split()
        words = [x for x in raw_words if len(self.process_word(x)) > 0]
        return words

    def remove_start_section(self, content: str,
                             frac_threshold: float = 0.60) -> str:
        """Removes the start section of a patent, identified by a large
        fraction of capital letters in the text"""
        if self.keep_start_section or len(content) == 0:
            return content

        words = self.split_and_clean(content)

        if len(words) == 0:
            return content

        first_words = self.get_first_words(words=words,
                                           frac_threshold=frac_threshold)
        start_section = self.get_start_section(first_words=first_words,
                                               frac_threshold=frac_threshold)

        return " ".join(words)[len(start_section):].strip()

    def get_first_words(self, words: list[str],
                        frac_threshold: float) -> list[str]:
        """Go through the first 1000 words, and break when the overall
        percentage of characters that are CAPS falls below the threshold"""
        tot_cap = 0
        tot_len = 0
        idx = 0
        for idx, word in enumerate(words[0:1000]):
            tot_cap += self.count_upper_case_letters(word)
            tot_len += len(word)
            if (idx > 10 and (tot_cap/tot_len) < frac_threshold):
                break

        return words[0:idx]

    def get_start_section(self, first_words: list, frac_threshold: float):
        """Go through the remaining words in chunks of three words at a time,
        and again calculate the percentage of CAPS-characters per chunk, and
        stop after it falls below the threshold. the first three chunks are
        always included (some patents start with a few lowercase words)"""
        chunks = self.chunker(first_words, 3)

        if len(chunks) == 0:
            return ""

        idx = 0
        chunk = []
        for idx, chunk in enumerate(chunks):
            c_upper = sum(map(self.count_upper_case_letters, chunk))
            c_all = sum(map(len, chunk))
            frac = (c_upper / c_all)
            if idx > 10 and frac < frac_threshold:
                break

        # join the chunks
        first_part = []
        for item in list(chunks)[:idx]:
            first_part += item

        # examine the last chunk, which fell below the threshold, but might
        # contain some uppercase words at the start
        rest = []
        for word in chunk:
            if self.count_upper_case_letters(word) / len(word) >= \
                                                     frac_threshold:
                rest.append(word)
            else:
                break

        start_section = first_part + rest
        for idx, item in enumerate(reversed(start_section)):
            if (self.count_upper_case_letters(item) / len(item)) > frac:
                break

        if idx > 0:
            start_section = start_section[0:len(start_section)-idx]

        # reassemble start section, and chop it from the original body
        start_section = " ".join(start_section)
        return start_section


if __name__ == '__main__':
    p = Preprocessor()
    p.from_arguments()
    p.preprocess_files()
