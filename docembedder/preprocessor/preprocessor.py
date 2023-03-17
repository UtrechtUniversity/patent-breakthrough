"""
Preprocessor for patent texts
"""
import argparse
import json
import re
from typing import List, Dict, Iterable, Tuple, Set, Optional, Union, overload, Any
from pathlib import Path

from typing_extensions import Literal

from hyperopt import hp  # type: ignore

from docembedder.preprocessor.parser import read_xz
from docembedder.typing import PathType


class Preprocessor:  # pylint: disable=too-many-instance-attributes too-many-public-methods
    """
    Preprocessor class
    """

    def __init__(  # pylint: disable=too-many-arguments too-many-locals
            self,
            keep_empty_patents: bool = False,
            keep_missing_years: bool = False,
            keep_caps: bool = False,
            keep_start_section: bool = False,
            remove_non_alpha: bool = False,
            input_dir: Optional[PathType] = None,
            output_dir: Optional[str] = None,
            lexicon_path: Optional[str] = None
            ):

        self.keep_empty_patents = keep_empty_patents
        self.keep_missing_years = keep_missing_years
        self.keep_caps = keep_caps
        self.keep_start_section = keep_start_section
        self.remove_non_alpha = remove_non_alpha

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.total_docs = {'processed': 0, 'skipped_empty': 0,
                           'skipped_no_year': 0, 'words_reassembled': 0}

        self.valid_single_letter_words = ['a', 'i']
        self.min_assembly_length = 5
        self.dictionary = self.read_dictionary(lexicon_path)

    @classmethod
    def from_arguments(cls):
        """Parses command line parameters and initiates class"""
        parser = argparse.ArgumentParser(description="Cleans patents")

        # Path to the folder that contains the text files
        parser.add_argument(
            "--input",
            type=str,
            default=None,
            help="path to input plus extension e.g. './input/*.jsonl'"
        )
        # Path to output directory
        parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='output directory string e.g. "../cleaned/"'
        )
        parser.add_argument(
            '--lexicon',
            type=str,
            default=None,
            help='csv or text file with lexicon'
        )
        parser.add_argument('--remove_non_alpha', action='store_true')
        parser.add_argument('--keep_caps', action='store_true')
        parser.add_argument('--keep_start_section', action='store_true')
        parser.add_argument('--keep_empty_patents', action='store_true')
        parser.add_argument('--keep_missing_years', action='store_true')
        args = vars(parser.parse_args())

        return cls(
            keep_empty_patents=args['keep_empty_patents'],
            keep_missing_years=args['keep_missing_years'],
            keep_caps=args['keep_caps'],
            keep_start_section=args['keep_start_section'],
            remove_non_alpha=args['remove_non_alpha'],
            input_dir=args['input'],
            output_dir=args['output'],
            lexicon_path=args['lexicon']
        )

    @property
    def file_list(self) -> List[Path]:
        """Reads files from input directory"""
        if self.input_dir is None:
            return []
        input_dir = Path(self.input_dir)
        return list(input_dir.glob("*.jsonl")) + list(input_dir.glob("*.xz"))

    @staticmethod
    def read_dictionary(lexicon_path) -> Set[str]:
        """Reads words from dictionary file"""
        if lexicon_path is None:
            return set([])

        path = Path(lexicon_path)
        assert path.is_file(), \
            f"lexicon file '{lexicon_path}' does not exist"
        lexicon_extension = path.suffix.lower()
        lexicon_extensions = ['.txt', '.csv']
        assert lexicon_extension in lexicon_extensions, \
            "lexicon should be one of: " + \
            f"[{', '.join(lexicon_extensions)}] (got {lexicon_extension})"

        with open(lexicon_path, encoding="utf-8") as file:
            dictionary = file.readlines()

        dictionary = [line.strip() for line in dictionary]
        return set(dictionary)

    def preprocess_files(self) -> Tuple[List[Dict], Dict[str, int]]:
        """Iterates all input JSONL-files and calls preprocessing for each"""
        all_patents = []
        for file in self.file_list:
            processed_patents, stats = self.preprocess_file(  # type: ignore # pylint: disable=unpacking-non-sequence
                file, return_stats=True)

            all_patents.extend(processed_patents)
            self.total_docs['processed'] += len(processed_patents)
            self.total_docs['skipped_empty'] += stats["skipped_empty"]
            self.total_docs['skipped_no_year'] += stats["skipped_no_year"]
        return all_patents, self.total_docs

    def yield_document(self, file: PathType) -> Iterable[Dict]:
        """Generator yielding single JSON-doc from input file"""
        suffix = Path(file).suffix
        if suffix == ".jsonl":
            return self.patent_get_jsonl(file)
        if suffix == ".xz":
            return self.patent_get_xz(file)
        raise ValueError(f"Unsupported format for document '{file}': '{suffix}'")

    def patent_get_jsonl(self, file: PathType) -> Iterable[Dict]:
        """Generate patents from a JSONL file"""
        with open(file, encoding="utf-8") as handle:
            line = handle.readline()
            while line:
                yield json.loads(line)
                line = handle.readline()

    def patent_get_xz(self, file: PathType) -> Iterable[Dict]:
        """Generate patents from a compressed xz file"""
        for pat in read_xz(file):
            yield pat

    @overload
    def preprocess_file(self, file: PathType, max_patents: Optional[int],
                        return_stats: Literal[False] = ...) -> List[Dict]: ...

    @overload
    def preprocess_file(self, file: PathType, max_patents: Optional[int],
                        return_stats: Literal[True]) -> Tuple[List[Dict], Dict[str, int]]: ...

    def preprocess_file(self, file: PathType,
                        max_patents: Optional[int]=None,
                        return_stats: bool=False) -> Union[
                            List[Dict], Tuple[List[Dict], Dict[str, int]]]:
        """Iterates individual JSON-docs in JSONL-file and calls preprocsseing
        for each"""
        # print("current level", self.logger.level)
        # self.logger.setLevel(logging.ERROR)

        processed = 0
        skipped_empty = 0
        skipped_no_year = 0
        processed_patents: List[Dict] = []

        for patent in self.yield_document(file):
            if max_patents is not None and len(processed_patents) >= max_patents:
                break

            if patent['year'] == 0 or patent['year'] is None:
                # self.logger.warning('Patent #%s has no year',
                                    # str(patent["patent"]))
                if not self.keep_missing_years:
                    skipped_no_year += 1
                    continue

            body = patent['contents']

            if len(body) == 0:
                # self.logger.warning('Patent #%s has no content',
                                    # str(patent["patent"]))
                if not self.keep_empty_patents:
                    skipped_empty += 1
                    continue

            body = self.remove_unprintable(body)
            body = self.reassemble_words(body)
            body = self.remove_start_section(body)
            body = self.clean_document(body)
            body = self.remove_remains(body)

            patent['contents'] = body
            processed += 1
            processed_patents.append(patent)

        if self.output_dir is not None:
            path = Path(file)
            new_file = Path(self.output_dir,
                            path.stem + '_cleaned' + path.suffix)
            self.write_document(new_file, processed_patents)

        stats = {
            "processed": processed,
            "skipped_empty": skipped_empty,
            "skipped_no_year": skipped_no_year,
        }
        if return_stats:
            return processed_patents, stats
        return processed_patents

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

        if (len(lower_word) < 2 and
                lower_word not in self.valid_single_letter_words):
            return ''

        if (self.remove_non_alpha and not lower_word.isalpha()):
            return ''

        if bool(contains_only_ident_chars.search(lower_word)):
            return ''

        if bool(contains_multiple_ident_chars.search(lower_word)):
            return ''

        return word if self.keep_caps else lower_word

    @staticmethod
    def write_document(output_fp, all_patents: List[dict]):
        """Writes processed docs"""
        with open(output_fp, "w", encoding="utf-8") as handle:
            handle.write("\n".join([json.dumps(patent) for patent
                                    in all_patents]))

    @staticmethod
    def count_upper_case_letters(str_obj: str) -> int:
        """Counts the number of uppercase letters in a string"""
        count = 0
        for elem in str_obj:
            if elem.isupper():
                count += 1
        return count

    @staticmethod
    def chunker(seq, size) -> List[str]:
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

    def get_first_words(self, words: List[str],
                        frac_threshold: float) -> List[str]:
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

    def get_start_section(self, first_words: List[str], frac_threshold: float):
        """Go through the remaining words in chunks of three words at a time,
        and again calculate the percentage of CAPS-characters per chunk, and
        stop after it falls below the threshold. the first three chunks are
        always included (some patents start with a few lowercase words)"""
        chunks = self.chunker(first_words, 3)

        if len(chunks) == 0:
            return ""

        idx = 0
        chunk = ""
        for idx, chunk in enumerate(chunks):
            c_upper = sum(map(self.count_upper_case_letters, chunk))
            c_all = sum(map(len, chunk))
            frac = c_upper / c_all
            if idx > 10 and frac < frac_threshold:
                break

        # join the chunks
        first_part: List[str] = []
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
        joined_start_section = " ".join(start_section)
        return joined_start_section

    def reassemble_words(self, body: str) -> str:
        """Tokenize, walk through tokens and attempts to reassemble split
        words"""
        if len(self.dictionary) == 0:
            return body

        word_list = body.split()
        new_word_list = []
        skip = False

        for key, token in enumerate(word_list):
            if skip:
                skip = False
                continue

            if token in self.dictionary or not token.isalpha():
                # word exist as it is or is not just letters
                new_word_list.append(token)
                continue

            if key >= len(word_list)-1:
                # last in list, no next token to attempt reassembly
                new_word_list.append(token)
                continue

            # get the next token
            next_token = word_list[key+1]

            if next_token not in self.dictionary or (len(next_token) == 1
               and next_token not in self.valid_single_letter_words):
                # if it's not a word in itself, combine with current token
                assembly = token+next_token
                if len(assembly) >= self.min_assembly_length \
                   and assembly in self.dictionary:
                    # we reassembled a word!
                    new_word_list.append(assembly)
                    # self.logger.debug("%s + %s -> %s", token, next_token,
                    # assembly)

                    self.total_docs['words_reassembled'] += 1

                    # make sure to skip over the next token, which is now
                    # part of the newly reassembled word
                    skip = True
                else:
                    new_word_list.append(token)
            else:
                new_word_list.append(token)

        return " ".join(new_word_list)

    @property
    def settings(self):
        """Settings of the preprocessor."""
        return {
            "keep_empty_patents": self.keep_empty_patents,
            "keep_missing_years": self.keep_missing_years,
            "keep_caps": self.keep_caps,
            "keep_start_section": self.keep_start_section,
            "remove_non_alpha": self.remove_non_alpha,
        }

    @classmethod
    def hyper_space(cls) -> Dict[str, Any]:
        """Parameter space for hyperopt."""
        return {
            "keep_caps": hp.choice("keep_caps", [True, False]),
            "keep_start_section": hp.choice("keep_start_section", [True, False]),
            "remove_non_alpha": hp.choice("remove_non_alpha", [True, False]),
        }
