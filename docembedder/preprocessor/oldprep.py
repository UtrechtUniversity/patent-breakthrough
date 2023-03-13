from __future__ import annotations
from itertools import chain
import re
from typing import Optional, Union, overload, Literal

from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.typing import PathType


class OldPreprocessor(Preprocessor):
    def __init__(self, list_path=Path("..", "data"), keep_missing_years=False,
                 keep_empty_patents=False, job_id: Optional[str] = None):
        self.list_path = str(list_path)
        self.keep_missing_years = keep_missing_years
        self.keep_empty_patents = keep_empty_patents
        self.import_lists()

    @overload
    def preprocess_file(self, file: PathType, max_patents: Optional[int],
                        return_stats: Literal[False] = ...) -> list[dict]: ...

    @overload
    def preprocess_file(self, file: PathType, max_patents: Optional[int],
                        return_stats: Literal[True]) -> tuple[list[dict], dict[str, int]]: ...

    def preprocess_file(self, file: PathType,
                        max_patents: Optional[int]=None,
                        return_stats: bool=False) -> Union[
                        list[dict], tuple[list[dict], dict[str, int]]]:
        processed = 0
        skipped_empty = 0
        skipped_no_year = 0
        processed_patents: list[dict] = []

        for patent in self.yield_document(file):
            if max_patents is not None and len(processed_patents) >= max_patents:
                break
            if patent['year'] == 0 or patent['year'] is None:
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

            body = self.stem_clean_patent(body)
            # body = self.remove_unprintable(body)
            # body = self.reassemble_words(body)
            # body = self.remove_start_section(body)
            # body = self.clean_document(body)
            # body = self.remove_remains(body)

            patent['contents'] = body
            processed += 1
            processed_patents.append(patent)
        if return_stats:
            return processed_patents, {}
        return processed_patents

    def import_lists(self):
        # compile a set of words that need deleting
        list_path = Path(self.list_path)
        nltk.download('stopwords', quiet=True)
        with open(list_path / 'stopwords.txt', mode='r') as f:
            self.STOPWORDS = set(
                f.read().split() +
                ['.sub.', '.sup.'] +
                stopwords.words('english')
            )
        # print(f'...read {len(self.STOPWORDS)} stopwords...')

        # these symbols are to be replaced with a single space
        with open(list_path / 'symbols.txt', mode='r') as f:
            symbols = [s.strip().split(',') for s in f.read().split()]
            self.SYMBOLS = set(chain(*symbols))

        # and these Greek words/characters are to be replaced with standards
        greek = pd.read_csv(list_path / 'greek.txt')
        pairs = [[c, 'Version 3'] for c in ['letter', 'Version 1', 'Version 2']]
        subs = [greek[p].to_records(index=False) for p in pairs]
        GREEK = dict([t for t in chain(*subs) if t[0] != '-'])
        self.GREEK_KEYS = set(GREEK.keys())

        # and we need a stemmer
        self.STEMMER = SnowballStemmer('english')

        # this is to break up word patterns
        self.WORD_PATTERN = re.compile(r'\w+')
        # regex for consecutive identical characters
        self.CONTAINS_IDENT_CHARS = re.compile(r'(\w)\1{2,}')

    def stem_clean_patent(self, body):
        # if not isinstance(patent, dict):
            # return [self.stem_clean_patent(p) for p in patent]

        def process_word(matchobj):
            """Takes token and checks if it should be removed from the
            patent string. If not, the token is stemmed."""
            # convert word to lowkeys
            word = matchobj.group(0).lower()

            if word in self.GREEK_KEYS:
                # standardize Greek if possible
                word = self.GREEK.get(word, word)
            elif len(word) < 2 \
                or not(word.isalpha()) \
                or re.fullmatch(r'[mdcxvi]+[a-z]', word) \
                or bool(self.CONTAINS_IDENT_CHARS.search(word)) \
                or word in self.STOPWORDS \
                or word in self.SYMBOLS:
                # word must contain characters only, not a Roman number,
                # doesn't contain 3 or more identical characters, is not a
                # stopword and is not a symbol. The order matters here,
                # we would like to filter out bad tokens before checking them
                # against all stopwords
                return ''

            # stem
            token = self.STEMMER.stem(word)
            return '' if len(token) < 2 else token

        def process_contents(sentence):
            sentence = '' if type(sentence) is not str else sentence
            # and start processing the words
            return self.WORD_PATTERN.sub(
                    lambda word: process_word(word),
                    sentence
            )

        return process_contents(body)

    @property
    def settings(self):
        return {
            "keep_missing_years": self.keep_missing_years,
            "keep_empty_patents": self.keep_empty_patents,
            "list_path": str(self.list_path),
        }

    @classmethod
    def hyper_space(cls):
        return {}
