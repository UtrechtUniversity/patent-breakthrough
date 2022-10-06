"""
Reassembler of words split in two
"""
import json
import csv
import argparse
import os
import logging
from typing import List, Dict, Iterable, Tuple


class SplitWordReassembler:
    """
    Split Word Reassembler class

    """
    def __init__(self,
                 lexicon: str,
                 input_path: str,
                 output_dir: str,
                 contents_property: str = 'contents',
                 id_property: str = 'patent',
                 single_letter_words: List[str] = None,
                 min_assembly_length: int = 5,
                 save_replacements: bool = False,
                 log_level: int = logging.INFO,
                 log_file: str = None,
                 log_format: str = '%(asctime)s [%(levelname)s] %(message)s',
                 ):
        """Initializes vars"""

        self.min_assembly_length = min_assembly_length
        self.save_replacements = save_replacements
        self.contents_property = contents_property
        self.id_property = id_property

        if single_letter_words is None:
            self.single_letter_words = ['a', 'i']
        else:
            self.single_letter_words = single_letter_words

        # initialize logging
        self.logger = logging.getLogger('reassembler')
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

        self.debug = log_level == logging.DEBUG

        # input & output
        self.input_path = input_path
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        assert os.path.exists(self.output_dir), \
            f"{self.output_dir} doesn't exist and could not be created"
        self.logger.info("writing to '%s'", output_dir)

        # limites of length of assembled words
        assert isinstance(min_assembly_length, int), \
            "min_assembly_length must be integer"
        assert 2 < min_assembly_length < 100, \
            "min_assembly_length must be between 2 and 100"

        self.min_assembly_length = min_assembly_length
        self.logger.info("min_assembly_length set to %s", min_assembly_length)

        # processing database containing lexicon
        assert os.path.exists(lexicon), \
            f"lexicon file '{lexicon}' does not exist"
        lexicon_extension = os.path.splitext(lexicon)[1].lower()
        lexicon_extensions = ['.txt', '.csv']
        assert lexicon_extension in lexicon_extensions, \
            "lexicon should be one of: " + \
            f"[{', '.join(lexicon_extensions)}] (got {lexicon_extension})"

        # loading lexicon into dictionary
        self.logger.info("reading lexicon from '%s'", lexicon)
        with open(lexicon, encoding="utf-8") as file:
            dictionary = file.readlines()
        dictionary = [line.rstrip() for line in dictionary]
        self.dictionary = set(list(set(dictionary)))
        self.logger.info("loaded %s words", len(self.dictionary))

    @classmethod
    def from_arguments(cls):
        """Parses command line parameters and initiates class"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", "-i", type=str, required=True,
                            help="input folder or file (script reads json, \
                                 jsonl and csv)")
        parser.add_argument("--output_dir", "-o", type=str,
                            help="output directory")
        parser.add_argument("--lexicon", "-l", type=str,
                            help="csv or text file with lexicon")
        parser.add_argument("--min_assembly_length", "-m", type=int,
                            help="minimum length for reassembled words \
                                 (min. 3, default 5)")
        parser.add_argument("--save_replacements", "-s", action="store_true",
                            help="save replacement details (to seperate file)")
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()

        return cls(
            lexicon=args.lexicon,
            log_level=logging.DEBUG if args.debug else logging.INFO,
            input_path=args.input_path,
            output_dir=args.output_dir,
            min_assembly_length=args.min_assembly_length,
            save_replacements=args.save_replacements)

    @property
    def file_list(self) -> List[str]:
        """Reads files from input directory"""
        valid_extensions = ['.json', '.jsonl']
        file_list = []
        if os.path.isfile(self.input_path):
            extension = os.path.splitext(self.input_path)[1].lower()
            if extension in valid_extensions:
                file_list.append(self.input_path)
        else:
            for file in os.listdir(self.input_path):
                extension = os.path.splitext(file)[1].lower()
                if extension in valid_extensions:
                    file_list.append(os.path.join(self.input_path, file))
        return file_list

    @staticmethod
    def yield_document(file: str, doc_format: str) -> Iterable[List]:
        """Generator yielding single JSON-doc from JSONL-input file"""
        with open(file, encoding="utf-8") as handle:
            line = handle.readline()
            while line:
                if doc_format == "json":
                    yield json.loads(line)
                elif doc_format == "csv":
                    yield list(csv.reader([line]))[0]
                line = handle.readline()

    def get_file_names(self, file: str) -> Tuple[str, str, str]:
        """Determine output filenames"""
        parts = os.path.splitext(os.path.basename(file))
        extension = parts[1].lower()
        new_file = os.path.join(self.output_dir,
                                f"{parts[0]}_reassembled{parts[1]}")
        replacements_file = os.path.join(self.output_dir,
                                         f"{parts[0]}_replacements{parts[1]}")
        return extension, new_file, replacements_file

    def run_reassembly(self):
        """Iterates through file list, opens document(s),
        calls main processor, keeps score"""

        tot_docs = 0
        tot_replacements = 0

        for file in self.file_list:
            extension, new_file, replacements_file = self.get_file_names(file)
            with open(new_file, "w", encoding="utf-8") as new_file:
                self.logger.info("processing '%s'", file)
                new_docs = []
                replacements = []

                if extension == ".jsonl":
                    for doc in self.yield_document(file, 'json'):
                        new_doc, replaced = self.process_doc(doc=doc)
                        new_docs.append(new_doc)
                        replacements.extend(replaced)

                elif extension == ".json":
                    with open(file, "r", encoding="utf-8") as handle:
                        doc = json.load(handle)
                        new_doc, replacements = self.process_doc(doc=doc)
                        new_docs.append(new_doc)
                        replacements.extend(replaced)

                elif extension == ".csv":
                    csv_header = None
                    for doc in self.yield_document(file, 'csv'):
                        if not csv_header:
                            csv_header = doc
                            self.write_document(file_handle=new_file,
                                                docs=[dict(zip(csv_header,
                                                      csv_header))],
                                                doc_format="csv")
                        else:
                            doc = dict(zip(csv_header, doc))
                            new_doc, replacements = self.process_doc(doc=doc)
                            new_docs.append(new_doc)
                            replacements.extend(replaced)

                self.write_document(file_handle=new_file, docs=new_docs,
                                    doc_format="csv" if extension == ".csv"
                                    else "json")

                if self.save_replacements:
                    with open(replacements_file, "w", encoding="utf-8") \
                         as replacements_file:
                        if extension == ".csv":
                            self.write_document(file_handle=replacements_file,
                                                docs=replacements,
                                                doc_format="csv")
                        else:
                            self.write_document(file_handle=replacements_file,
                                                docs=replacements)

                tot_docs += len(new_docs)
                replacement_count = sum(len(x['replacements']) for x
                                        in replacements)

                tot_replacements += replacement_count

                self.logger.info("%s replacements in %s documents (%s avg)",
                                 replacement_count, len(new_docs),
                                 round(replacement_count/len(new_docs), 2))

        self.logger.info("total: %s replacements in %s documents (%s avg)",
                         tot_replacements, tot_docs,
                         round(tot_replacements/tot_docs, 2))

    @staticmethod
    def write_document(file_handle, docs: List[Dict],
                       doc_format: str = 'json'):
        """Writes processed docs"""
        for doc in docs:
            if doc_format == 'json':
                file_handle.write(json.dumps(doc) + "\n")
            elif doc_format == 'csv':
                row = []
                for key in doc:
                    row.append(doc[key])
                writer = csv.writer(file_handle)
                writer.writerow(row)

    def process_doc(self, doc: Dict) -> Tuple[Dict, List[Dict]]:
        """Tokenize and call reassembler"""
        word_list = doc[self.contents_property].split()
        new_word_list, replaced_tokens = self.reassemble(word_list)
        doc[self.contents_property] = " ".join(new_word_list)
        replacements = []
        if self.save_replacements:
            if self.id_property is not None:
                replacements.append({"id": doc[self.id_property],
                                    "replacements": replaced_tokens})
            else:
                replacements.append({"replacements": replaced_tokens})

        return doc, replacements

    def reassemble(self, word_list: List[str]) -> Tuple[List[str], List[Dict]]:
        """Walks through tokens and attempts to reassemble split words"""
        new_word_list = []
        replaced_tokens = []
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
               and next_token not in self.single_letter_words):
                # if it's not a word in itself, combine with current token
                assembly = token+next_token
                if len(assembly) >= self.min_assembly_length \
                   and assembly in self.dictionary:
                    # we reassembled a word!
                    new_word_list.append(assembly)
                    replaced_tokens.append({'tokens': [token, next_token],
                                           'replacement': assembly})
                    # make sure to skip over the next token, which is now
                    # part of the newly reassembled word
                    skip = True
                else:
                    new_word_list.append(token)
            else:
                new_word_list.append(token)

        return new_word_list, replaced_tokens


if __name__ == "__main__":
    swr = SplitWordReassembler.from_arguments()
    swr.run_reassembly()
