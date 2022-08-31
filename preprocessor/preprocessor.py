import argparse, glob, logging, json, re, os

class Preprocessor:

    ARGS = None
    file_list = []
    current_file = None
    current_document = None
    new_file = None
    doc_property_content = 'contents'
    doc_property_patent_number = 'patent'
    valid_single_letter_words = ['a','i']
    remove_non_alpha = False
    keep_caps = False
    keep_start_section = False

    def __init__(self,log_level=logging.INFO,log_file=None):
        self.logger = logging.getLogger('preprocessor')
        self.logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(ch)
        if log_file:
            self.set_log_file(log_file,log_level)

    def set_log_file(self,log_file,log_level=logging.INFO):
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)

    def parse_arguments(self):
        """Parses command line parameters"""
        parser = argparse.ArgumentParser(
            description="Cleans patents"
        )
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
        parser.add_argument('--log_file',type=str)

        self.ARGS = vars(parser.parse_args())

        self.set_remove_non_alpha(self.ARGS['remove_non_alpha'])
        self.set_keep_caps(self.ARGS['keep_caps'])
        self.set_keep_start_section(self.ARGS['keep_start_section'])

        os.makedirs(self.ARGS['output'], exist_ok=True)

        if self.ARGS['log_file']:
            self.set_log_file(self.ARGS['log_file'])

    def get_args(self):
        return self.ARGS

    def set_doc_property_content(self,doc_property_content):
        self.doc_property_content = doc_property_content

    def set_doc_property_patent_number(self,doc_property_patent_number):
        self.doc_property_patent_number = doc_property_patent_number

    def set_remove_non_alpha(self,remove_non_alpha):
        self.remove_non_alpha = remove_non_alpha

    def set_keep_caps(self,keep_caps):
        self.keep_caps = keep_caps

    def set_keep_start_section(self,keep_start_section):
        self.keep_start_section = keep_start_section

    def read_folder(self):
        self.file_list = glob.glob(self.ARGS['input'])

    def print_settings(self):
        self.logger.info(f"input: {self.ARGS['input']}")
        self.logger.info(f"output: {self.ARGS['output']}")
        self.logger.info(f"remove_non_alpha: {self.remove_non_alpha}")
        self.logger.info(f"keep_caps: {self.keep_caps}")

    def process_files(self):
        for file in self.file_list:
            self.logger.info(f'processing {file}')
            self.current_file = file
            t = os.path.splitext(os.path.basename(self.current_file))
            self.new_file = os.path.join(self.ARGS['output'],t[0]+'_cleaned'+t[1])
            self.process_file()

    def process_file(self):
        f = open(self.current_file, "r")
        self.f_new = open(self.new_file, "w")

        while True:
            line = f.readline()
            if not line:
                break
            self.current_document = json.loads(line)
            self.remove_unprintable()
            self.remove_start_section()
            self.clean_document()
            self.remove_remains()
            self.write_document()

        f.close()
        self.f_new.close()

    def remove_unprintable(self):
        content = self.current_document[self.doc_property_content]
        # remove unprintable characters
        cleaned = re.sub(r'[\u0080-\uffff]', "", content)
        self.current_document[self.doc_property_content] = cleaned

    def remove_remains(self):
        content = self.current_document[self.doc_property_content]
        # remove non-alphanumeric substrings in words
        cleaned = re.sub(r'(\B[^\w\s]+\B)', '', content)
        # replace multiple spaces with single ones
        cleaned = re.sub(r'[ ]+',' ',cleaned)
        self.current_document[self.doc_property_content] = cleaned.strip()

    def clean_document(self):
        content = self.current_document[self.doc_property_content]

        WORD_PATTERN = re.compile(r'\w+')
        content_cleaned = WORD_PATTERN.sub(
            lambda word: self.process_word(word.group(0)),
            content
        )

        self.current_document[self.doc_property_content] = content_cleaned

    def process_word(self,word):
        lower_word = word.lower()

        # regex for consecutive identical characters
        # CONTAINS_IDENT_CHARS = re.compile(r'(\w)\1{2,}')
        CONTAINS_ONLY_IDENT_CHARS = re.compile(r'\b(([a-z])\2+)\b')
        CONTAINS_MULTIPLE_IDENT_CHARS = re.compile(r'\b([a-z]+)(([a-z])\3{3,})([a-z]+)\b')

        if (len(lower_word) < 2 and not lower_word in self.valid_single_letter_words) \
            or (self.remove_non_alpha and not(lower_word.isalpha())) \
            or bool(CONTAINS_ONLY_IDENT_CHARS.search(lower_word)) \
            or bool(CONTAINS_MULTIPLE_IDENT_CHARS.search(lower_word)):
            return ''

        return word if self.keep_caps else lower_word

    def write_document(self):
        self.f_new.write(json.dumps(self.current_document) + "\n")

    def count_upper_case_letters(self,str_obj):
        count = 0
        for elem in str_obj:
            if elem.isupper():
                count += 1
        return count

    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def remove_start_section(self):
        if self.keep_start_section:
            return

        # get content body
        content = self.current_document[self.doc_property_content]

        if len(content)==0:
            self.logger.warning(f"Patent #{self.current_document[self.doc_property_patent_number]} has no content")
            return

        # split into words & clean
        raw_words = content.split()
        words = [x for x in raw_words if len(self.process_word(x)) > 0 ]

        if len(words)==0:
            self.logger.warning(f"Patent #{self.current_document[self.doc_property_patent_number]} has no words")
            return

        frac_threshold = 0.60

        # go through the first 1000 words, and break when the overall percentage of characters that are CAPS falls
        # below the threshold
        tot_cap = 0
        tot_len = 0
        for idx, word in enumerate(words[0:1000]):
            tot_cap += self.count_upper_case_letters(word)
            tot_len += len(word)
            if (idx > 10 and (tot_cap/tot_len) < frac_threshold):
                break

        first_words = words[0:idx]
        # print(first_words)

        # go through the remaining words in chunks of three words at a time, and again calculate the percentage
        # of CAPS-characters per chunk, and stop after it falls below the threshold. the first three chunks are
        # always included (some patents start with a few lowercase words)
        chunk_size = 3
        chunks = list(self.chunker(first_words, chunk_size))

        if len(chunks)==0:

            start_section = ""

        else:

            for idx, chunk in enumerate(chunks):
                c_upper = sum(map(self.count_upper_case_letters, chunk))
                c_all = sum(map(len, chunk))
                frac = (c_upper / c_all)
                # print(frac,chunk)
                if idx > 10 and  frac < frac_threshold:
                    break

            # join the chunks
            first_part = []
            for item in list(chunks)[:idx]:
                first_part += item

            # examine the last chunk, which fell below the threshold, but might contain some uppercase words at the start
            rest = []
            for word in chunk:
                if (self.count_upper_case_letters(word) / len(word) >= frac_threshold):
                    rest.append(word)
                else:
                    break

            start_section = first_part + rest
            for idx,item in enumerate(reversed(start_section)):
                if (self.count_upper_case_letters(item) / len(item)) > frac:
                    break

            if idx > 0:
                start_section = start_section[0:len(start_section)-idx]

        # reassemble start section, and chop it from the original body
        start_section = " ".join(start_section)
        self.current_document[self.doc_property_content] = " ".join(words)[len(start_section):].strip()
        self.logger.info(f"Patent #{self.current_document[self.doc_property_patent_number]} has {len(words)} words")

if __name__ == '__main__':

    pp = Preprocessor()
    pp.parse_arguments()
    pp.print_settings()
    pp.read_folder()
    pp.process_files()
