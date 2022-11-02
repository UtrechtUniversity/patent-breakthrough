# Patents Preprocessor
This collection of scripts converts and cleans US patent texts in order to
make them more suitable for training and analysis with NLP machine learning
models, such as BERT.


## Conversion of text to JSON
Use `parser.py` to convert raw text files containing US patent texts to
structured JSONL-files, packaged in LZMA-compressed archives (.xz files).
By calling the function `compress_raw()` with as attributes file pointers to
the text file with raw patents, and a file containing a key for translating
patent number to year, patents are divided according to the year of their
original application, and each .xz file covers one year.

Depending on the data, it might be necessary to alter the regular expression
`PATENT_FILE_PATTERN` to make it correctly recognize the start of each pattern.


## Preprocessor
The `preprocessor.py` script performs the following cleaning actions. Some are
specific to use on historical US patent texts, while others are are more
generally applicable:

- Removal of unprintable characters (\u0080-\uffff).
- Reassembly of split words. OCR often introduces erroneous spaces, sometimes
  resulting in split words ("tem perature", "collaps ible" etc). When providing
  a word list of valid words, the preprocessor attempts to restore the original
  words.
- Removal of start section with 'boilerplate' text. Many patents start with
  a section that contains legal information pertaining to the patent holder and
  patent office, which carries little information useful to document embedding
  techniques. These sections can be heuristically recognized by the dominant use
  of capital letters (example: "ALSON, Straw Cutter, No. 10,001, Patented Sept,
  6, 1853, UNITED STATES PATENT OFFICE. THOS. ALLISON, OF MITTON, NEW YORK.
  STRAW-CUTTER") (optional; default True).
- Removal of words that consist of a single character, other than 'a' and 'i'.
- Removal of numbers and words that contain numbers. (optional; default False)
- Removal of words that consist of identical characters only.
- Removal of words that contain sequences of more than two identical characters.

Note that some of these actions are useful only for errors that result from
faulty OCR, but will not unnecessarily remove words from texts that have no
such errors.

Optionally, the program converts all text to lowercase (optional; default True).

### Parameters
+ log_level: id. (default: logging.INFO)
+ log_file: path to logfile (default: None = only logging to Stdout)
+ log_format: id. (default: %(asctime)s [%(levelname)s] %(message)s)
+ keep_empty_patents: whether to retain patents that have no body text
  (default: False)
+ keep_missing_years: idem, if year is missing (default: False)
+ keep_caps: keep capital letters, do not make everything lower case
  (default: False)
+ keep_start_section: do not remove the boilerplate text at the (default: False)
+ remove_non_alpha: whether to remove all non-alpha characters (default: False)
+ input_dir: input; see below (default: None).
+ output_dir: output; see below (default: None).
+ lexicon_path: path to lexicon; see below (default: None). Specifying a
  lexicon path automatically implies the application of word reassembly.

### Input
The program expects a JSONL-file, or a folder with JSONL-file(s) as input.
When specifying a folder with files, use a pattern that is recognized by glob;
so `/data/*.jsonl`, rather than `/data/`.

Each line in the input files is expected to contain a JSON-document describing a
single patent. Each is expected to have a property 'contents' containing the
patent text, and a property 'patent' containing the patents registration number
(these can be set through `set_doc_content_property()` and
`doc_property_patent_number()`). Other properties are copied to the output
_as is_.


### Output
Cleaned patents are written to the specified output folder, maintaining the
grouping of documents in JSONL-files as the input. Filenames are retained,
while adding a '_cleaned' suffix. Cleaned documents have the same format and
properties as the input.

When the output folder is omitted, the script does not write output to file.
Instead, call `preprocess_file()` directly to access preprocessed patents
(function return a list of processed patents, and a dict with some statistics).


### Lexicon
Should be a path to a CSV or text file with a list of valid words, one word per
line. If the preprocessor comes across two consecutive strings (tokens) that do
not appear in the lexicon individually, but do appear in the lexicon as a
concatenated string, the tokens are replaced with that word.

During development, a lexicon was created from the [wordlist-english](https://github.com/jacksonrayhamilton/wordlist-english),
repository, by concatenating all files in the [sources folder](https://github.com/jacksonrayhamilton/wordlist-english/tree/master/sources)
into one lexicon file.
