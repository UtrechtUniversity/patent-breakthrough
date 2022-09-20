# Patents Preprocessor

This program cleans US patent texts in order to make them more suitable for
training and analysis with NLP machine learning models, such as BERT.


## Cleaning actions
The program performs the following cleaning actions, some of which are specific
to use on historical US patent texts, while others are are more generally
applicable:

- Removal of unprintable characters (\u0080-\uffff)
- Removal of start section with 'boilerplate' text. Many patents start with
  a section that contains legal information pertaining to the patent holder and
  patent office, which carries little information useful to document embedding
  techniques. These sections can be recognized by structural use of capital
  letters (example: "ALSON, Straw Cutter, No. 10,001, Patented Sept, 6, 1853,
  UNITED STATES PATENT OFFICE. THOS. ALLISON, OF MITTON, NEW YORK. STRAW-CUTTER")
  (optional; default True)
- Removal of words that consist of a single character, other than 'a' and 'i'.
- Removal of numbers and words that contain numbers. (optional; default False)
- Removal of words that consist of identical characters only.
- Removal of words that contain sequences of more than two identical characters.

Note that some of these actions are useful only for errors that result from
faulty OCR, but will not unnecessarily remove words from texts that have no
such errors.

Optionally, the program converts all text to lowercase (optional; default True).

## Input
The program expects a JSONL-file, or a folder with JSONL-file(s) as input. Each
line in the input files is expected to contain a JSON-document describing a
single patent. Each is expected to have a property 'contents' containing the
patent text, and a property 'patent' containing the patents registration number
(these can be set through `set_doc_content_property()` and
`doc_property_patent_number()`). Other properties are copied to the output
_as is_.

## Output
Cleaned patents are written to the specified output folder, maintaining the
grouping of documents in JSONL-files as the input. Filenames are retained,
while adding a '_cleaned' suffix. Cleaned documents have the same format and
properties as the input.

## Usage
```
python preprocessor.py \
  --input INPUT \
  --output OUTPUT \
  [--remove_non_alpha] \
  [--keep_caps] \
  [--keep_start_section] \
  [--log_file LOG_FILE]
```
`--input`: expects a path to input plus extension e.g. './input/*.jsonl'.

`--output`: expects a folder, which is created if it doesn't exist.

`--log_file`: if you specify a log file, all log information that is written to
std out is also written to file.

## Converting text-files to JSONL
For the conversion of original text-files to JSONL, code from the earlier [patent impact](https://github.com/UtrechtUniversity/patent-impact)
&#128274; project was used, specifically the [parser](https://github.com/UtrechtUniversity/patent-impact/blob/main/production/01_parse.py).
