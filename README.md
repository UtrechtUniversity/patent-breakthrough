# Patent breakthrough

![tests](https://github.com/UtrechtUniversity/patent-breakthrough/actions/workflows/python-package.yml/badge.svg)


The code in this repository is used to identify breakthrough innovations in historical patents from the [USPTO](https://www.uspto.gov/).
The `docembedding` Python package contains a variety of methods for creating document embeddings. We have optimized and tested these methods for their ability to predict similarity between patents. This was done by maximizing the cosine similarity between patents that are classified into the same technology class, and minimizing cosine similarity between patents that fall into different technology classes. These methods with optimized parameters are then used to create document embeddings. From these embeddings, novelty scores are created using cosine similarities between the focal patent and patents in the previous n years and subsequent n years.

## Getting Started

Clone this repository to your working station to obtain example notebooks and python scripts:

```
git clone https://github.com/UtrechtUniversity/patent-breakthrough.git
```

### Prerequisites

To install and run this project you need to have the following prerequisites installed.

```
- Python [>=3.8, <3.11]
- jupyterlab (or any other program to run jupyter notebooks)
```
To install jupyterlab:
```
pip install jupyterlab
```

### Installation

To run the project, ensure to install the project's dependencies

```sh
pip install git+https://github.com/UtrechtUniversity/patent-breakthrough.git
```

### Built with

These packages are automatically installed in the step above:

- [scikit-learn](https://scikit-learn.org/)
- [gensim](https://pypi.org/project/gensim/)
- [sbert](https://www.sbert.net/)
- [bpemb](https://bpemb.h-its.org/)


## Usage

### 1. Preparation

First you need to make sure that you have the data prepared. There should be a directory with *.xz files, which should have the year, so 1923.xz, 1924.xz, 1925.xz, etc. If this is not the case and you have only the raw .txt files, then you have to compress your data:

```python
from docembedder.preprocessor.parser import compress_raw
compress_raw(some_file_name, "year.csv", some_output_dir)
```

Here, "year.csv" should be a file that that contains the patent ids and the year in which they were issued.



### 2. Hyper parameter optimization

There are procedures to optimize the preprocessor and ML models with respect to predicting CPC classifications. This is not a necessary step to compute the novelties and impacts, and has already been done for patents 1838-1951. For more information on how to optimize the models, see the [documentation](docs/hyperparameter.md).

### 3. Preprocessing

To improve the quality of the patents, and process/remove the start sections and such, it is necessary to preprocess these raw files. This is done using the `Preprocessor` and `OldPreprocessor` classes, for example:

```python
from docembedder.preprocessor import Preprocessor, OldPreprocessor

prep = Preprocessor()
old_prep = OldPreprocessor()
documents = prep.preprocess_file("1928.xz")
```

Normally however, we do not need to do preprocessing as a seperate step. We can compute the embeddings directly, which is explained in the next section.


### 4. Embedding models

There are 5 different embedding models implemented to compute the embeddings:

```python
from docembedder.models import CountVecEmbedder, D2VEmbedder, BPembEmbedder
from docembedder.models import TfidfEmbedder, BERTEmbedder
model = BERTEmbedder()
model.fit(documents)
embeddings = model.transform(documents)
```

These models can have different parameters for training, see the section on hyper parameter models. The result can be either sparse or dense matrices. The functions and methods in this package work with either in the same way.

### 5. Computing embeddings

The prepared data can be analysed to compute the embeddings for each of the patents using the `run_models` function. This function has the capability to run in parallel, in case you have more than one core on your CPU for examples.

Before we can run, we have to tell docembedder the parameters of the run, which is done through the `SimulationSpecification` class:

```python
from docembedder.utils import SimulationSpecification
sim_spec = SimulationSpecification(
    year_start=1838,  # Starting year for computing the embeddings.
    year_end=1951, # Last year for computing the embeddings.
    window_size=21,  # Size of the window to compute the embeddings for.
    window_shift=1,  # How many years between subsequent windows.
    debug_max_patents=100  # For a trial run we sample the patents instead, remove for final run.
)
```

An example to create a file with the embeddings is:

```python
from docembedder.utils import run_models
run_models({"bert": BERTEmbedder()}, model, sim_spec, output_fp, cpc_fp)
```

The output file is then a HDF5 file, which stores the embeddings for all patents in all windows.

### 6. Computing novelty and impact

To compute the novelty and impact we're using the `Analysis` class:
```python
from docembedder.analysis import DocAnalysis
with DataModel(output_fp, read_only=False) as data
    analysis = DocAnalysis(data)
    results = analysis.compute_impact_novelty("1920-1940", "bert")
```

The result is a dictionary that contains the novelties and impacts for each of the patents in that window (in this case 1920-1940).

## About the Project

**Date**: February 2023

**Researcher(s)**:

- Benjamin Cornejo Costas (b.j.cornejocostas@uu.nl)

**Research Software Engineer(s)**:

- Raoul Schram
- Shiva Nadi
- Maarten Schermer
- Casper Kaandorp
- Jelle Treep (h.j.treep@uu.nl)

### License

The code in this project is released under [MIT license](LICENSE).

### Attribution and academic use

Manuscript in preparation

## Contributing

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

To contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Benjamin Cornejo Costas - b.j.cornejocostas@uu.nl

Project Link: [https://github.com/UtrechtUniversity/patent-breakthrough](https://github.com/UtrechtUniversity/patent-breakthrough)
