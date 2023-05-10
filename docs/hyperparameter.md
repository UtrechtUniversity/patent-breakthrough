# Hyper parameter optimization

Work on the optimization of the different document embedders in the context
of encoding patents is done in the `docembedder.hyperopt` sub-package. We use
the [hyperopt](https://github.com/hyperopt/hyperopt) package for the optimization
of the parameters.

Parameters are optimized according to the [classification](classification.md) performance.

Each of the models contains a method that defines the parameter space. This is searched
with the `ModelHyperopt` and `PreprocessorHyperopt` classes for the embedding models and
preprocessors respectively.

## Embedding model optimization

An example of how to optimize the model:

```python
from docembedder.hyperopt.utils import ModelHyperopt
from docembedder.models import TfidfEmbedder

hyper = ModelHyperopt(sim_spec, cpc_fp, patent_dir, trials="tfidf.pkl")
hyper.optimize("tfidf", TfidfEmbedder, max_evals=100, n_jobs=4)
hyper.dataframe("tfidf", TfidfEmbedder)
```

Using the trials keyword argument saves the trial runs with their parameters. For an explanation of
the sim_spec paramter, see [simulation_specifications](simulation_specifications.md)


## Preprocessing optimization

Since the preprocessing steps only include boolean parameters, the optimization
routines in the optimization is exhaustive (since hyperopt generally fails at this).

An example:

```python
from docembedder.hyperopt.utils import PreprocessorHyperopt
from docembedder.preprocessor import Preprocessor

hyper = PreprocessorHyperopt(sim_spec, cpc_fp, patent_dir, trials="tfidf.pkl")
hyper.optimize("prep", Preprocessor, n_jobs=4)
hyper.dataframe("prep")
```
