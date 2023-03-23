# Analysis of data

The analysis of the embedding data is done with the `DocAnalysis` class. It uses the [DataModel](datamodel.md)
as data input. The following analysis methods are implemented:

- Compute impact
- Compute novelty
- Compute performance with CPC classifications
- Compute autocorrelation

Usage:

```python
from docembedder.analysis import DocAnalysis
from docembedder.datamodel import DataModel

with DataModel("some_file.h5", read_only=False) as data:
	analysis = DocAnalysis(data)
	novelty = analysis.patent_novelties(some_window, some_model_name)
	impact = analysis.patent_impacts(some_window, some_model_name)
	auto_correlation = analysis.auto_correlation(some_window, some_model_name)
	cpc_cor = analysis.cpc_correlations(models=some_model_name)
```


