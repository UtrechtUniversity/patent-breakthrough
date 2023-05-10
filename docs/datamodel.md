# Data model of docembedder

Docembedder uses a data model to store the results of the models, and analyses thereof.
The data model uses h5py to store the data in an hdf5 file (`io.BytesIO` is also supported
for in memory computation).

The most detailed and up-to-date information is found in the `docembedder.datamodel` module.
Currently the following (and more) can be stored/cached in the data file:

- embeddings
- impact/novelty of each patent
- CPC classification correlations
- models
- preprocessors
- windows: This is defined as a series of consecutive years for which the embeddings are computed.
To open a file containing the model data:

```python
from docembedder.datamodel import DataModel

with DataModel("some_file.h5", read_only=False) as data:
	embeddings = data.load_embeddings(window_name="1845-1856", model_name="tfidf")
```
