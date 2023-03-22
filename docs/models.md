# Models in docembedder

We have currently implemented 5 different models in the docembedder package:

- BERT: Pre-trained transformer models
- TF-IDF: Term-Frequency-Inverse Document Frequency model
- BP-EMB: Byte pair embedding model
- CountVec: A model similar to TF-IDF
- Doc2Vec: A model that uses deep-learning to train on the documents

Creating and fitting a model is done as follows:

```python
from docembedder.models import TfidfEmbedder

model = TfidfEmbedder(min_df=1)
model.fit(documents) # documents being a list of patents
embeddings = model.transform(documents)
```
