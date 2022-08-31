""" pytest file for bert.py"""
import dill as pickle

from docembedder import BERTEmbedder


def test_bert():
    """ Function to test bert.py functionality
    """
    documents = [
        "This is a very interesting sentence",
        "And here is another one"
    ]

    embedder = BERTEmbedder(model_path="./models/test_document_embeddings.dill")
    embedder.fit(documents)

    with open(embedder.model_path, 'rb') as file:
        embeddings = pickle.load(file)
        print(embeddings)
        assert embeddings.shape == (2, 768)
