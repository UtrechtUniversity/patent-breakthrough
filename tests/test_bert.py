""" pytest file for bert.py"""
from docembedder.models import BERTEmbedder


def test_bert():
    """ Function to test bert.py functionality
    """
    documents = [
        "This is a very interesting sentence",
        "And here is another one"
    ]

    embedder = BERTEmbedder()
    embedder.fit(documents)
    embeddings = embedder.transform(documents)
    assert embeddings.shape == (2, 64)
