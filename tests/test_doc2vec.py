""" pytest file for doc2vec.py"""
from docembedder import D2VEmbedder


def test_doc2vec():
    """ Function to test doc2vec.py functionality
    """
    documents = [
        "This is a very interesting sentence",
        "And here is another one"
    ]

    embedder = D2VEmbedder()
    embedder.fit(documents)
    embeddings = embedder.transform(documents)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 100
    assert len(embeddings[1]) == 100
