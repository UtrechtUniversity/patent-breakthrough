from pytest import mark

from docembedder import BPembEmbedder


@mark.parametrize(
    "vector_size", [100, 300])
def test_tfidf(vector_size):
    documents = [
        "This is a very interesting sentence",
        "And here is another one",
    ]

    embedder = BPembEmbedder(vector_size=vector_size)
    embedder.fit(documents)
    X = embedder.transform(documents)
    assert X.shape == (2, vector_size)
