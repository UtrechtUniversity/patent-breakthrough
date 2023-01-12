from docembedder.models import TfidfEmbedder


def test_tfidf():
    documents = [
        "This is a very interesting sentence",
        "And here is another one",
    ]

    embedder = TfidfEmbedder(stop_words=None, min_df=1)
    embedder.fit(documents)
    X = embedder.transform(documents)
    assert X.shape == (2, 9)
