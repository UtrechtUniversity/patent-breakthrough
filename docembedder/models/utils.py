from docembedder.models.bert import BERTEmbedder
from docembedder.models.tfidf import TfidfEmbedder
from docembedder.models.doc2vec import D2VEmbedder
from docembedder.models.bpemb import BPembEmbedder
from docembedder.models.countvec import CountVecEmbedder


def create_model(model_type, **kwargs):
    all_models = [BERTEmbedder, TfidfEmbedder, D2VEmbedder, BPembEmbedder,
                  CountVecEmbedder]

    try:
        model_class = [x for x in all_models if x.__name__ == model_type][0]
    except IndexError as e:
        raise ValueError(f"Unknown model type: {model_type}.") from e
    return model_class(**kwargs)
