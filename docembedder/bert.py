""" Huggingface BERT class."""

from typing import Iterable, Union

from pathlib import Path
import json
import time


import numpy as np
import numpy.typing as npt
import scipy
import pandas as pd
import dill

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer

from docembedder.base import BaseDocEmbedder


class BERTEmbedder(BaseDocEmbedder):
    """BERT embedding based on Hugging face pre-trained models.
    There are a number of pre-trained models on the patent data available.
    - prithivida/bert-for-patents-64d : https://huggingface.co/prithivida/bert-for-patents-64d
    - anferico/bert-for-patents : https://huggingface.co/anferico/bert-for-patents
    - AI-Growth-Lab/PatentSBERTa : https://huggingface.co/AI-Growth-Lab/PatentSBERTa
     """

    def __init__(self,
                 pretrained_model: str = "sentence-transformers/stsb-distilbert-base",
                 text_column: str = "contents"):
        self.pretraiend_model = pretrained_model
        self.text_column = text_column
        self._sbert_model = SentenceTransformer(pretrained_model)

    def fit(self, documents: Iterable[str]) -> None:
        embedding_vectors = self._sbert_model.encode(documents)

        path_model = Path(__file__).parent / "../models/document_embeddings.model"
        with open(path_model, 'wb') as file:
            dill.dump(embedding_vectors, file)

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:
        pass

    @staticmethod
    def create_similarity_matrix(embeddings):
        """create similarity matrix
        """
        pairwise_similarities = cosine_similarity(embeddings)
        return pairwise_similarities

    @staticmethod
    def create_difference_matrix(embeddings):
        """ Create difference matrix
        """
        pairwise_similarities = euclidean_distances(embeddings)
        return pairwise_similarities

    @staticmethod
    def most_similar(documents, doc_id, similarity_matrix, matrix):
        """ Find most similar documents
        """
        print(f'Document: {documents.iloc[doc_id]["contents"]}')
        print('\n')
        print('Most similar Document:')
        if matrix == 'Cosine Similarity':
            similar_index = np.argsort(similarity_matrix[doc_id])[-2::]
        elif matrix == 'Euclidean Distance':
            similar_index = np.argsort(similarity_matrix[doc_id])
        for index in similar_index:
            if index == doc_id:
                continue
            print('\n')
            print(f'Document: {documents.iloc[index]["contents"]}')
            print(f'{matrix} : {similarity_matrix[doc_id][index]}')

    @property
    def embedding_size(self) -> int:
        pass


if __name__ == "__main__":
    start_time = time.time()

    with open('../data/sample_cleaned_v3.jsonl', encoding="utf-8") as p:
        patent_df = pd.DataFrame(json.loads(line) for line in p)
    # print(patent_df)
    documents_df = patent_df['contents']
    patent = BERTEmbedder()
    # patent.fit(documents_df)
    #
    path = Path(__file__).parent / "../models/document_embeddings.model"
    with open(path, 'rb') as f:
        document_embeddings = dill.load(f)
    print(document_embeddings.shape)

    patent.most_similar(patent_df, 0, patent.create_similarity_matrix(document_embeddings),
                        'Cosine Similarity')

    # print("--- %s seconds ---" % (time.time() - start_time))
