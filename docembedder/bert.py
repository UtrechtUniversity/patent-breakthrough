""" Huggingface BERT class."""

from typing import Iterable, Union

import numpy as np
import numpy.typing as npt
import scipy
import time
import json
import pandas as pd
from pathlib import Path
import dill

from docembedder.base import BaseDocEmbedder

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


class BERTEmbedder(BaseDocEmbedder):
    """BERT embedding based on Hugging face pre-trained models.
    There are a number of pre-trained models on the patent data available.
    - prithivida/bert-for-patents-64d : https://huggingface.co/prithivida/bert-for-patents-64d
    - anferico/bert-for-patents : https://huggingface.co/anferico/bert-for-patents
    - AI-Growth-Lab/PatentSBERTa : https://huggingface.co/AI-Growth-Lab/PatentSBERTa
     """

    def __init__(self, pretrained_model: str = "sentence-transformers/stsb-distilbert-base", text_column: str = "contents"):
        self.pretraiend_model = pretrained_model
        self.text_column = text_column
        self._sbert_model = SentenceTransformer(pretrained_model)

    def fit(self, documents: Iterable[str]) -> None:
        document_embeddings = self._sbert_model.encode(documents)

        path = Path(__file__).parent / "../models/document_embeddings.model"
        with open(path, 'wb') as f:
            dill.dump(document_embeddings, f)

    def transform(self, documents: Union[str, Iterable[str]]) -> Union[
            scipy.sparse.base.spmatrix, npt.NDArray[np.float_]]:
        pass

    @property
    def embedding_size(self) -> int:
        return self._sbert_model

    @property
    def most_similar(self, doc_id, similarity_matrix, matrix):
        print(f'Document: {documents_df.iloc[doc_id]["contents"]}')
        print('\n')
        print('Similar Documents:')
        if matrix == 'Cosine Similarity':
            similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
        elif matrix == 'Euclidean Distance':
            similar_ix = np.argsort(similarity_matrix[doc_id])
        for ix in similar_ix:
            if ix == doc_id:
                continue
            print('\n')
            print(f'Document: {documents_df.iloc[ix]["contents"]}')
            print(f'{matrix} : {similarity_matrix[doc_id][ix]}')


if __name__ == "__main__":
    start_time = time.time()

    patent_vector = BERTEmbedder()
    with open('../data/sample_cleaned_v3.jsonl') as f:
        patent_df = pd.DataFrame(json.loads(line) for line in f)
    documents_df = patent_df['contents']
    # patent_vector.fit(documents_df)
    path = Path(__file__).parent / "../models/document_embeddings.model"
    with open(path, 'rb') as f:
        document_embeddings = dill.load(f)
    print(document_embeddings.shape)
    # pairwise_similarities = cosine_similarity(document_embeddings)
    # pairwise_differences = euclidean_distances(document_embeddings)
    #
    # print(patent_vector.most_similar(0, pairwise_similarities, 'Cosine Similarity'))
    # print(patent_vector.most_similar(0, pairwise_differences, 'Euclidean Distance'))
    #
    # print("--- %s seconds ---" % (time.time() - start_time))



