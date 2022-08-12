"""Module containing patent similarity analysis"""

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import dill


class DOCSimilarity:
    """ Class to create similarity and difference matrix

    Arguments
    ---------
    path: path to the saved document embeddings
     """

    def __init__(self, path=Path(__file__).parent / "../models/document_embeddings.model"):
        self.path = path
        with open(self.path, 'rb') as file:
            self._embeddings = dill.load(file)

    def create_similarity_matrix(self):
        """create similarity matrix
        """
        pairwise_similarities = cosine_similarity(self._embeddings)
        return pairwise_similarities

    def create_difference_matrix(self):
        """ Create difference matrix
        """
        pairwise_similarities = euclidean_distances(self._embeddings)
        return pairwise_similarities

    @staticmethod
    def most_similar(documents, doc_id, similarity_matrix, measure):
        """ Find most similar documents
        """
        print(f'Document: {documents.iloc[doc_id]["contents"]}')
        print('\n')
        print('Most similar Document:')
        if measure == 'Cosine Similarity':
            similar_index = np.argsort(similarity_matrix[doc_id])[-2::]
        elif measure == 'Euclidean Distance':
            similar_index = np.argsort(similarity_matrix[doc_id])
        for index in similar_index:
            if index == doc_id:
                continue
            print('\n')
            print(f'Document: {documents.iloc[index]["contents"]}')
            print(f'{measure} : {similarity_matrix[doc_id][index]}')
