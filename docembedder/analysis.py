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

    def __init__(self, embeddings):
        self.embeddings = embeddings

    @classmethod
    def from_drill(cls, path=Path(__file__).parent / "../models/document_embeddings.model"):
        with open(path, 'rb') as file:
            embeddings = dill.load(file)
        return cls(embeddings)

    def cosine_similarity_matrix(self):
        """create similarity matrix
        """
        pairwise_similarities = cosine_similarity(self.embeddings)
        return pairwise_similarities

    def euclidean_similarity_matrix(self):
        """ Create difference matrix
        """
        pairwise_similarities = 1-euclidean_distances(self.embeddings)
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
            most_similar_documents = documents.iloc[index]["contents"]
            most_similar_document_index = similarity_matrix[doc_id][index]
            return most_similar_documents, most_similar_document_index
            # print('\n')
            # print(f'Document: {documents.iloc[index]["contents"]}')
            # print(f'{measure} : {similarity_matrix[doc_id][index]}')
