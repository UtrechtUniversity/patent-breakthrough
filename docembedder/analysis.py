"""Module containing patent similarity analysis"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import dill
import pandas as pd


class DOCSimilarity:
    """ Class to create similarity and difference matrix

    Arguments
    ---------
    embeddings: numpy.ndarray
        Document vectors generated by BERT/other methods
     """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.embeddings_df = pd.DataFrame()
        self.embeddings_df['embeddings'] = self.embeddings.tolist()
        # self.df_patents = pd.read_csv('../data/patents_concatenated.csv')
        self.df_patents = pd.read_csv('data/tst_sample.csv')
        self.df_patents_embeddings = self.df_patents.join(self.embeddings_df, how='left')
        self.forward_block = None
        self.backward_block = None
        self.window_size = 3

    @classmethod
    def from_dill(cls, path="models/document_embeddings_tst.dill"):
        """ Load embeddings to the memory

        Arguments
        ---------
        path: str
            path to the embedding file

        Returns
        -------
        embeddings:
            Initialized embeddings

        """
        with open(path, 'rb') as file:
            embeddings = dill.load(file)
        return cls(embeddings)

    def collect_blocks(self, patent_index, look_up_window):
        """
        Collect a block of patents for a window of n years regarding the year of the focus
        patent.

        """

        # patent_year = self.df_patents_embeddings[
        #     self.df_patents_embeddings['patent'] == patent_number]['year'].values[0]
        grouped_df = self.df_patents_embeddings.groupby('year')
        max_year = max(grouped_df.groups.keys())
        min_year = min(grouped_df.groups.keys())

        patent_year = self.df_patents.loc[patent_index]['year']

        backward_years = patent_year - look_up_window
        backward_years = max(backward_years, min_year)
        forward_years = patent_year + look_up_window
        forward_years = min(forward_years, max_year)

        forward_block_list = []
        backward_block_list = []

        for key in grouped_df.groups.keys():
            if backward_years <= key < patent_year:  # backward n-years patents
                backward_block_sub = grouped_df.get_group(key)
                backward_block_list.append(backward_block_sub)

            if patent_year < key <= forward_years:  # forward n-years patents
                forward_block_sub = grouped_df.get_group(key)
                forward_block_list.append(forward_block_sub)

        if forward_block_list:
            self.forward_block = pd.concat(forward_block_list)
        if backward_block_list:
            self.backward_block = pd.concat(backward_block_list)

    def compute_impact(self, patent_index):

        backward_similarity = 0
        forward_similarity = 0
        focus_patent_vector = \
            np.array(self.df_patents_embeddings.loc[patent_index]['embeddings'])

        # Calculate backward similarities
        if self.backward_block is not None:

            for bkw in self.backward_block.index:
                backward_similarity += cosine_similarity(
                    [focus_patent_vector],
                    np.array([self.df_patents_embeddings.loc[bkw]['embeddings']])
                )

            average_backward_similarity = backward_similarity / len(self.backward_block)
            average_backward_similarity_list = average_backward_similarity.tolist()
            average_backward_similarity_number = average_backward_similarity_list[0][0]
        else:
            average_backward_similarity_number = None

        # Calculate forward similarities
        if self.forward_block is not None:
            for frw in self.forward_block.index:
                forward_similarity += cosine_similarity(
                    [focus_patent_vector],
                    np.array([self.df_patents_embeddings.loc[frw]['embeddings']])
                )
            average_forward_similarity = forward_similarity / len(self.forward_block)
            average_forward_similarity_list = average_forward_similarity.tolist()
            average_forward_similarity_number = average_forward_similarity_list[0][0]
        else:
            average_forward_similarity_number = None

        # Calculate influence. backwards/forwards
        if (average_backward_similarity_number is not None) & (average_forward_similarity_number is not None):
            self.df_patents_embeddings.loc[patent_index, 'impact'] = \
                average_backward_similarity_number / average_forward_similarity_number

    def compute_novelty(self, patent_index):
        """
        Function for calculating the focused patent's novelty for the period of n-year.
        Novelty score is calculated as the average of the cosine-similarity between Pi and
        patents in the years < focus_year

        Arguments
        ----------
        patent_index: int
            The index of focused patent
        """
        backward_dissimilarity = 0
        focus_patent_vector = \
            np.array(self.df_patents_embeddings.loc[patent_index]['embeddings'])

        # Calculate novelty of the focus patent
        if self.backward_block is not None:
            for brow in self.backward_block.index:
                backward_dissimilarity += 1 - cosine_similarity(
                    [focus_patent_vector],
                    np.array([self.df_patents_embeddings.loc[brow]['embeddings']])
                )
            average_backward_similarity = backward_dissimilarity / len(self.backward_block)
            average_backward_similarity_list = average_backward_similarity.tolist()
            self.df_patents_embeddings.loc[patent_index, 'novelty'] = \
                average_backward_similarity_list[0][0]

    def compute_similarity(self):
        """
        Function to compute the novelty score, impact score, and influence of the patents for
        a window size of n-year using cosine similarity.
        """
        for patent_index in range(len(self.df_patents_embeddings)):
            self.collect_blocks(patent_index, self.window_size)
            self.compute_novelty(patent_index)
            self.compute_impact(patent_index)


if __name__ =='__main__':
    patent_analyser = DOCSimilarity.from_dill()
    patent_analyser.compute_similarity()
    patent_analyser.df_patents_embeddings
