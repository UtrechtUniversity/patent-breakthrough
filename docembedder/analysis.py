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
        self.df_patents = pd.read_csv('../data/patents_concatenated.csv')
        self.df_patents_embeddings = self.df_patents.join(self.embeddings_df)
        self.forward_block = None
        self.backward_block = None

    @classmethod
    def from_dill(cls, path="../models/document_embeddings.dill"):
        """ Load embeddings to the memory

        Arguments
        ---------
        path: str
            path to the embedding file
        patent_index: int


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
        patent_year = self.df_patents.loc[patent_index]['year']

        print(patent_year)

        backward_years = patent_year - look_up_window
        forward_years = patent_year + look_up_window

        forward_block_list = []
        backward_block_list = []

        grouped_df = self.df_patents_embeddings.groupby('year')

        for key in grouped_df:
            if backward_years <= key < patent_year:  # backward n-years patents
                backward_block_sub = grouped_df.get_group(key)
                backward_block_list.append(backward_block_sub)

            if patent_year < key <= forward_years:  # forward n-years patents
                forward_block_sub = grouped_df.get_group(key)
                forward_block_list.append(forward_block_sub)

        self.forward_block = pd.concat(forward_block_list)
        self.backward_block = pd.concat(backward_block_list)

    def compute_similarity(self, patent_index):
        """
        Function to compute the novelty score, impact score, and influence of the patents for
        a window size of n-year using cosine similarity.

        Novelty score is calculated as the average of the cosine-similarity between Pi and
        patents in the years < focus_year

        Impact score is calculated as the average of the cosine_similarity between Pi and
        patents in the years > target_year

        Influence score = Impact_score / Novelty_score

        Arguments
        ---------
        patent_index: int
            index of the focus patent

        """

        backward_similarity = 0
        forward_similarity = 0
        self.collect_blocks(patent_index, 3)
        target_patent_vector = np.array(self.df_patents_embeddings.loc[patent_index]['embeddings'])

        # Calculate novelty of the focus patent
        for frow in range(len(self.backward_block)):
            backward_similarity += \
                cosine_similarity(target_patent_vector, np.array(
                    self.df_patents_embeddings.loc[frow]['embeddings']))
        average_backward_similarity = backward_similarity/len(self.backward_block)
        self.df_patents_embeddings.loc[patent_index]['novelty'] = average_backward_similarity

        # Calculate impact of the focus patent
        for brow in range(len(self.backward_block)):
            forward_similarity += \
                cosine_similarity(target_patent_vector, np.array
                (self.df_patents_embeddings.loc[brow]['embeddings']))
        average_forward_similarity = forward_similarity / len(self.forward_block)
        self.df_patents_embeddings.loc[patent_index]['impact'] = average_forward_similarity

        # calculate influence of the focus patent
        self.df_patents_embeddings.loc[patent_index]['influence'] = \
            average_forward_similarity/average_backward_similarity


if __name__ == "__main__":
    patent_analyser = DOCSimilarity.from_dill()
    patent_analyser.collect_blocks2(patent_number=2445033, look_up_window=3)
    # patent_analyser.compute_similarity()
