"""Module containing patent similarity analysis"""

from typing import Sequence, Dict, Union, Tuple, Any, Optional
from pathlib import Path

from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from numpy import typing as npt

from docembedder.base import BaseDocEmbedder
from docembedder.classification import PatentClassification


class DOCSimilarity:
    """ Class to create similarity and difference matrix

    Arguments
    ---------
    embeddings: numpy.ndarray
        Document vectors generated by BERT/other methods
     """

    def __init__(self, embeddings, window_size, df_patent):
        self.embeddings = embeddings
        self.embeddings_df = pd.DataFrame()
        self.embeddings_df['embeddings'] = self.embeddings.tolist()
        self.df_patents = df_patent
        self.df_patents_embeddings = self.df_patents.join(self.embeddings_df, how='left')
        self.forward_block = None
        self.backward_block = None
        self.window_size = window_size

    def collect_blocks(self, patent_index):
        """
        Collect a block of patents for an n-year window regarding the year of the focus
        patent.

        """

        # patent_year = self.df_patents_embeddings[
        #     self.df_patents_embeddings['patent'] == patent_number]['year'].values[0]
        grouped_df = self.df_patents_embeddings.groupby('year')
        max_year = max(grouped_df.groups.keys())
        min_year = min(grouped_df.groups.keys())

        patent_year = self.df_patents.loc[patent_index]['year']

        backward_years = patent_year - self.window_size
        backward_years = max(backward_years, min_year)
        forward_years = patent_year + self.window_size
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
        """ Function to calculate the impact of the focused patent for the period of n-year.
        Impact score is calculated as the average of the backward similarity / the average of
        the forward similarity.
        Arguments
        ----------
        patent_index: int
            The index of focused patent

        """

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
        if (average_backward_similarity_number is not None)\
                & (average_forward_similarity_number is not None):
            self.df_patents_embeddings.loc[patent_index, 'impact'] = \
                average_backward_similarity_number / average_forward_similarity_number

    def compute_novelty(self, patent_index):
        """
        Function for calculating the focused patent's novelty for the period of n-year.
        Novelty score is calculated as the average of the 1- cosine-similarity between Pi and
        patents in the n-backward years

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
        Function to compute the novelty score, and impact score of the patents for
        a window size of n-year using cosine similarity.
        """
        for patent_index in range(len(self.df_patents_embeddings)):
            self.collect_blocks(patent_index)
            self.compute_novelty(patent_index)
            self.compute_impact(patent_index)

def get_model_correlations(model: BaseDocEmbedder,
                           train_documents: Sequence[str],
                           test_documents: Optional[Sequence[str]]=None
                           ) -> npt.NDArray[np.float_]:
    """Get all cross correlations of the embeddings for a model

    Arguments
    ---------
    model:
        Model to create the embeddings for.
    documents:
        Patents to encode.

    Returns
    -------
    cross_cor:
        Correlations between all the patents [len(documents, len(documents)].
    """
    if test_documents is None:
        test_documents = train_documents
    model.fit(train_documents)
    embeddings = model.transform(test_documents)
    cross_cor = embeddings.dot(embeddings.T)
    return cross_cor


def _sample_class_similarity(patents: Sequence[Dict],
                             pat_class: PatentClassification) -> Tuple[int, int, float]:
    # Sometimes we are missing patent classifications, resample if this is the case.
    n_try = 1000
    for _ in range(n_try):
        i_patent, j_patent = np.random.choice(len(patents), size=2, replace=False)
        try:
            class_cor = pat_class.get_similarity(
                patents[i_patent]["patent"],
                patents[j_patent]["patent"])
            return i_patent, j_patent, class_cor
        except ValueError:
            pass

    raise ValueError("Cannot find patents with classification.")


def classification_benchmark(
        patents: Sequence[Dict],
        models: Dict[str, BaseDocEmbedder],
        n_patents: int=1000,
        n_class_sample: int=500,
        class_fp: Union[str, Path]=Path("..", "data", "GPCPCs.txt"),
        ) -> Dict[str, float]:
    """Benchmark different models with the GPCPC classifications.

    See `compare_classification_similarity` function for more details.

    Arguments
    ---------
    patents:
        Patents to sample the correlations from.
    models:
        Dictionary with name: model.
    class_fp:
        Filename for the patent classifications.
    n_patents:
        Number of patents selected.
    n_class_sample:
        Number of samples (i_patent vs j_patent) for benchmarking purposes. Higher means longer
        running times but better accuracy (at least up to sampling the whole matrix).

    """
    if n_patents is None or n_patents > len(patents):
        n_patents = len(patents)
    documents = [p["contents"] for p in patents]
    sampled_idx = np.random.choice(len(patents), size=n_patents, replace=False)
    sampled_patents = [patents[i] for i in sampled_idx]
    sampled_documents = [documents[i] for i in sampled_idx]
    model_cor: Dict[str, Any] = {}
    for model_name, model in models.items():
        model_cor[model_name] = get_model_correlations(model, documents, sampled_documents)

    results = compare_classification_similarity(sampled_patents, model_cor, class_fp=class_fp,
                                                n_sample=n_class_sample)

    return results


def compare_classification_similarity(
        patents: Sequence[Dict],
        similarity_matrices: Dict[str, npt.NDArray[np.float_]],
        class_fp: Union[str, Path]=Path("..", "data", "GPCPCs.txt"),
        n_sample: int=10000) -> Dict[str, float]:
    """Compute the performance of models by comparing patent similarities with classifications

    It samples combinations of patents, say patent i and j. For each model it retrieves
    the similarity between patent i and j. It also retrieves the similariry of GPCPC classifications
    i and j. The details for this can be found in the PatentClassification class. After sampling
    n_sample of these classifications, a performance indicator is created by computing the
    spearman-r correlation between the model similarities and the classification similarities;
    if two patents have very similar patent classifications, we expect that a good model will
    indicate high(er) similarities between those two classifications as well.

    Arguments
    ---------
    patents:
        Patents to sample the correlations from.
    similarity_matrices:
        For each model, this contains a cross correlation/similarity matrix.
    class_fp:
        Filename for the patent classifications.
    n_patents:
        Number of patents selected.
    n_sample:
        Number of samples (i_patent vs j_patent) for benchmarking purposes. Higher means longer
        running times but better accuracy (at least up to sampling the whole matrix).
    """
    pat_class = PatentClassification(class_fp)

    # Create empty arrays for the model correlations for each model.
    sampled_correlations = {model_name: np.zeros(n_sample) for model_name in similarity_matrices}
    class_correlations = np.zeros(n_sample)
    for i_sample in range(n_sample):
        i_patent, j_patent, class_cor = _sample_class_similarity(patents, pat_class)

        # Fill the model correlation arrays.
        for model_name, sim_mat in similarity_matrices.items():
            sampled_correlations[model_name][i_sample] = sim_mat[i_patent, j_patent]
        class_correlations[i_sample] = class_cor

    # Compute the performance of each model with the spearman-r correlation.
    model_performances = {model_name: stats.spearmanr(class_correlations, model_cor).correlation
                          for model_name, model_cor in sampled_correlations.items()}
    return model_performances
