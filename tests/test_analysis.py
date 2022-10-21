"""Test analysis.py functionality"""
from pandas.testing import assert_frame_equal
import pandas as pd
import pytest

from docembedder import DOCSimilarity
from docembedder import D2VEmbedder


PATENTS = {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           'patent': [100, 202, 250, 320, 370, 385, 415, 440, 500, 550],
           'contents': ['test sentence', 'test sentence', 'test sentence', 'test sentence',
                        'test sentence', 'test sentence', 'test sentence', 'test sentence',
                        'test sentence', 'test sentence'],
           'year': [1997, 1998, 1998, 1999, 2000, 2001, 2001, 2002, 2003, 2003]
           }

PATENT_INDEX = 4
BACKWARD_BLOCK_EXPECTED = pd.DataFrame({'index': [0, 1, 2, 3],
                                        'patent': [100, 202, 250, 320],
                                        'contents': ['test sentence', 'test sentence',
                                                     'test sentence', 'test sentence'],
                                        'year': [1997, 1998, 1998, 1999]
                                        })
FORWARD_BLOCK_EXPECTED = pd.DataFrame({'index': [5, 6, 7, 8, 9],
                                       'patent': [385, 415, 440, 500, 550],
                                       'contents': ['test sentence', 'test sentence',
                                                    'test sentence', 'test sentence',
                                                    'test sentence'],
                                       'year': [2001, 2001, 2002, 2003, 2003]
                                       })


def process_data():
    """ Set parameters for expected dataset"""
    df_patents = pd.DataFrame(data=PATENTS)
    documents = df_patents['contents']
    model = D2VEmbedder()
    model.fit(documents)
    embeddings = model.transform(documents)
    embeddings = pd.Series(embeddings)
    sim = DOCSimilarity(embeddings=embeddings, window_size=3, df_patent=df_patents)
    return sim


def test_collect_blocks():
    """ Function to test collect_blocks() functionality"""
    sim = process_data()

    sim.collect_blocks(PATENT_INDEX)

    assert_frame_equal(
        sim.backward_block[['index', 'patent', 'contents', 'year']].reset_index(drop=True),
        BACKWARD_BLOCK_EXPECTED.reset_index(drop=True))

    assert_frame_equal(
        sim.forward_block[['index', 'patent', 'contents', 'year']].reset_index(drop=True),
        FORWARD_BLOCK_EXPECTED.reset_index(drop=True))


@pytest.mark.parametrize("patent_index", [3, 4])
def test_compute_impact(patent_index):
    """Function to test compute_impact() functionality"""
    sim = process_data()
    sim.collect_blocks(patent_index)
    sim.compute_impact(patent_index)

    assert int(sim.df_patents_embeddings.loc[patent_index, 'impact']) == 1


@pytest.mark.parametrize("patent_index", [3, 4])
def test_compute_novelty(patent_index):
    """Function to test compute_novelty functionality"""
    sim = process_data()
    sim.collect_blocks(patent_index)
    sim.compute_impact(patent_index)
    sim.compute_novelty(patent_index)

    assert int(sim.df_patents_embeddings.loc[patent_index, 'novelty']) == 0
