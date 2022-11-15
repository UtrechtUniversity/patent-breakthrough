from docembedder import FileIO
import random
import json
import numpy as np


def test_save_load_embeddings():
    file_name_expected = 'test_embedding_file.npz'
    train_id_expected = 100
    patent_id_expected = range(1, 200)
    embeddings_expected = [random.sample(range(101), 10), random.sample(range(101), 10), random.sample(range(101), 10)]
    preprocessing_setting_expected = json.dumps({'keep_caps': True, 'keep_start_section': True, 'remove_non_alpha': True})
    model_setting_expected = json.dumps({'embedder': 'tfidf', 'ngram_max': 1, 'stop_words': 'english'})
    version_expected = '0.0.1'

    f = FileIO()
    f.save_embeddings(file_name_expected, train_id_expected, patent_id_expected, embeddings_expected,
                      preprocessing_setting_expected, model_setting_expected, version_expected)

    loaded = f.load_embeddings(file_name_expected)
    assert np.array_equal(train_id_expected, loaded['train_id'])
    assert np.array_equal(patent_id_expected, loaded['patent_id'])
    assert np.array_equal(embeddings_expected, loaded['embeddings'])
    assert np.array_equal(preprocessing_setting_expected, loaded['preprocessing_setting'])
    assert np.array_equal(model_setting_expected, loaded['model_setting'])
    assert np.array_equal(version_expected, loaded['version'])

