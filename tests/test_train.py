# This test file contains unit tests for the model train script

import os
import json
import pandas as pd


def test_configuration_loading():
    with open('configs/model_config.json') as config_file:
        config = json.load(config_file)

    # Check for the existence of specific configuration fields
    assert 'max_features' in config
    # Assuming max_features should be an integer
    assert isinstance(config['max_features'], int)

    assert 'stop_words' in config
    # stop_words can be a string or None
    assert isinstance(config['stop_words'],
                      str) or config['stop_words'] is None

    assert 'test_size' in config
    # test_size is typically a float
    assert isinstance(config['test_size'], float)
    # test_size should be between 0 and 1
    assert 0.0 < config['test_size'] < 1.0

    assert 'random_state' in config
    # random_state is typically an integer
    assert isinstance(config['random_state'], int)

    assert 'solver' in config
    assert isinstance(config['solver'], str)  # solver is a string

    assert 'C' in config
    assert isinstance(config['C'], float)  # C is typically a float


def test_data_loading():
    train_data = pd.read_csv('./src/data/train.csv')
    assert not train_data.empty


def test_model_saving():
    assert os.path.exists('./models/toxic_comment_classifier.pkl')
    assert os.path.exists('./models/tfidf_vectorizer.pkl')
