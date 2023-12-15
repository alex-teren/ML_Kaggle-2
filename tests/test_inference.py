# This test file contains unit tests for the inference script.
import os
from src.inference import predict, download_file


def test_predict_output_structure():
    text = "example text"
    prediction = predict(text)
    assert isinstance(prediction, dict)

def test_predict_output_classes():
    text = "example text"
    prediction = predict(text)
    expected_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for class_name in expected_classes:
        assert class_name in prediction

def test_predict_output_probabilities_range():
    text = "example text"
    prediction = predict(text)
    for prob in prediction.values():
        assert 0.0 <= prob <= 1.0

def test_download_file_success():
    url = 'https://github.com/alex-teren/ML_Kaggle-2/raw/main/models/toxic_comment_classifier.pkl' 
    local_filename = 'testfile.pkl'
    downloaded_file = download_file(url, local_filename)
    assert downloaded_file is not None
    # Clean up: remove the downloaded file if it exists
    if os.path.exists(local_filename):
        os.remove(local_filename)

def test_download_file_failure():
    url = 'https://example.com/nonexistentfile.pkl'
    local_filename = 'testfile.pkl'
    downloaded_file = download_file(url, local_filename)
    assert downloaded_file is None
