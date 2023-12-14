# This test file contains unit tests for the inference script.

from src.inference import predict

def test_predict():
    # Assuming that the predict function returns a dictionary of class probabilities
    prediction = predict("Example text")

    # Check if the prediction is not None
    assert prediction is not None

    # Check if the prediction is a dictionary
    assert isinstance(prediction, dict)

    # Expected class names
    expected_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Check if all expected classes are in the prediction and their values are probabilities
    for class_name in expected_classes:
        assert class_name in prediction
        prob = prediction[class_name]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0