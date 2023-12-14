# This script contains the code to perform inference using the trained model.

import requests
import sys
import joblib
from .clean_text import clean_text


def download_file(url, local_filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print(f"File '{local_filename}' downloaded successfully.")
        return local_filename
    else:
        print(f"Error in file downloading: {response.status_code}")
        return None


# Download the trained model and vectorizer
model_file = download_file(
    'https://github.com/alex-teren/ML_Kaggle-2/raw/main/models/toxic_comment_classifier.pkl', 'toxic_comment_classifier.pkl')
model = joblib.load(model_file)

# Load the TfidfVectorizer from the trained model
tfidf_vectorizer_file = download_file(
    'https://github.com/alex-teren/ML_Kaggle-2/raw/main/models/tfidf_vectorizer.pkl', 'tfidf_vectorizer.pkl')
tfidf_vectorizer = joblib.load(tfidf_vectorizer_file)


def predict(text):
    # Clean and preprocess the text
    cleaned_text = clean_text(text)

    # Transform the cleaned text using the loaded vectorizer
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])

    # Make a prediction using the vectorized text
    prediction = model.predict_proba(vectorized_text)

    # Class names as used during model training
    class_names = ['toxic', 'severe_toxic', 'obscene',
                   'threat', 'insult', 'identity_hate']

    # Pairing class names with their predicted probabilities
    class_probabilities = dict(zip(class_names, prediction[0]))

    return class_probabilities


if __name__ == "__main__":
    input_text = sys.argv[1]
    predictions = predict(input_text)
    for class_name, probability in predictions.items():
        print(f"{class_name}: {probability}")
