# This script contains the code to perform inference using the trained model.

import sys
import joblib
from .clean_text import clean_text

# Load the trained model
model = joblib.load('models/toxic_comment_classifier.pkl')

def predict(text):
    # Clean and preprocess the text
    cleaned_text = clean_text(text)

    # Load the TfidfVectorizer from the trained model
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    # Transform the cleaned text using the loaded vectorizer
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    
    # Make a prediction using the vectorized text
    prediction = model.predict_proba(vectorized_text)

    # Class names as used during model training
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Pairing class names with their predicted probabilities
    class_probabilities = dict(zip(class_names, prediction[0]))

    return class_probabilities

if __name__ == "__main__":
    input_text = sys.argv[1]
    predictions = predict(input_text)
    for class_name, probability in predictions.items():
        print(f"{class_name}: {probability}")