# This script contains the text cleaning function used in the preprocessing step.
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import sys

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove special characters and punctuation
    text = re.sub(r'\\W+', ' ', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
        cleaned_text = clean_text(input_text)
        print(cleaned_text)
    else:
        print("Please provide text to clean as a command-line argument.")