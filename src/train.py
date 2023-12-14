# This script contains the code to train the model based on the configuration provided.

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import os
import joblib
from clean_text import clean_text


# Load configuration
with open('configs/model_config.json') as config_file:
    config = json.load(config_file)

# Load data
train_data = pd.read_csv('./src/data/train.csv')

# Preprocess and vectorize data
train_data['comment_text'] = train_data['comment_text'].apply(clean_text)
tfidf_vectorizer = TfidfVectorizer(max_features=config['max_features'], stop_words=config['stop_words'])
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['comment_text'])

# Train the model
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_full = train_data[labels]
X_train, X_val, y_train_split, y_val = train_test_split(X_train_tfidf, y_full, test_size=config['test_size'], random_state=config['random_state'])
model = OneVsRestClassifier(LogisticRegression(solver=config['solver'], C=config['C']))
model.fit(X_train, y_train_split)

# Create models directory if it doesn't exist
models_dir = './models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save the model and the vectorizer
joblib.dump(model, os.path.join(models_dir, 'toxic_comment_classifier.pkl'))
joblib.dump(tfidf_vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
