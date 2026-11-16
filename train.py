"""
train.py
========
This script trains the intent classification model for our AI Chatbot.

Machine Learning Workflow:
1. Load patterns and intent tags from 'intents.json'.
2. Preprocess text using the pipeline from 'preprocess.py'.
3. Convert textual patterns into numerical features using TF-IDF with sublinear TF scaling.
4. Train a Logistic Regression classifier (C=5.0) on the TF-IDF feature matrix.
   - Using C=5.0 balances regularization so that distinct intent keywords produce
     confident probabilities (>= 0.5) while irrelevant queries remain low.
5. Save the trained model ('model.pkl') and vectorizer ('vectorizer.pkl') using pickle.
"""

import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import our custom preprocessing function from preprocess.py
from preprocess import preprocess


def load_dataset(file_path="intents.json"):
    """
    Loads intents, patterns, and tags from the JSON dataset file.
    
    Returns:
        sentences (list of str): The user query patterns.
        tags (list of str): The corresponding intent tags for each pattern.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' was not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    tags = []

    # Loop through each intent and collect its example patterns
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            sentences.append(pattern)
            tags.append(tag)

    return sentences, tags


def train_and_save_model(data_path="intents.json",
                         model_output="model.pkl",
                         vectorizer_output="vectorizer.pkl"):
    """
    Trains the TF-IDF + Logistic Regression pipeline and persists artifacts.
    """
    print("Step 1: Loading dataset from", data_path, "...")
    sentences, tags = load_dataset(data_path)
    print(f"Loaded {len(sentences)} training patterns across {len(set(tags))} unique intents.")

    # --------------------------------------------------------------------------
    # Step 2: Preprocess all sentences
    # WHY: Preprocessing strips noise, lowercases, and lemmatizes so the model
    # learns the core intent keywords rather than superficial variations.
    # --------------------------------------------------------------------------
    print("Step 2: Preprocessing training patterns with NLTK...")
    cleaned_sentences = [preprocess(s) for s in sentences]

    # --------------------------------------------------------------------------
    # Step 3: TF-IDF Vectorization
    # WHY TF-IDF: Machine learning algorithms cannot work with raw text.
    # We use sublinear_tf=True (1 + log(tf)) so high-frequency words do not dominate.
    # --------------------------------------------------------------------------
    print("Step 3: Converting text patterns to numerical features using TF-IDF...")
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X = vectorizer.fit_transform(cleaned_sentences)
    y = tags

    print(f"TF-IDF feature matrix shape: {X.shape} (patterns x vocabulary words)")

    # --------------------------------------------------------------------------
    # Step 4: Model Training with Logistic Regression
    # WHY C=5.0: With 9 intents, standard C=1.0 over-regularizes short query phrases.
    # Setting C=5.0 allows key words to establish confident probabilities (>50%).
    # --------------------------------------------------------------------------
    print("Step 4: Training Logistic Regression classifier (C=5.0)...")
    classifier = LogisticRegression(C=5.0, max_iter=300, random_state=42)
    classifier.fit(X, y)

    # --------------------------------------------------------------------------
    # Step 5: Save Model and Vectorizer to disk using pickle
    # --------------------------------------------------------------------------
    print("Step 5: Saving artifacts with pickle...")
    with open(model_output, "wb") as f_model:
        pickle.dump(classifier, f_model)

    with open(vectorizer_output, "wb") as f_vec:
        pickle.dump(vectorizer, f_vec)

    print(f"SUCCESS: Saved model to '{model_output}' and vectorizer to '{vectorizer_output}'.")


if __name__ == "__main__":
    train_and_save_model()
