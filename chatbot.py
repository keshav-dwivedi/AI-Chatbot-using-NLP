"""
chatbot.py
==========
This module handles intent recognition and response generation.

Workflow:
1. Loads trained artifacts ('model.pkl', 'vectorizer.pkl') and knowledge ('intents.json').
2. Preprocesses incoming user messages using 'preprocess.py'.
3. Computes class probabilities using Logistic Regression.
4. Returns a random response from the predicted intent.
5. Employs a confidence threshold (0.5): if the model's confidence is below 0.5,
   it returns a helpful fallback response asking the user to rephrase.
"""

import os
import json
import random
import pickle
import numpy as np

# Import preprocessing logic
from preprocess import preprocess

# Configuration constants
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
INTENTS_FILE = "intents.json"
CONFIDENCE_THRESHOLD = 0.5

FALLBACK_RESPONSES = [
    "Sorry, I didn't understand. Can you rephrase?",
    "I am not sure I understand that. Could you ask in a different way?",
    "I couldn't match your query to our college helpdesk topics. Can you please rephrase?"
]


class CollegeChatbot:
    """
    Core chatbot engine that loads models, classifies user intent,
    and returns appropriate college helpdesk responses.
    """

    def __init__(self, model_path=MODEL_FILE, vectorizer_path=VECTORIZER_FILE, intents_path=INTENTS_FILE):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.intents_path = intents_path
        self.model = None
        self.vectorizer = None
        self.responses = {}
        self.is_loaded = False

        self.load_resources()

    def load_resources(self):
        """
        Loads saved model, vectorizer, and intents data with friendly error handling.
        """
        # Basic error handling: check if model artifacts exist
        missing_files = []
        for path in [self.model_path, self.vectorizer_path]:
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            raise FileNotFoundError(
                f"Missing required model artifact(s): {', '.join(missing_files)}.\n"
                f"Please run 'python train.py' first to train and save the model."
            )

        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(
                f"Missing intents dataset at '{self.intents_path}'. "
                f"Make sure intents.json is in the project folder."
            )

        # Load trained classifier and TF-IDF vectorizer
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        # Load intents and map tags to responses for quick lookup
        with open(self.intents_path, "r", encoding="utf-8") as f:
            intents_data = json.load(f)

        self.responses = {
            item["tag"]: item["responses"]
            for item in intents_data["intents"]
        }
        self.is_loaded = True

    def predict_intent(self, text, threshold=CONFIDENCE_THRESHOLD):
        """
        Predicts the intent tag and confidence probability for a given text.
        
        Returns:
            intent (str): Predicted intent tag or 'unknown' if confidence < threshold.
            confidence (float): Probability score of the top prediction (0.0 to 1.0).
        """
        # Basic error handling: check for empty or non-string message
        if not text or not isinstance(text, str) or not text.strip():
            return "empty", 0.0

        # Step 1: Preprocess user input
        cleaned_text = preprocess(text)

        # If preprocessing leaves nothing (e.g. only punctuation entered)
        if not cleaned_text.strip():
            return "unknown", 0.0

        # Step 2: Transform into TF-IDF vector
        X_input = self.vectorizer.transform([cleaned_text])

        # Step 3: Compute probabilities across all intents
        probabilities = self.model.predict_proba(X_input)[0]
        max_idx = np.argmax(probabilities)
        max_confidence = float(probabilities[max_idx])
        predicted_tag = self.model.classes_[max_idx]

        # Step 4: Confidence threshold check (default 0.5)
        # WHY: If confidence is below 0.5, returning a fallback response
        # prevents the chatbot from giving misleading or incorrect answers.
        if max_confidence < threshold:
            return "unknown", max_confidence

        return predicted_tag, max_confidence

    def get_response(self, text):
        """
        Generates an end-to-end response for a given user message.
        
        Returns:
            reply (str): A suitable response text.
        """
        # Handle empty input
        if not text or not text.strip():
            return "Please type a question so I can assist you."

        # Predict intent
        intent, confidence = self.predict_intent(text)

        # Handle fallback if intent could not be identified with confidence
        if intent in ("unknown", "empty"):
            return random.choice(FALLBACK_RESPONSES)

        # Pick a random response from the matching intent
        possible_responses = self.responses.get(intent)
        if possible_responses:
            return random.choice(possible_responses)

        return random.choice(FALLBACK_RESPONSES)


# ------------------------------------------------------------------------------
# Interactive CLI Test Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        bot = CollegeChatbot()
        print("=" * 65)
        print("College Helpdesk Chatbot is Ready!")
        print("Confidence Threshold:", CONFIDENCE_THRESHOLD)
        print("=" * 65)

        test_queries = [
            "Hi there!",
            "When is the college open?",
            "What courses do you offer in engineering?",
            "How much are the tuition fees?",
            "What is your contact phone number?",
            "Who are you?",
            "What is the weather in Antarctica right now?"  # Out of domain query
        ]

        for query in test_queries:
            tag, conf = bot.predict_intent(query)
            reply = bot.get_response(query)
            status = "CONFIDENT" if conf >= CONFIDENCE_THRESHOLD else "FALLBACK (< 0.5)"
            print(f"User    : {query}")
            print(f"Intent  : {tag} ({conf * 100:.1f}%) -> {status}")
            print(f"Bot     : {reply}")
            print("-" * 65)

    except Exception as e:
        print(f"Error initializing chatbot: {e}")
