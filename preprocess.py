"""
preprocess.py
=============
This module handles all text preprocessing for our AI Chatbot using NLTK.

NLP Preprocessing Pipeline:
1. Lowercase: Converts text to lowercase so words like 'College' and 'college' match.
2. Punctuation Removal: Removes punctuation marks (!, ?, .) that do not contribute to intent.
3. Tokenization: Splits a sentence into individual words (tokens).
4. Stopword Removal: Removes generic grammatical words (is, the, at) while keeping
   intent-bearing words (who, what, when, how).
5. Lemmatization: Reduces words to their base dictionary form (e.g. 'courses' -> 'course').
"""

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------------------
# Initialize lemmatizer and stop words once so they stay in memory for fast reuse
# ------------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()

# Standard English stopwords from NLTK
# WHY WE KEEP QUESTION WORDS: In a chatbot, question words like 'who', 'what', 'when',
# 'where', and 'how' distinguish user intents (e.g. "Who are you?" vs "What are the timings?").
# Therefore, we remove general filler words but preserve question words.
QUESTION_WORDS = {"who", "what", "where", "when", "how", "why"}
DEFAULT_STOPWORDS = set(stopwords.words("english"))
FILTERED_STOPWORDS = DEFAULT_STOPWORDS - QUESTION_WORDS


def clean_text(text):
    """
    Takes raw user input string and returns a list of cleaned, lemmatized tokens.
    
    Parameters:
        text (str): The raw text sentence from the user or dataset.
        
    Returns:
        list of str: Cleaned and normalized word tokens.
    """
    # Defensive check: return empty list if input is empty or invalid
    if not text or not isinstance(text, str):
        return []

    # --------------------------------------------------------------------------
    # Step 1: Lowercase
    # WHY: Python is case-sensitive. Lowercasing ensures 'Hello', 'HELLO',
    # and 'hello' are treated as the exact same word.
    # --------------------------------------------------------------------------
    text = text.lower()

    # --------------------------------------------------------------------------
    # Step 2: Remove Punctuation
    # WHY: Punctuation marks like '?', '!', and ',' carry no semantic intent
    # and would otherwise prevent words like 'hours?' from matching 'hours'.
    # --------------------------------------------------------------------------
    text = text.translate(str.maketrans("", "", string.punctuation))

    # --------------------------------------------------------------------------
    # Step 3: Tokenize
    # WHY: A machine cannot understand a whole sentence at once. Tokenizing splits
    # the sentence into a list of separate word units (tokens) for analysis.
    # --------------------------------------------------------------------------
    tokens = word_tokenize(text)

    # --------------------------------------------------------------------------
    # Step 4: Remove Stopwords & Step 5: Lemmatize
    # WHY Stopwords: High-frequency words like 'is', 'an', 'at' add noise and
    # do not help distinguish whether the user asks about fees or courses.
    # WHY Lemmatization: Reduces words to their base lemma (e.g., 'timings' -> 'timing',
    # 'studies' -> 'study'). This ensures variations of a word are recognized as one.
    # --------------------------------------------------------------------------
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in FILTERED_STOPWORDS and token.strip()
    ]

    # Fallback: if all words were filtered out, keep the lemmatized tokens
    # so we don't end up with completely empty feature vectors.
    if not cleaned_tokens and tokens:
        cleaned_tokens = [lemmatizer.lemmatize(t) for t in tokens if t.strip()]

    return cleaned_tokens


def preprocess(text):
    """
    Convenience function that takes a text string and returns a single cleaned
    string (space-separated tokens), ready for scikit-learn's TF-IDF vectorizer.
    
    Parameters:
        text (str): The raw text sentence.
        
    Returns:
        str: Space-separated cleaned and lemmatized words.
    """
    tokens = clean_text(text)
    return " ".join(tokens)


# ------------------------------------------------------------------------------
# Demonstration / Self-Test Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    sample_queries = [
        "Hello! Is anyone available to help me?",
        "What are the college hours and library timings?",
        "How much is the annual tuition fee for B.Tech?",
        "Who are you and what can you do?"
    ]

    print("=" * 60)
    print("DEMO: Testing preprocess.py with sample college queries")
    print("=" * 60)
    for query in sample_queries:
        print(f"Original  : {query}")
        print(f"Tokens    : {clean_text(query)}")
        print(f"Processed : '{preprocess(query)}'")
        print("-" * 60)
