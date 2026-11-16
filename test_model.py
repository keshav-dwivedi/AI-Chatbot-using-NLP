"""
test_model.py
=============
This script evaluates the chatbot intent classification model and demonstrates
the before-and-after improvement metrics required by Step 6 & Step 7.

Required Tasks Implemented:
1. Splits dataset into train and test sets (stratified by intent).
2. Prints accuracy score and scikit-learn classification report.
3. Tests 10 sample test queries with completely different phrasing (NOT copied
   from intents.json) to measure honest generalization.
4. Demonstrates Model Improvement: shows accuracy BEFORE vs AFTER improving patterns
   and tuning parameters, along with confidence analysis.
"""

import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Import preprocessing pipeline
from preprocess import preprocess


def run_full_evaluation():
    print("=" * 75)
    print("AI CHATBOT: MODEL EVALUATION & BEFORE/AFTER IMPROVEMENT ANALYSIS")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------------------------
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    all_sentences = []
    all_tags = []
    baseline_sentences = []
    baseline_tags = []

    for intent in data["intents"]:
        tag = intent["tag"]
        patterns = intent["patterns"]

        # First 10 patterns represent baseline
        for p in patterns[:10]:
            baseline_sentences.append(p)
            baseline_tags.append(tag)

        # All 12 patterns represent enriched dataset
        for p in patterns:
            all_sentences.append(p)
            all_tags.append(tag)

    # Preprocess
    cleaned_baseline = [preprocess(s) for s in baseline_sentences]
    cleaned_all = [preprocess(s) for s in all_sentences]

    # --------------------------------------------------------------------------
    # PART A: BASELINE MODEL EVALUATION (10 patterns per intent = 90 total)
    # --------------------------------------------------------------------------
    print("\n[PART 1: BASELINE MODEL (Before Improvement)]")
    print(f"Dataset Size: {len(baseline_sentences)} patterns across 9 intents")

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        cleaned_baseline, baseline_tags, test_size=0.20, random_state=42, stratify=baseline_tags
    )

    vec_base = TfidfVectorizer()
    X_train_b_vec = vec_base.fit_transform(X_train_b)
    X_test_b_vec = vec_base.transform(X_test_b)

    clf_base = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    clf_base.fit(X_train_b_vec, y_train_b)

    y_pred_b = clf_base.predict(X_test_b_vec)
    acc_before = accuracy_score(y_test_b, y_pred_b)
    print(f">> BASELINE TEST ACCURACY (Before): {acc_before * 100:.2f}%")

    # --------------------------------------------------------------------------
    # PART B: IMPROVED MODEL EVALUATION (12 patterns + sublinear TF-IDF)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("[PART 2: IMPROVED MODEL (After Pattern Expansion & Feature Tuning)]")
    print(f"Dataset Size: {len(all_sentences)} patterns across 9 intents (+18 new patterns)")

    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        cleaned_all, all_tags, test_size=0.20, random_state=42, stratify=all_tags
    )

    vec_improved = TfidfVectorizer(sublinear_tf=True)
    X_train_i_vec = vec_improved.fit_transform(X_train_i)
    X_test_i_vec = vec_improved.transform(X_test_i)

    clf_improved = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    clf_improved.fit(X_train_i_vec, y_train_i)

    y_pred_i = clf_improved.predict(X_test_i_vec)
    acc_after = accuracy_score(y_test_i, y_pred_i)
    print(f">> IMPROVED TEST ACCURACY (After) : {acc_after * 100:.2f}%")
    print(f">> NET ACCURACY IMPROVEMENT       : +{(acc_after - acc_before) * 100:.2f}%")

    print("\nCLASSIFICATION REPORT (IMPROVED MODEL):")
    print("-" * 75)
    print(classification_report(y_test_i, y_pred_i, zero_division=0))
    print("-" * 75)

    # --------------------------------------------------------------------------
    # PART C: 10 UNSEEN PARAPHRASED TEST QUERIES
    # (Completely rephrased queries that do not appear in intents.json)
    # --------------------------------------------------------------------------
    print("\n[PART 3: 10 UNSEEN PARAPHRASED TEST QUERIES]")
    print("(Testing honest generalization on diverse student queries)")
    print("-" * 75)

    unseen_test_queries = [
        ("Hey bot, good morning!", "greeting"),
        ("I have to leave now, catch you tomorrow.", "goodbye"),
        ("I really appreciate your quick guidance.", "thanks"),
        ("What is your identity, what should students call you?", "name"),
        ("I feel confused, could you guide me through this?", "help"),
        ("At what hour does the campus gates shut down?", "timings"),
        ("What bachelor engineering branches can I enroll in?", "courses"),
        ("What is the cost per semester for the CSE degree?", "fees"),
        ("Where is the main university campus situated?", "contact"),
        ("Can I talk to someone in the administration desk by phone?", "contact")
    ]

    correct_predictions = 0
    # In a 9-class system, random chance is 1/9 = 11.1%.
    # A confidence above 25% represents strong discriminative conviction (>2x random chance).
    for idx, (query, expected_intent) in enumerate(unseen_test_queries, 1):
        cleaned_q = preprocess(query)
        q_vec = vec_improved.transform([cleaned_q])
        probs = clf_improved.predict_proba(q_vec)[0]
        top_idx = np.argmax(probs)
        predicted_tag = clf_improved.classes_[top_idx]
        confidence = float(probs[top_idx])

        is_match = (predicted_tag == expected_intent)
        if is_match:
            correct_predictions += 1

        match_status = "CORRECT" if is_match else "INCORRECT"
        threshold_check = "Passes >=0.5" if confidence >= 0.5 else f"Confidence = {confidence*100:.1f}%"

        print(f"[{idx:02d}] Query    : \"{query}\"")
        print(f"     Expected : {expected_intent}")
        print(f"     Predicted: {predicted_tag} ({confidence * 100:.1f}%) -> [{match_status}] ({threshold_check})")
        print("-" * 75)

    print(f"\nUnseen Generalization Score: {correct_predictions}/{len(unseen_test_queries)} "
          f"({correct_predictions / len(unseen_test_queries) * 100:.1f}%)")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # SUMMARY OF IMPROVEMENTS
    # --------------------------------------------------------------------------
    print("\nSUMMARY OF MODEL IMPROVEMENTS (Step 7):")
    print("1. Pattern Expansion: Added 2 diverse patterns per intent (from 10 to 12 each).")
    print("2. Sublinear TF Scaling: Replaced raw term counts with 1 + log(tf) to prevent dominant words from overpowering.")
    print(f"3. Accuracy Before : {acc_before * 100:.2f}%")
    print(f"4. Accuracy After  : {acc_after * 100:.2f}% (Gain: +{(acc_after - acc_before) * 100:.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    run_full_evaluation()
