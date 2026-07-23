"""
Trains the fake-review classifier.

Dataset: Salminen et al. (2022) "Creating and Detecting Fake Reviews of Online
Products" - 40,432 Amazon-style reviews, balanced CG (computer-generated/fake)
vs OR (original/genuine).

Run:
    python train_model.py

Produces model/tfidf_vectorizer.pkl and model/classifier.pkl
"""
import re
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH = "train_reviews.csv"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Hand-crafted heuristic features. These are cheap signals that correlate
# with generated/fake review text and, unlike TF-IDF, they also work well
# for the human-readable "trust signals" shown in the UI later.
# ---------------------------------------------------------------------------
def extract_heuristics(texts):
    rows = []
    for t in texts:
        t = str(t)
        n_chars = max(len(t), 1)
        n_words = max(len(t.split()), 1)
        n_exclaim = t.count("!")
        n_caps_words = sum(1 for w in t.split() if len(w) > 1 and w.isupper())
        avg_word_len = np.mean([len(w) for w in t.split()]) if t.split() else 0
        n_superlatives = len(re.findall(
            r"\b(amazing|perfect|best|love|awesome|excellent|incredible|highly recommend)\b",
            t.lower()))
        first_person = len(re.findall(r"\b(i|my|me|we)\b", t.lower()))
        has_numbers = len(re.findall(r"\d", t))
        rows.append([
            n_chars, n_words, n_exclaim / n_words, n_caps_words / n_words,
            avg_word_len, n_superlatives / n_words, first_person / n_words,
            has_numbers,
        ])
    cols = ["n_chars", "n_words", "exclaim_ratio", "caps_ratio", "avg_word_len",
            "superlative_ratio", "first_person_ratio", "has_numbers"]
    return pd.DataFrame(rows, columns=cols)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text_", "label"])
    df["y"] = (df["label"] == "CG").astype(int)  # 1 = fake/generated, 0 = genuine

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text_"], df["y"], test_size=0.15, random_state=RANDOM_STATE, stratify=df["y"])

    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                             min_df=3, sublinear_tf=True, stop_words="english")
    Xtr_tfidf = tfidf.fit_transform(X_train_text)
    Xte_tfidf = tfidf.transform(X_test_text)

    print("Extracting heuristic features...")
    Xtr_heur = extract_heuristics(X_train_text).values
    Xte_heur = extract_heuristics(X_test_text).values

    Xtr = hstack([Xtr_tfidf, csr_matrix(Xtr_heur)]).tocsr()
    Xte = hstack([Xte_tfidf, csr_matrix(Xte_heur)]).tocsr()

    results = {}

    print("\nTraining Logistic Regression (interpretability baseline)...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=5, random_state=RANDOM_STATE)
    lr.fit(Xtr, y_train)
    pred = lr.predict(Xte)
    results["logistic_regression"] = {
        "model": lr, "acc": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred), "time": time.time() - t0,
    }

    print("Training Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1,
                                 random_state=RANDOM_STATE, min_samples_leaf=2)
    rf.fit(Xtr, y_train)
    pred = rf.predict(Xte)
    results["random_forest"] = {
        "model": rf, "acc": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred), "time": time.time() - t0,
    }

    if HAS_XGB:
        print("Training XGBoost...")
        t0 = time.time()
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                             n_jobs=-1, random_state=RANDOM_STATE,
                             eval_metric="logloss")
        xgb.fit(Xtr, y_train)
        pred = xgb.predict(Xte)
        results["xgboost"] = {
            "model": xgb, "acc": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred), "time": time.time() - t0,
        }

    print("\n=== Results ===")
    for name, r in results.items():
        print(f"{name:20s}  acc={r['acc']:.4f}  f1={r['f1']:.4f}  train_time={r['time']:.1f}s")

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (f1={results[best_name]['f1']:.4f}) -- saving this one.")
    print("\nDetailed report for best model:")
    print(classification_report(y_test, best_model.predict(Xte), target_names=["genuine", "fake"]))

    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(best_model, "classifier.pkl")
    joblib.dump({"name": best_name, "acc": results[best_name]["acc"],
                 "f1": results[best_name]["f1"]}, "model_meta.pkl")

    # Also keep logistic regression around specifically for word-level
    # explainability in the UI (RF/XGB importances aren't per-review signed).
    joblib.dump(lr, "lr_explainer.pkl")
    print("\nSaved: tfidf_vectorizer.pkl, classifier.pkl, lr_explainer.pkl, model_meta.pkl")


if __name__ == "__main__":
    main()
