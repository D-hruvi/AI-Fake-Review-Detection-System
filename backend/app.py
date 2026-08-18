"""
Fake Review Detector — deployable Flask app.

Serves the TF-IDF + XGBoost model trained in scripts/04_retrain_augmented.py
(real + traditional_fake + LLM_fake augmented training set).

IMPORTANT caveat surfaced to the end user in the UI, not just buried in a README:
this model's improvement on LLM-style fakes was measured against fakes written by
a single LLM author with limited prompt diversity. It may overfit to that author's
phrasing rather than generalizing to all LLM-written fakes. Treat predictions as a
signal, not a verdict.
"""
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import shap

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")

with open(os.path.join(MODEL_DIR, "augmented_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "augmented_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

feature_names = np.array(vectorizer.get_feature_names_out())
explainer = shap.TreeExplainer(model)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide review text."}), 400
    if len(text.split()) < 8:
        return jsonify({"error": "Text is very short — predictions are unreliable under ~8 words."}), 400

    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    pred_label = int(np.argmax(proba))
    confidence = float(proba[pred_label])

    # Per-prediction SHAP explanation
    shap_vals = explainer.shap_values(X)
    row = np.asarray(shap_vals).reshape(-1)
    nonzero_idx = np.where(X.toarray()[0] != 0)[0]

    contributions = sorted(
        [(feature_names[i], float(row[i])) for i in nonzero_idx],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:8]

    top_fake_signals = [{"word": w, "shap": round(v, 4)} for w, v in contributions if v > 0]
    top_real_signals = [{"word": w, "shap": round(v, 4)} for w, v in contributions if v < 0]

    return jsonify({
        "prediction": "fake" if pred_label == 1 else "real",
        "confidence": round(confidence, 3),
        "top_fake_signals": top_fake_signals,
        "top_real_signals": top_real_signals,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
