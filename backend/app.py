import os
import re
import sqlite3
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_from_directory
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "history.db")

app = Flask(__name__, static_folder="static", static_url_path="")

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
clf = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
lr_explainer = joblib.load(os.path.join(MODEL_DIR, "lr_explainer.pkl"))
meta = joblib.load(os.path.join(MODEL_DIR, "model_meta.pkl"))
FEATURE_NAMES = np.array(tfidf.get_feature_names_out())
LR_COEF = lr_explainer.coef_[0][: len(FEATURE_NAMES)]  # word part of the coef vector


def extract_heuristics_single(t):
    t = str(t)
    words = t.split()
    n_words = max(len(words), 1)
    n_exclaim = t.count("!")
    n_caps_words = sum(1 for w in words if len(w) > 1 and w.isupper())
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    n_superlatives = len(re.findall(
        r"\b(amazing|perfect|best|love|awesome|excellent|incredible|highly recommend)\b",
        t.lower()))
    first_person = len(re.findall(r"\b(i|my|me|we)\b", t.lower()))
    has_numbers = len(re.findall(r"\d", t))
    return np.array([[len(t), n_words, n_exclaim / n_words, n_caps_words / n_words,
                       avg_word_len, n_superlatives / n_words, first_person / n_words,
                       has_numbers]])


def predict(text):
    """Returns (label, confidence 0-100, top_signal_words[list of (word, direction)])"""
    text = (text or "").strip()
    if not text:
        return None
    vec_tfidf = tfidf.transform([text])
    vec_heur = extract_heuristics_single(text)
    vec = hstack([vec_tfidf, csr_matrix(vec_heur)]).tocsr()

    proba = clf.predict_proba(vec)[0]
    fake_prob = float(proba[1])
    label = "fake" if fake_prob >= 0.5 else "genuine"
    confidence = round((fake_prob if label == "fake" else 1 - fake_prob) * 100, 1)

    # Word-level explainability from the logistic-regression coefficients:
    # for each word present in the review, coef > 0 pushes toward "fake".
    nz = vec_tfidf.nonzero()[1]
    contributions = [(FEATURE_NAMES[i], vec_tfidf[0, i] * LR_COEF[i]) for i in nz]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:6]
    signals = [{"word": w, "pushes": "fake" if c > 0 else "genuine"} for w, c in top]

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": round(fake_prob * 100, 1),
        "signals": signals,
        "word_count": len(text.split()),
    }


# ---------------------------------------------------------------------------
# History DB (powers the dashboard)
# ---------------------------------------------------------------------------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, source TEXT, label TEXT, confidence REAL, word_count INTEGER
    )""")
    return conn


def log_result(source, result):
    conn = db()
    conn.execute("INSERT INTO history (ts, source, label, confidence, word_count) VALUES (?,?,?,?,?)",
                 (datetime.now(timezone.utc).isoformat(), source, result["label"],
                  result["confidence"], result["word_count"]))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Review text is empty."}), 400
    result = predict(text)
    log_result("manual", result)
    return jsonify(result)


@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400

    domain = "amazon" if "amazon." in url else ("flipkart" if "flipkart." in url else "other")
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as e:
        return jsonify({"error": f"Could not reach that URL: {e}"}), 502

    if resp.status_code != 200:
        return jsonify({
            "error": f"Site returned status {resp.status_code}. Amazon/Flipkart "
                     "actively block server-side scraping (CAPTCHA/bot detection), "
                     "so this often fails even for a valid URL. Paste review text "
                     "directly into the single-review checker as a reliable fallback."
        }), 502

    soup = BeautifulSoup(resp.text, "html.parser")
    review_texts = []
    if domain == "amazon":
        for el in soup.select("span[data-hook='review-body']"):
            review_texts.append(el.get_text(strip=True))
    elif domain == "flipkart":
        for el in soup.select("div._27M-vq, div.t-ZTKy"):
            review_texts.append(el.get_text(strip=True))
    else:
        for el in soup.select("p, span"):
            txt = el.get_text(strip=True)
            if 40 < len(txt) < 1000:
                review_texts.append(txt)

    review_texts = [t for t in review_texts if len(t.split()) >= 4][:25]

    if not review_texts:
        return jsonify({
            "error": "No reviews could be extracted. Amazon/Flipkart render reviews "
                     "with JavaScript and/or block non-browser requests, so this page's "
                     "HTML likely didn't contain plain review text. Paste review text "
                     "directly into the single-review checker instead — that always works."
        }), 422

    results = []
    for t in review_texts:
        r = predict(t)
        r["text"] = t[:200]
        results.append(r)
        log_result(domain, r)

    fake_count = sum(1 for r in results if r["label"] == "fake")
    return jsonify({
        "domain": domain,
        "total": len(results),
        "fake_count": fake_count,
        "genuine_count": len(results) - fake_count,
        "fake_pct": round(100 * fake_count / len(results), 1),
        "results": results,
    })


@app.route("/api/stats")
def api_stats():
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), SUM(label='fake') FROM history")
    total, fake = cur.fetchone()
    total = total or 0
    fake = fake or 0

    cur.execute("""SELECT substr(ts,1,10) as day, COUNT(*), SUM(label='fake')
                   FROM history GROUP BY day ORDER BY day DESC LIMIT 14""")
    by_day = [{"day": r[0], "total": r[1], "fake": r[2]} for r in cur.fetchall()]

    cur.execute("SELECT source, COUNT(*), SUM(label='fake') FROM history GROUP BY source")
    by_source = [{"source": r[0], "total": r[1], "fake": r[2]} for r in cur.fetchall()]

    cur.execute("SELECT AVG(confidence) FROM history")
    avg_conf = cur.fetchone()[0]
    conn.close()

    return jsonify({
        "total_analyzed": total,
        "fake_count": fake,
        "genuine_count": total - fake,
        "fake_pct": round(100 * fake / total, 1) if total else 0,
        "avg_confidence": round(avg_conf, 1) if avg_conf else 0,
        "by_day": list(reversed(by_day)),
        "by_source": by_source,
        "model": {"name": meta["name"], "accuracy": round(meta["acc"] * 100, 1),
                   "f1": round(meta["f1"] * 100, 1)},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
