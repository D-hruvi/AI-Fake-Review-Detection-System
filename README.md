# Verdict — AI Fake Review Detection

A deployable web app + browser extension that flags likely AI/template-generated
product reviews.

## What it actually does (read this before demoing it)

The model is trained on the Salminen et al. (2022) dataset: 40,432 Amazon-style
reviews, half real human reviews, half GPT-2-generated fakes. That means:

- **It detects generated/templated text**, not "reviews that sound exaggerated."
  A genuine review full of exclamation marks can still be scored genuine, and a
  flat, generic-sounding real review can get flagged. This is the standard
  definition used in fake-review research — but it's worth knowing before you
  present it as a general lie detector.
- **91.4% F1 on held-out test data.** Not 100%. Say this number, not "highly
  accurate," when you present it.
- **Server-side URL scraping (`/api/scrape`) will frequently fail on Amazon**
  because Amazon actively blocks non-browser requests (403s, CAPTCHAs, JS-
  rendered content). The app tells the user this honestly instead of pretending
  to have scraped when it hasn't. The browser extension is the real fix for
  this — see below.

## Why Logistic Regression, not Random Forest/XGBoost

I trained all three on the same TF-IDF features. Logistic Regression won on
every axis:

| Model | F1 | Train time |
|---|---|---|
| Logistic Regression | **91.4%** | 5s |
| XGBoost | 88.9% | 91s |
| Random Forest | 88.7% | 67s |

This is expected, not a fluke: linear models generally beat tree ensembles on
sparse, high-dimensional TF-IDF text features. Tree splits don't exploit sparse
one-hot-ish word features well, and RF/XGBoost need far more data per feature
to find good splits across 15,000+ dimensions. Logistic Regression is also
~15x smaller and gives clean, signed per-word coefficients — which is what
powers the "evidence words" explainability in the UI. Retrain with
`backend/model/train_model.py` if you want to verify this yourself or try a
different feature set.

## Project structure

```
backend/
  app.py                 Flask app: /api/analyze, /api/scrape, /api/stats
  requirements.txt
  Procfile                gunicorn start command (Render/Heroku-style)
  model/
    train_model.py        retrain from scratch
    train_reviews.csv     training data (not needed at runtime, kept for reproducibility)
    tfidf_vectorizer.pkl  fitted vectorizer (runtime)
    classifier.pkl         Logistic Regression model (runtime)
    lr_explainer.pkl       same model, used for word-level explainability
    model_meta.pkl          accuracy/f1 metadata shown in the dashboard
  static/
    index.html, style.css, app.js   frontend (no build step — plain HTML/JS)
browser-extension/
  manifest.json, content.js, content.css, popup.html, popup.js
render.yaml               one-click Render deploy config
```

## Run locally

```bash
cd backend
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

## Deploy to Render (matches your existing deployment setup)

1. Push this repo to GitHub.
2. Render dashboard → New → Blueprint → point at the repo (it will read
   `render.yaml` automatically), or manually create a Web Service with:
   - Root directory: `backend`
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
3. Free tier spins down on inactivity — first request after idle takes ~30-50s.
4. `history.db` (SQLite) resets on every redeploy on Render's free tier since
   the filesystem is ephemeral. Fine for a demo/portfolio piece; if you want
   the dashboard history to survive redeploys, swap it for Render's free
   Postgres instead of SQLite.

## Browser extension (the real fix for Amazon/Flipkart scraping)

Server-side scraping gets blocked by Amazon/Flipkart's bot detection. A browser
extension doesn't have that problem — it reads the DOM *after your browser has
already rendered the page*, using your real session.

1. Deploy the backend first and copy its URL.
2. `chrome://extensions` → enable Developer Mode → Load unpacked →
   select the `browser-extension/` folder.
3. Click the extension icon → paste your deployed API URL → Save.
4. Visit an Amazon or Flipkart product page — reviews get an inline
   `LIKELY FAKE` / `LIKELY GENUINE` badge as they load.

Note: Amazon/Flipkart's DOM structure changes over time and differs by region;
the CSS selectors in `content.js` (`getSelectors()`) may need small updates if
badges stop appearing.

## API

- `POST /api/analyze` `{text}` → `{label, confidence, fake_probability, signals, word_count}`
- `POST /api/scrape` `{url}` → aggregated results for reviews found on the page, or an honest error
- `GET /api/stats` → dashboard aggregates (totals, daily trend, by source, model metadata)

## Retraining

```bash
cd backend/model
python train_model.py
```
Trains Logistic Regression, Random Forest, and XGBoost, prints a comparison
table, and saves whichever wins on F1.
