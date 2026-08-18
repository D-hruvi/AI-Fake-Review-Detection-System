# Fake Review Detector — Deployable App

Flask app serving the TF-IDF + XGBoost model (trained in `scripts/04_retrain_augmented.py`)
with per-prediction SHAP explanations, shown as inline highlighted words in the UI.

## Run locally

```bash
cd deploy
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

## Deploy on Render

1. Push this `deploy/` folder to a GitHub repo (or push the whole project and set Render's
   root directory to `deploy/`).
2. On Render: **New → Web Service** → connect the repo.
3. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 60`
     (already set in `Procfile`, Render should pick it up automatically)
   - **Instance type**: the free/starter tier is fine — model is small (~600KB pickled),
     no GPU needed, matches the CPU-only constraint from the research phase.
4. No environment variables required for the base app (the Groq generation script is
   separate and NOT part of this deployed app — it's a data-prep tool, not a runtime
   dependency).
5. Deploy. First request may be slow (cold start + XGBoost/SHAP import), subsequent
   requests are fast (SHAP TreeExplainer is efficient for tree models).

## What's included

- `app.py` — Flask backend: `/predict` (POST, JSON `{"text": "..."}`) and `/health`
- `templates/index.html` — single-page frontend, no build step needed
- `model_files/augmented_model.pkl`, `model_files/augmented_vectorizer.pkl` — the trained
  model and TF-IDF vectorizer from the research phase, pickled

## Known limitation surfaced in the UI

The augmented model's improvement on LLM-style fakes was measured against a single LLM
author's writing (see the main project's `RESEARCH_SUMMARY.md`, Experiment 2). The app's
UI includes this caveat directly so users don't over-trust the "fake" verdict on text
written by a different LLM or a different prompting style than what the model saw in
training. If you generate a larger, multi-generator training set later (e.g. via
`scripts/00_generate_with_groq.py` in the parent project), retrain and swap the two
`.pkl` files in `model_files/` — the app code doesn't need to change.

## API example

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "We stayed at this hotel for three nights and the room was clean and comfortable. Staff were friendly and the location was convenient for downtown."}'
```
