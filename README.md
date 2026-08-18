# 📩 SMS Spam Classifier — Streamlit App

A ready-to-run web UI for the model built in `SMS_Spam_Classification_NLP.ipynb`
(TF-IDF vectorizer + Multinomial Naive Bayes).

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web UI |
| `train_model.py` | Recreates the notebook's preprocessing + training pipeline, saves `spam_model.pkl` and `metrics.json` |
| `SMSSpamCollection` | Training dataset (tab-separated, label + message) |
| `spam_model.pkl` | Pre-trained model (already trained for you — 96.3% test accuracy) |
| `metrics.json` | Saved accuracy/confusion-matrix/report, shown in the app sidebar |
| `requirements.txt` | Python dependencies |

## Run it on your own computer

Requires Python 3.9+.

```bash
# 1. Go into the folder
cd sms_spam_app

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) retrain the model — a trained one is already included
python train_model.py

# 5. Launch the app
streamlit run app.py
```

Streamlit will open the app in your browser at `http://localhost:8501`.

## Get a public shareable link (no server management needed)

The easiest free option is **Streamlit Community Cloud**:

1. Create a free GitHub account if you don't have one, and a new repository.
2. Upload every file in this folder to that repository (`app.py`, `train_model.py`,
   `SMSSpamCollection`, `spam_model.pkl`, `metrics.json`, `requirements.txt`).
3. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
4. Click **"New app"**, pick your repository/branch, set the main file to `app.py`,
   and click **Deploy**.
5. In a minute or two you'll get a public URL like
   `https://your-app-name.streamlit.app` that you (or anyone) can open from any
   device, no installation required.

Other options that also give you a public link:
- **Hugging Face Spaces** (choose the "Streamlit" SDK) — similarly free and simple.
- **Render** or **Railway** — deploy `streamlit run app.py` as a web service.
- **ngrok** — run the app locally (`streamlit run app.py`) then `ngrok http 8501`
  for a quick temporary public tunnel, useful for demos without deploying anywhere.

## Notes

- The model is retrained on first launch automatically if `spam_model.pkl` is
  missing, so the app also works even if you only copy `app.py`,
  `train_model.py`, `SMSSpamCollection`, and `requirements.txt`.
- The preprocessing (lowercasing, stopword removal, lemmatization) is identical
  to the notebook, so predictions match what you saw there.
