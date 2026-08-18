"""
Trains the SMS Spam Classifier exactly as in SMS_Spam_Classification_NLP.ipynb
(TF-IDF + Multinomial Naive Bayes) and saves the fitted pipeline + metrics
so the Streamlit app can load them instantly without retraining.

Run once:  python train_model.py
"""

import json
import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 0. Make sure NLTK data is available (safe to call every run)
# ---------------------------------------------------------------------------
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Same preprocessing used in the notebook: lowercase, drop stopwords, lemmatize."""
    review = re.sub("^a-zA-Z0-9", " ", text)
    review = review.lower()
    review = review.split()
    review = [w for w in review if w not in STOPWORDS]
    review = [LEMMATIZER.lemmatize(w) for w in review]
    return " ".join(review)


def main():
    # -----------------------------------------------------------------
    # 1. Data gathering
    # -----------------------------------------------------------------
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["Label", "Msg"])
    df = df.dropna()

    # -----------------------------------------------------------------
    # 2. Preprocessing
    # -----------------------------------------------------------------
    df["Msg"] = df["Msg"].apply(clean_text)

    # -----------------------------------------------------------------
    # 3. Train / test split
    # -----------------------------------------------------------------
    x = df["Msg"]
    y = df["Label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=10
    )

    # -----------------------------------------------------------------
    # 4. Pipeline: TF-IDF + Multinomial Naive Bayes
    # -----------------------------------------------------------------
    text_mnb = Pipeline([("tfidf", TfidfVectorizer()), ("mnb", MultinomialNB())])
    text_mnb.fit(x_train, y_train)

    # -----------------------------------------------------------------
    # 5. Evaluation
    # -----------------------------------------------------------------
    y_pred_test = text_mnb.predict(x_test)
    y_pred_train = text_mnb.predict(x_train)

    metrics = {
        "train_accuracy": round(accuracy_score(y_train, y_pred_train) * 100, 2),
        "test_accuracy": round(accuracy_score(y_test, y_pred_test) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        "confusion_matrix_labels": sorted(y_test.unique().tolist()),
        "classification_report": classification_report(
            y_test, y_pred_test, output_dict=True
        ),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "class_counts": df["Label"].value_counts().to_dict(),
    }

    # -----------------------------------------------------------------
    # 6. Persist model + metrics
    # -----------------------------------------------------------------
    with open("spam_model.pkl", "wb") as f:
        pickle.dump(text_mnb, f)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Train accuracy: {metrics['train_accuracy']}%")
    print(f"Test accuracy:  {metrics['test_accuracy']}%")


if __name__ == "__main__":
    main()
