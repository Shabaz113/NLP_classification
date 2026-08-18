import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


# ---------------------------------------------------------------------------
# Ensure model exists (train on first run if missing, e.g. fresh deploy)
# ---------------------------------------------------------------------------
def ensure_model():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return
    with st.spinner("First-time setup: training the model, this takes ~20 seconds..."):
        subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], check=True, cwd=BASE_DIR)


ensure_model()


# ---------------------------------------------------------------------------
# Load model + NLTK resources (cached so this only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    import nltk
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return model, metrics, stop_words, lemmatizer


model, metrics, STOPWORDS, LEMMATIZER = load_model()


def clean_text(text: str) -> str:
    review = re.sub("^a-zA-Z0-9", " ", text)
    review = review.lower()
    review = review.split()
    review = [w for w in review if w not in STOPWORDS]
    review = [LEMMATIZER.lemmatize(w) for w in review]
    return " ".join(review)


def predict(text: str):
    cleaned = clean_text(text)
    label = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    prob_dict = dict(zip(classes, proba))
    return label, prob_dict


# ---------------------------------------------------------------------------
# Sidebar — model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("**Pipeline:** TF-IDF + Multinomial Naive Bayes")
    col1, col2 = st.columns(2)
    col1.metric("Test Accuracy", f"{metrics['test_accuracy']}%")
    col2.metric("Train Accuracy", f"{metrics['train_accuracy']}%")

    st.markdown("---")
    st.markdown("**Dataset**")
    st.write(f"Ham: {metrics['class_counts'].get('ham', 0):,}")
    st.write(f"Spam: {metrics['class_counts'].get('spam', 0):,}")
    st.write(f"Train rows: {metrics['n_train']:,} · Test rows: {metrics['n_test']:,}")

    with st.expander("Classification report"):
        report_df = pd.DataFrame(metrics["classification_report"]).T
        st.dataframe(report_df.round(3))

    with st.expander("Confusion matrix (test set)"):
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=[f"Actual: {l}" for l in metrics["confusion_matrix_labels"]],
            columns=[f"Predicted: {l}" for l in metrics["confusion_matrix_labels"]],
        )
        st.dataframe(cm)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("📩 SMS Spam Classifier")
st.caption("Paste any SMS/text message below and the model will tell you if it's Spam or Ham (legitimate).")

if "message" not in st.session_state:
    st.session_state.message = ""

examples = {
    "🎉 Prize scam": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 now.",
    "💬 Casual chat": "Hey are we still on for lunch tomorrow? Let me know what time works for you.",
    "📈 Loan spam": "URGENT! You have qualified for a $5000 loan. No credit check needed. Reply YES to claim now.",
    "🏠 Everyday msg": "Can you pick up milk on your way home? Also don't forget mom's birthday is Sunday.",
}

st.write("**Try an example:**")
cols = st.columns(len(examples))
for col, (label, text) in zip(cols, examples.items()):
    if col.button(label, use_container_width=True):
        st.session_state.message = text

message = st.text_area(
    "Message",
    value=st.session_state.message,
    height=120,
    placeholder="Type or paste an SMS message here...",
    key="message_input",
)

predict_clicked = st.button("🔍 Classify Message", type="primary", use_container_width=True)

if predict_clicked:
    if not message.strip():
        st.warning("Please enter a message to classify.")
    else:
        label, prob_dict = predict(message)
        spam_prob = prob_dict.get("spam", 0)
        ham_prob = prob_dict.get("ham", 0)

        if label == "spam":
            st.error(f"🚨 **This message looks like SPAM** (confidence: {spam_prob*100:.1f}%)")
        else:
            st.success(f"✅ **This message looks like HAM (legitimate)** (confidence: {ham_prob*100:.1f}%)")

        st.write("**Probability breakdown**")
        prob_df = pd.DataFrame(
            {"Class": list(prob_dict.keys()), "Probability": list(prob_dict.values())}
        ).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Class"))

        with st.expander("See preprocessed text sent to the model"):
            st.code(clean_text(message))

st.markdown("---")
st.caption("Built from the SMS_Spam_Classification_NLP notebook · TF-IDF + Multinomial Naive Bayes · scikit-learn + NLTK")
