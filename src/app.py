import streamlit as st
import joblib
import re
import string
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -----------------------------
# Text Preprocessing Function
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📰 Fake News Detection")

st.write(
    """
    This application uses a **Machine Learning model** with **TF-IDF Vectorization**
    to classify a news article as **Fake** or **True**.
    """
)

news = st.text_area(
    "Paste a News Article",
    placeholder="Paste the complete news article here...",
    height=250
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("⚠️ Please enter a news article.")
    else:

        clean_news = preprocess_text(news)

        with st.spinner("Analyzing article..."):

            vector = vectorizer.transform([clean_news])

            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]

            confidence = max(probability) * 100

        st.divider()

        if prediction == 1:
            st.success("✅ This news is predicted to be **TRUE**.")
        else:
            st.error("❌ This news is predicted to be **FAKE**.")

        st.metric("Confidence", f"{confidence:.2f}%")

        st.progress(confidence / 100)

st.divider()

st.caption("Developed using Python, Scikit-learn, TF-IDF, and Streamlit")