import streamlit as st
import joblib
import re
import string

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("📰 Fake News Detection")

news = st.text_area("Paste your news article")

if st.button("Predict"):
    news = wordopt(news)
    vector = vectorizer.transform([news])

    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]

    confidence = max(prob) * 100

    if pred == 1:
        st.success("✅ True News")
    else:
        st.error("❌ Fake News")

    st.write(f"Confidence: {confidence:.2f}%")