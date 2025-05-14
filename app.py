import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_input(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

st.title("🧠 Emotion Detection from Social Media Text")

user_input = st.text_area("Enter your message:")

if st.button("Analyze Emotion"):
    clean_text = preprocess_input(user_input)
    vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized)[0]
    st.success(f"Detected Emotion: **{prediction.upper()}**")
