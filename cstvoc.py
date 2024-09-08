import streamlit as st
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

# Preprocessing function (basic text cleaning)
def preprocess_arabic_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    arabic_diacritics = re.compile(r'[ً-ْ]')
    text = re.sub(arabic_diacritics, '', text)
    return text

# Load the trained SVM model and vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Add the image at the top
st.image('Sentiment.PNG', caption='Customer Sentiment Analyzer', use_column_width=True)

# Add a title with emoji and styling
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💬 Customer Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #777;'>Unlock the Voice of Your Customers Instantly! Enter the text below to analyze customer sentiment. </h4>", unsafe_allow_html=True)

# Custom CSS for larger font in text area
st.markdown("""
    <style>
    .text_input_area label {
        font-size: 20px;
        color: #4CAF50;
    }
    .text_input_area textarea {
        font-size: 18px;
        height: 150px;
    }
    </style>
""", unsafe_allow_html=True)

# Add a wide text input area with larger font
st.markdown("<div class='text_input_area'>", unsafe_allow_html=True)
user_input = st.text_area("📝 Enter the feedback you want to analyze:", height=200, placeholder="Type customer feedback here...")
st.markdown("</div>", unsafe_allow_html=True)

# Add a predict button with a custom style
if st.button("🔍 Analyze Sentiment"):
    if user_input:
        # Preprocess the user input
        cleaned_input = preprocess_arabic_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Make predictions with the SVM model
        prediction_svm = svm_model.predict(vectorized_input)[0]
        
        # Map the prediction to sentiment labels and colors
        if prediction_svm == 1:
            sentiment = "Positive 😊"
            st.markdown("<h3 style='text-align: center; color: green;'>🟢 Positive Feedback</h3>", unsafe_allow_html=True)
        elif prediction_svm == -1:
            sentiment = "Negative 😔"
            st.markdown("<h3 style='text-align: center; color: red;'>🔴 Negative Feedback</h3>", unsafe_allow_html=True)
        else:
            sentiment = "Neutral 😐"
            st.markdown("<h3 style='text-align: center; color: gray;'>⚪ Neutral Feedback</h3>", unsafe_allow_html=True)
        
        # Display the result with a colored message
        st.success(f"The feedback is classified as: **{sentiment}**")
    else:
        st.warning("⚠️ Please enter some feedback to analyze.")
