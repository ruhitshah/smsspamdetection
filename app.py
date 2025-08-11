import streamlit as st
import joblib
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer
import numpy as np

# Load the saved model and vectorizer
model = joblib.load('sms_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to remove URLs, mentions and punctuation
def remove_punc(text):
    url_pattern = r'https?://\S+|www\.\S+'
    mention_pattern = r'@\w+'
    text = re.sub(url_pattern, '', text)
    text = re.sub(mention_pattern, '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text.strip()

# Remove stopwords using scikit-learn's built-in list
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])

# Stem words using PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

# Streamlit app interface
st.title("SMS Spam Detection")

# User input box for SMS text
sms_text = st.text_area("Enter the SMS to check if it's spam or not:")

# When the user submits the input
if st.button("Classify"):
    # Preprocess the input text
    sms_clean = remove_punc(sms_text)
    sms_clean = remove_stopwords(sms_clean)
    sms_clean = stem_words(sms_clean)

    # Vectorize the input text
    sms_vec = vectorizer.transform([sms_clean])

    # Predict using the model (get first element of array)
    prediction = model.predict(sms_vec)[0]

    # Display the result: handle both string and numeric labels
    if prediction == "spam" or prediction == 1:
        st.subheader("This message is SPAM!")
    else:
        st.subheader("This message is NOT SPAM.")
