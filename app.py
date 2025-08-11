import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved model and vectorizer
model = joblib.load('sms_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess the input text (similar to what was done for training data)
def remove_punc(text):
    url_pattern = r'https?://\S+|www\.\S+'
    mention_pattern = r'@\w+'
    text = re.sub(url_pattern, '', text)  
    text = re.sub(mention_pattern, '', text) 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text.strip()

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def remove_stopwords(text):
    # Lowercase words so comparisons match the list
    return " ".join([word for word in text.split()
                     if word.lower() not in ENGLISH_STOP_WORDS])

def stem_words(text):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

# Streamlit app interface
st.title("SMS Spam Detection")

# User input box for SMS text
sms_text = st.text_area("Enter the SMS to check if it's spam or not:")

# When the user submits the input
# When the user submits the input
if st.button("Classify"):
    # Preprocess the input text
    sms_text_pre = remove_punc(sms_text)
    sms_text_pre = remove_stopwords(sms_text_pre)
    sms_text_pre = stem_words(sms_text_pre)

    # Vectorize the input text
    sms_vec = vectorizer.transform([sms_text_pre])

    # Predict using the model
    prediction = model.predict(sms_vec)[0]  # get the first element of the array

    # Display the result based on the string label
    if prediction == "spam":
        st.subheader("This message is SPAM!")
    else:
        st.subheader("This message is NOT SPAM.")

