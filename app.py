import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize stemmer once
ps = PorterStemmer()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load models efficiently with caching
@st.cache_resource
def load_models():
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_models()

st.title("Email/SMS Spam Classifier")

# Preprocessing function
@st.cache_data
def get_stop_words():
    """Cache stopwords for efficiency"""
    return set(stopwords.words('english'))

def transform_text(text):
    """
    Text preprocessing pipeline for spam classification.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string ready for vectorization
    """
    # Handle empty/None inputs
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric, stopwords, and punctuation
    stop_words = get_stop_words()
    tokens = [
        ps.stem(token) 
        for token in tokens 
        if token.isalnum() and token not in stop_words and token not in string.punctuation
    ]
    
    return ' '.join(tokens)

# User input
user_input = st.text_input("Enter your message below:", key="message_input")

# Process and predict
if user_input:
    # Preprocess
    processed_text = transform_text(user_input)
    
    if processed_text:
        # Vectorize
        vector_input = tfidf.transform([processed_text])
        
        # Predict
        result = model.predict(vector_input)[0]
        
        # Display result
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
    else:
        st.warning("Please enter meaningful text")

