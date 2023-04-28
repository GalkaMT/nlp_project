import streamlit as st
import pandas as pd
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string
from nltk.stem import WordNetLemmatizer
import time

# Load the saved model
with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the stop words
stop_words = stopwords.words('english')
# Create a tokenizer
tokenizer = RegexpTokenizer(r'\w+')

def data_preprocessing(text: str) -> str:
    """preprocessing string: lowercase, removing html-tags, punctuation and stopwords

    Args:
        text (str): input string for preprocessing

    Returns:
        str: preprocessed string
    """

    text = text.lower()
    text = re.sub('<.*?>', '', text) # html tags
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    lemmatizer = WordNetLemmatizer()
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if not word.isdigit() and word not in stop_words]
    return ' '.join(tokens)


# Define the function to reset the input field


# Create the Streamlit app
def main():
    st.title('Sentiment Analysis App')
    user_input = st.text_input('Please enter your review:')
    if user_input:
        # Preprocess the user input
        preprocessed_input = data_preprocessing(user_input)
        # Vectorize the preprocessed input
        input_vector = vectorizer.transform([preprocessed_input])

        start_time = time.time()

        proba = loaded_model.predict_proba(input_vector)[:, 1]
        # Predict the sentiment using the loaded model
        #prediction = loaded_model.predict(input_vector)[0]
        prediction = round(proba[0])
        end_time = time.time()

        # Display the predicted sentiment
        if prediction == 0:
            st.write('The sentiment of your review is negative.')
            st.write('Predicted probability:', (1 - round(proba[0], 2))*100, '%')
        else:
            st.write('The sentiment of your review is positive.')
            st.write('Predicted probability:', (round(proba[0], 2))*100, '%')
        st.write('Processing time:', round(end_time - start_time, 4), 'seconds')

if __name__ == '__main__':
    main()