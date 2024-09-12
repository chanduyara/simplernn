import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,SimpleRNN
import streamlit as st

from keras.models import load_model
model1=load_model('simplernn_imdb.h5')

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model1.predict(preprocessed_input)
    # scalar_prediction = prediction[0][0].item()

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

review=st.text_input("Please enter movie review")

if st.button('Classify!'):

    preprocessed_input=preprocess_text(review)
    prediction=model1.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    sentiment,score=predict_sentiment(review)
    # This block of code is executed when the button is clicked
    st.write(f'Review: {review}')
    st.write(f'Sentiment: {sentiment}')
    st.write(f'score: {score}')

else:
    st.write('Please enter your review')


